// CUDA backing for a single stable virtual address range per object.
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (message ? message : std::to_string(result)));
}

class CurrentContext {
public:
    explicit CurrentContext(CUcontext context) {
        check(cuCtxPushCurrent(context), "cuCtxPushCurrent");
    }

    ~CurrentContext() {
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
    }

    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
};

struct Chunk {
    CUmemGenericAllocationHandle handle{};
    bool owned = false;
    bool mapped = false;
};

class Backing {
public:
    ~Backing() { close(); }

    void initialize(int ordinal, uint64_t logical_bytes, uint64_t chunk_bytes) {
        if (logical_bytes == 0 || chunk_bytes == 0 ||
            logical_bytes > std::numeric_limits<size_t>::max() ||
            chunk_bytes > std::numeric_limits<size_t>::max()) {
            throw std::invalid_argument("capacity and chunk size must be positive size_t values");
        }
        chunk_bytes_ = static_cast<size_t>(chunk_bytes);
        const uint64_t chunks = logical_bytes / chunk_bytes +
                                (logical_bytes % chunk_bytes != 0);
        if (chunks > std::numeric_limits<size_t>::max() / chunk_bytes_) {
            throw std::invalid_argument("aligned capacity overflows size_t");
        }
        reserved_bytes_ = static_cast<size_t>(chunks) * chunk_bytes_;
        check(cuInit(0), "cuInit");
        check(cuDeviceGet(&device_, ordinal), "cuDeviceGet");
        int supported = 0;
        check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                   device_), "query VMM support");
        if (!supported) {
            throw std::runtime_error("device does not support CUDA VMM");
        }

        prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop_.location.id = device_;
        check(cuMemGetAllocationGranularity(&granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
        if (chunk_bytes_ % granularity_ != 0) {
            throw std::invalid_argument("chunk_bytes must be a multiple of VMM granularity " +
                                        std::to_string(granularity_));
        }
        check(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
        CurrentContext current(context_);
        check(cuMemAddressReserve(&base_, reserved_bytes_, granularity_, 0, 0),
              "cuMemAddressReserve");
        check(cuStreamCreate(&zero_stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    }

    void ensure(const uint64_t* ids, size_t count) {
        ensure_impl(ids, count, true);
    }

    void precommit(const uint64_t* ids, size_t count) {
        ensure_impl(ids, count, false);
    }

private:
    void ensure_impl(const uint64_t* ids, size_t count, bool wait_for_zero) {
        const auto pending = normalize(ids, count, false);
        if (pending.empty()) {
            if (wait_for_zero) {
                wait_for_initialization();
            }
            return;
        }
        CurrentContext current(context_);
        size_t started = 0;
        try {
            for (uint64_t id : pending) {
                auto& chunk = chunks_.try_emplace(id).first->second;
                ++started;
                check(cuMemCreate(&chunk.handle, chunk_bytes_, &prop_, 0), "cuMemCreate");
                chunk.owned = true;
                check(cuMemMap(address(id), chunk_bytes_, 0, chunk.handle, 0), "cuMemMap");
                chunk.mapped = true;
            }

            CUmemAccessDesc access{};
            access.location = prop_.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            for (size_t start = 0; start < pending.size();) {
                size_t end = start + 1;
                while (end < pending.size() && pending[end] == pending[end - 1] + 1) {
                    ++end;
                }
                const size_t bytes = (end - start) * chunk_bytes_;
                const CUdeviceptr begin = address(pending[start]);
                check(cuMemSetAccess(begin, bytes, &access, 1), "cuMemSetAccess");
                check(cuMemsetD8Async(begin, 0, bytes, zero_stream_), "zero new backing");
                zero_pending_ = true;
                start = end;
            }
            if (wait_for_zero) {
                wait_for_initialization();
            }
        } catch (...) {
            // Do not release a range if we cannot establish completion of
            // its zeroing. The caller must stop using a poisoned backing.
            if (zero_pending_ && cuStreamSynchronize(zero_stream_) != CUDA_SUCCESS) {
                poisoned_ = true;
                throw;
            }
            zero_pending_ = false;
            while (started > 0) {
                auto it = chunks_.find(pending[--started]);
                if (release(it->first, it->second) == CUDA_SUCCESS) {
                    chunks_.erase(it);
                } else {
                    poisoned_ = true;
                }
            }
            throw;
        }
    }

    void wait_for_initialization() {
        if (!zero_pending_) {
            return;
        }
        check(cuStreamSynchronize(zero_stream_), "wait for backing initialization");
        zero_pending_ = false;
    }

public:
    // Caller must establish that no stream, graph replay, transfer, or other
    // in-flight user can access these addresses when they are decommitted.
    void drop(const uint64_t* ids, size_t count) {
        const auto pending = normalize(ids, count, true);
        if (pending.empty()) {
            return;
        }
        CurrentContext current(context_);
        wait_for_initialization();
        for (uint64_t id : pending) {
            auto it = chunks_.find(id);
            const CUresult result = release(id, it->second);
            if (result != CUDA_SUCCESS) {
                // Earlier successful drops cannot be undone without restoring
                // data. Reject future changes and let the caller tear down.
                poisoned_ = true;
                check(result, "decommit backing");
            }
            chunks_.erase(it);
        }
    }

    CUresult close() noexcept {
        if (!context_) {
            return CUDA_SUCCESS;
        }
        CUresult first = cuCtxPushCurrent(context_);
        auto record = [&first](CUresult result) {
            if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) {
                first = result;
            }
        };
        if (first == CUDA_SUCCESS) {
            record(cuCtxSynchronize());
            for (auto& entry : chunks_) {
                record(release(entry.first, entry.second));
            }
            if (base_) {
                record(cuMemAddressFree(base_, reserved_bytes_));
                base_ = 0;
            }
            if (zero_stream_) {
                record(cuStreamDestroy(zero_stream_));
                zero_stream_ = nullptr;
            }
            CUcontext popped = nullptr;
            record(cuCtxPopCurrent(&popped));
        }
        record(cuDevicePrimaryCtxRelease(device_));
        context_ = nullptr;
        return first;
    }

    uint64_t base() const { return base_; }
    uint64_t granularity() const { return granularity_; }
    uint64_t reserved() const { return reserved_bytes_; }
    uint64_t backed() const {
        if (!poisoned_) {
            return chunks_.size() * chunk_bytes_;
        }
        size_t committed = 0;
        for (const auto& entry : chunks_) {
            committed += entry.second.owned || entry.second.mapped;
        }
        return committed * chunk_bytes_;
    }

private:
    CUdeviceptr address(uint64_t id) const { return base_ + id * chunk_bytes_; }

    std::vector<uint64_t> normalize(const uint64_t* ids, size_t count, bool mapped) const {
        if (poisoned_) {
            throw std::runtime_error("backing is poisoned after a partial CUDA failure");
        }
        if (count != 0 && ids == nullptr) {
            throw std::invalid_argument("chunk index array is null");
        }
        std::vector<uint64_t> pending;
        pending.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t id = ids[i];
            if (id >= reserved_bytes_ / chunk_bytes_) {
                throw std::out_of_range("chunk index outside reserved range");
            }
            if ((chunks_.find(id) != chunks_.end()) == mapped) {
                pending.push_back(id);
            }
        }
        std::sort(pending.begin(), pending.end());
        pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
        return pending;
    }

    CUresult release(uint64_t id, Chunk& chunk) noexcept {
        if (chunk.mapped) {
            const CUresult result = cuMemUnmap(address(id), chunk_bytes_);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            chunk.mapped = false;
        }
        if (chunk.owned) {
            const CUresult result = cuMemRelease(chunk.handle);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            chunk.owned = false;
        }
        return CUDA_SUCCESS;
    }

    CUcontext context_ = nullptr;
    CUdevice device_{};
    CUdeviceptr base_ = 0;
    CUstream zero_stream_ = nullptr;
    CUmemAllocationProp prop_{};
    size_t reserved_bytes_ = 0;
    size_t chunk_bytes_ = 0;
    size_t granularity_ = 0;
    bool zero_pending_ = false;
    bool poisoned_ = false;
    std::map<uint64_t, Chunk> chunks_;
};

thread_local std::string last_error;

Backing& require(void* handle) {
    if (!handle) {
        throw std::invalid_argument("null backing handle");
    }
    return *static_cast<Backing*>(handle);
}

template <typename Result, typename Action>
Result call(Result error, Action action) {
    try {
        last_error.clear();
        return action();
    } catch (const std::exception& e) {
        last_error = e.what();
        return error;
    } catch (...) {
        last_error = "unknown C++ error";
        return error;
    }
}

constexpr uint64_t kInvalid = std::numeric_limits<uint64_t>::max();

}  // namespace

extern "C" {

void* ev_create(int device, uint64_t capacity_bytes, uint64_t chunk_bytes) {
    return call(static_cast<void*>(nullptr), [&] {
        auto backing = std::make_unique<Backing>();
        backing->initialize(device, capacity_bytes, chunk_bytes);
        return static_cast<void*>(backing.release());
    });
}

int ev_ensure(void* handle, const uint64_t* indices, size_t count) {
    return call(-1, [&] {
        require(handle).ensure(indices, count);
        return 0;
    });
}

int ev_precommit(void* handle, const uint64_t* indices, size_t count) {
    return call(-1, [&] {
        require(handle).precommit(indices, count);
        return 0;
    });
}

int ev_drop(void* handle, const uint64_t* indices, size_t count) {
    return call(-1, [&] {
        require(handle).drop(indices, count);
        return 0;
    });
}

uint64_t ev_base(void* handle) { return call(kInvalid, [&] { return require(handle).base(); }); }
uint64_t ev_backed(void* handle) { return call(kInvalid, [&] { return require(handle).backed(); }); }
uint64_t ev_reserved(void* handle) { return call(kInvalid, [&] { return require(handle).reserved(); }); }
uint64_t ev_granularity(void* handle) {
    return call(kInvalid, [&] { return require(handle).granularity(); });
}

const char* ev_error() { return last_error.c_str(); }

int ev_close(void* handle) {
    return call(-1, [&] {
        std::unique_ptr<Backing> backing(static_cast<Backing*>(handle));
        if (backing) {
            check(backing->close(), "close backing");
        }
        return 0;
    });
}

}  // extern "C"
