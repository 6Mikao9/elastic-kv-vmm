"""GPU tests for true VMM mapping and lifetime; require the compiled native library."""

import gc
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from elastic_kv.backing import (
    DEFAULT_CHUNK_BYTES,
    Backing,
    _resolve_broker_path,
)
from elastic_kv.global_broker import GlobalAdmissionBroker


class _FakeLibrary:
    def __init__(self, reserved_units=2, *, partial_failure=False, close_failure=False):
        self.mapped = 0
        self.calls = []
        self.reserved_units = reserved_units
        self.partial_failure = partial_failure
        self.close_failure = close_failure
        self.failure = None

    def ev_create(self, *_):
        return 1

    def ev_base(self, _):
        return 1 << 40

    def ev_backed(self, _):
        return self.mapped

    def ev_reserved(self, _):
        return self.reserved_units

    def ev_granularity(self, _):
        return 1

    def ev_ensure(self, _, _ids, count):
        if self.partial_failure:
            self.mapped += max(0, count - 1)
            self.calls.append(("ensure-partial", count))
            self.failure = b"injected partial map failure"
            return 1
        self.mapped += count
        self.calls.append(("ensure", count))
        return 0

    def ev_precommit(self, _, _ids, count):
        if self.partial_failure:
            self.mapped += max(0, count - 1)
            self.calls.append(("precommit-partial", count))
            self.failure = b"injected partial map failure"
            return 1
        self.mapped += count
        self.calls.append(("precommit", count))
        return 0

    def ev_drop(self, _, _ids, count):
        self.mapped -= count
        self.calls.append(("drop", count))
        return 0

    def ev_close(self, _):
        if self.close_failure:
            self.failure = b"injected close failure"
            return 1
        self.mapped = 0
        return 0

    def ev_error(self):
        return self.failure or b""


class GlobalBrokerBackingTest(unittest.TestCase):
    def test_fair_broker_environment_requires_a_broker_path(self):
        chunk = 1 << 20
        with patch.dict(
            os.environ,
            {
                "EKV_GLOBAL_BROKER_FAIR_MODE": "enforce",
                "EKV_GLOBAL_BROKER_TENANT": "model-a",
            },
            clear=True,
        ), patch(
            "elastic_kv.backing._load_library", return_value=_FakeLibrary(2 * chunk)
        ):
            with self.assertRaisesRegex(ValueError, "enabled broker path"):
                Backing(2 * chunk, chunk, 0, library="missing-broker.so")

    def test_broker_path_can_be_partitioned_by_physical_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = os.path.join(temporary, "brokers")
            resolved = _resolve_broker_path(directory + os.sep, "cuda:0")
            self.assertEqual(
                resolved,
                os.path.join(directory, "elastic-kv-cuda_0.sqlite"),
            )
            templated = _resolve_broker_path(
                os.path.join(temporary, "gpu-{gpu}.sqlite"), "GPU-123"
            )
            self.assertEqual(
                templated,
                os.path.join(temporary, "gpu-GPU-123.sqlite"),
            )

    def test_backing_leases_before_mapping_and_releases_after_unmap(self):
        chunk = 1 << 20
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "gpu0.sqlite")
            environment = {
                "EKV_GLOBAL_BROKER_PATH": path,
                "EKV_GLOBAL_BROKER_MIB": "2",
                "EKV_GLOBAL_BROKER_STALE_SECONDS": "30",
            }
            with patch.dict(os.environ, environment, clear=False):
                with patch(
                    "elastic_kv.backing._load_library",
                    side_effect=[
                        _FakeLibrary(2 * chunk), _FakeLibrary(2 * chunk)
                    ],
                ):
                    first = Backing(2 * chunk, chunk, 0, library="first.so")
                    second = Backing(2 * chunk, chunk, 0, library="second.so")
                observer = GlobalAdmissionBroker(
                    path, 2 * chunk, owner_id="observer", heartbeat_seconds=None
                )
                try:
                    self.assertEqual(first.ensure([0]), 1)
                    self.assertEqual(second.ensure([0]), 1)
                    self.assertEqual(observer.available_bytes(), 0)
                    with self.assertRaisesRegex(
                        RuntimeError, "global VMM capacity exhausted"
                    ):
                        first.ensure([1])
                    self.assertEqual(second.reclaim([0], synchronize=False), 1)
                    self.assertEqual(first.ensure([1]), 1)
                    self.assertEqual(first.metrics["global_broker_denials"], 1)
                finally:
                    first.close()
                    second.close()
                    observer.close()
            verifier = GlobalAdmissionBroker(
                path, 2 * chunk, owner_id="verifier", heartbeat_seconds=None
            )
            try:
                self.assertEqual(verifier.available_bytes(), 2 * chunk)
            finally:
                verifier.close()

    def test_partial_native_map_failure_retains_lease_until_close(self):
        chunk = 1 << 20
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "gpu0.sqlite")
            environment = {
                "EKV_GLOBAL_BROKER_PATH": path,
                "EKV_GLOBAL_BROKER_MIB": "2",
            }
            library = _FakeLibrary(2 * chunk, partial_failure=True)
            with patch.dict(os.environ, environment, clear=False), patch(
                "elastic_kv.backing._load_library", return_value=library
            ):
                first = Backing(2 * chunk, chunk, 0, library="partial.so")
            observer = GlobalAdmissionBroker(
                path, 2 * chunk, owner_id="observer", heartbeat_seconds=None
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "partial map failure"):
                    first.ensure([0, 1])
                self.assertEqual(observer.available_bytes(), 0)
                first.close()
                self.assertEqual(observer.available_bytes(), 2 * chunk)
            finally:
                first.close()
                observer.close()

    def test_temporary_claim_blocks_same_backing_growth_until_released(self):
        chunk = 1 << 20
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "gpu0.sqlite")
            environment = {
                "EKV_GLOBAL_BROKER_PATH": path,
                "EKV_GLOBAL_BROKER_MIB": "2",
            }
            with patch.dict(os.environ, environment, clear=False), patch(
                "elastic_kv.backing._load_library", return_value=_FakeLibrary(2 * chunk)
            ):
                backing = Backing(2 * chunk, chunk, 0, library="claim.so")
            observer = GlobalAdmissionBroker(
                path, 2 * chunk, owner_id="observer", heartbeat_seconds=None
            )
            try:
                self.assertTrue(
                    backing.claim_broker_capacity("request-0", 2 * chunk)
                )
                self.assertEqual(observer.available_bytes(), 0)
                self.assertEqual(
                    backing.shrink_broker_claim("request-0", chunk), chunk
                )
                self.assertEqual(observer.available_bytes(), chunk)
                with self.assertRaisesRegex(
                    RuntimeError, "global VMM capacity exhausted"
                ):
                    backing.ensure([0, 1])
                self.assertEqual(backing.ensure([1]), 1)
                self.assertEqual(
                    backing.release_broker_claim("request-0"), chunk
                )
                self.assertEqual(backing.ensure([0]), 1)
                self.assertEqual(
                    backing.metrics["global_broker_denials"], 1
                )
                self.assertEqual(
                    backing.metrics["global_broker_claim_releases"], 1
                )
                self.assertEqual(
                    backing.metrics["global_broker_claim_shrunk_bytes"], chunk
                )
            finally:
                backing.close()
                observer.close()

    def test_failed_native_close_retains_lease_until_teardown_retry(self):
        chunk = 1 << 20
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "gpu0.sqlite")
            environment = {
                "EKV_GLOBAL_BROKER_PATH": path,
                "EKV_GLOBAL_BROKER_MIB": "2",
            }
            library = _FakeLibrary(2 * chunk, close_failure=True)
            with patch.dict(os.environ, environment, clear=False), patch(
                "elastic_kv.backing._load_library", return_value=library
            ):
                backing = Backing(2 * chunk, chunk, 0, library="close-failure.so")
            observer = GlobalAdmissionBroker(
                path, 2 * chunk, owner_id="observer", heartbeat_seconds=None
            )
            try:
                backing.ensure([0])
                with self.assertRaisesRegex(RuntimeError, "close failure"):
                    backing.close()
                self.assertEqual(observer.available_bytes(), chunk)
                library.close_failure = False
                backing.close()
                self.assertEqual(observer.available_bytes(), 2 * chunk)
            finally:
                library.close_failure = False
                backing.close()
                observer.close()

    def test_garbage_collection_closes_native_backing_and_broker(self):
        chunk = 1 << 20
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "gpu0.sqlite")
            environment = {
                "EKV_GLOBAL_BROKER_PATH": path,
                "EKV_GLOBAL_BROKER_MIB": "2",
            }
            library = _FakeLibrary(2 * chunk)
            with patch.dict(os.environ, environment, clear=False), patch(
                "elastic_kv.backing._load_library", return_value=library
            ):
                def allocate_and_drop():
                    backing = Backing(2 * chunk, chunk, 0, library="gc.so")
                    backing.ensure([0])

                allocate_and_drop()
            observer = GlobalAdmissionBroker(
                path, 2 * chunk, owner_id="observer", heartbeat_seconds=None
            )
            try:
                gc.collect()
                self.assertEqual(observer.available_bytes(), 2 * chunk)
                self.assertEqual(library.mapped, 0)
            finally:
                observer.close()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class BackingTest(unittest.TestCase):
    def test_release_and_recommit_one_hole(self):
        chunk = DEFAULT_CHUNK_BYTES
        torch.cuda.set_device(0)
        backing = Backing(5 * chunk - 17, chunk_bytes=chunk, device=0)
        try:
            self._exercise(backing, chunk)
        finally:
            backing.close()

    def _exercise(self, backing, chunk):
        # Chunk 3 remains unmapped. Only explicitly mapped slices are accessed.
        self.assertEqual(backing.ensure([0, 1, 2, 4]), 4)
        self.assertEqual(backing.reserved_bytes, 5 * chunk)
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        raw = backing.raw
        base = raw.data_ptr()
        self.assertEqual(base, backing.base)
        self.assertEqual(raw.numel(), 5 * chunk - 17)
        views = {i: raw[i * chunk : (i + 1) * chunk] for i in (0, 1, 2, 4)}
        for i, value in ((0, 11), (1, 22), (2, 33), (4, 55)):
            views[i].fill_(value)

        self.assertEqual(backing.reclaim([1, 3, 1]), 1)
        self.assertEqual(backing.mapped_chunks, {0, 2, 4})
        self.assertEqual(backing.backed_bytes, 3 * chunk)
        for i, value in ((0, 11), (2, 33), (4, 55)):
            self.assertTrue(views[i].eq(value).all().item())

        # Precommit maps the chunk but leaves zeroing on the initialization
        # stream. ``ensure`` is still the readiness barrier before GPU use.
        self.assertEqual(backing.precommit([1]), 1)
        self.assertEqual(backing.mapped_chunks, {0, 2, 4})
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        self.assertEqual(backing.ensure([1]), 0)
        self.assertEqual(backing.mapped_chunks, {0, 1, 2, 4})
        self.assertEqual(backing.backed_bytes, 4 * chunk)
        self.assertEqual(backing.raw.data_ptr(), base)
        self.assertEqual(views[1].data_ptr(), base + chunk)
        self.assertEqual(torch.count_nonzero(views[1]).item(), 0)
        for i, value in ((0, 11), (2, 33), (4, 55)):
            self.assertEqual(views[i].data_ptr(), base + i * chunk)
            self.assertTrue(views[i].eq(value).all().item())


if __name__ == "__main__":
    unittest.main()
