"""Cross-process leases for aggregate CUDA VMM physical backing.

The broker coordinates callers on one host and one device.  It is deliberately
limited to chunks that a ``Backing`` is about to map plus optional temporary
capacity claims. Model weights and allocator caches remain outside its
capacity; a prefill claim reserves admission budget without owning the later
PyTorch activation allocation.
"""

from __future__ import annotations

import ctypes
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LeaseDecision:
    admit: bool
    reason: str
    required_bytes: int
    outstanding_bytes: int
    available_bytes: int
    acquired_keys: tuple[str, ...] = ()
    tenant_id: str | None = None
    tenant_usage_bytes: int = 0
    fair_target_bytes: int | None = None
    starved_tenants: tuple[str, ...] = ()
    retry_count: int = 0
    waited_seconds: float = 0.0


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _process_exists(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER and ERROR_NOT_FOUND prove that the PID is
        # absent. Access-denied and transient failures do not prove death, so
        # they retain the lease rather than risking cross-process overcommit.
        return ctypes.get_last_error() not in {87, 1168}
    kernel32.CloseHandle(handle)
    return True


FAIR_MODES = ("off", "observe", "enforce")
MAX_RESERVE_WAIT_SECONDS = 5.0


class GlobalAdmissionBroker:
    """SQLite-backed aggregate VMM leases with optional weighted fairness.

    Every caller has a unique owner token.  A stale owner is reclaimed only
    after its recorded PID is demonstrably gone, or when the machine boot ID
    changed.  A live but hung process therefore keeps its safety reservation;
    the broker can leak capacity, but it never invalidates backing that may
    still be mapped by that process.  Fair share is cooperative: an over-share
    owner is blocked from growth while another tenant has a live demand, but
    the broker never unmaps that owner's live KV itself.
    """

    def __init__(
        self,
        path,
        capacity_bytes,
        *,
        owner_id: str | None = None,
        stale_after_seconds: float = 120,
        heartbeat_seconds: float | None = None,
        timeout_seconds: float = 5,
        process_exists: Callable[[int], bool] = _process_exists,
        boot_id: str | None = None,
        resource_id: str | None = None,
        tenant_id: str | None = None,
        tenant_weight: float = 1.0,
        fair_mode: str = "off",
        demand_ttl_seconds: float = 10.0,
        reserve_wait_seconds: float = 0.0,
    ):
        try:
            capacity = int(capacity_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("global broker capacity must be an integer") from exc
        if isinstance(capacity_bytes, bool) or capacity < 1:
            raise ValueError("global broker capacity must be positive")
        if stale_after_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("broker timestamps and timeout must be positive")
        if heartbeat_seconds is not None and heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        if resource_id is not None and not str(resource_id).strip():
            raise ValueError("broker resource_id must be nonempty")
        if fair_mode not in FAIR_MODES:
            raise ValueError("broker fair mode must be off, observe, or enforce")
        if tenant_id is not None and (
            not isinstance(tenant_id, str) or not tenant_id.strip() or len(tenant_id) > 128
        ):
            raise ValueError("broker tenant_id must be a short nonempty string")
        if fair_mode != "off" and tenant_id is None:
            raise ValueError("fair broker modes require a stable tenant_id")
        if (
            isinstance(tenant_weight, bool)
            or not isinstance(tenant_weight, (int, float))
            or not math.isfinite(tenant_weight)
            or not 0 < tenant_weight <= 1e6
        ):
            raise ValueError("broker tenant_weight must be finite and in (0, 1e6]")
        if (
            isinstance(demand_ttl_seconds, bool)
            or not isinstance(demand_ttl_seconds, (int, float))
            or not math.isfinite(demand_ttl_seconds)
            or demand_ttl_seconds <= 0
        ):
            raise ValueError("broker demand TTL must be finite and positive")
        if (
            isinstance(reserve_wait_seconds, bool)
            or not isinstance(reserve_wait_seconds, (int, float))
            or not math.isfinite(reserve_wait_seconds)
            or reserve_wait_seconds < 0
            or reserve_wait_seconds > MAX_RESERVE_WAIT_SECONDS
        ):
            raise ValueError(
                "broker reserve wait must be finite and within [0, 5] seconds"
            )
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = capacity
        self.owner_id = owner_id or f"{os.getpid()}-{uuid.uuid4().hex}"
        self.stale_after_seconds = float(stale_after_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._process_exists = process_exists
        self._boot_id = boot_id if boot_id is not None else _boot_id()
        self.resource_id = None if resource_id is None else str(resource_id)
        self.tenant_id = tenant_id
        self.tenant_weight = float(tenant_weight)
        self.fair_mode = fair_mode
        self.demand_ttl_seconds = float(demand_ttl_seconds)
        self.reserve_wait_seconds = float(reserve_wait_seconds)
        self._closed = False
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._local = threading.local()
        self._connections = set()
        self._connections_lock = threading.Lock()
        try:
            self._initialize()
        except BaseException:
            self._close_all_connections()
            raise
        if heartbeat_seconds is not None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(heartbeat_seconds,),
                name=f"ekv-broker-{self.owner_id[-8:]}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                str(self.path),
                timeout=self.timeout_seconds,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.execute(
                f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}"
            )
            self._local.connection = connection
            with self._connections_lock:
                self._connections.add(connection)
        return connection

    def _close_thread_connection(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            return
        self._local.connection = None
        with self._connections_lock:
            self._connections.discard(connection)
        if connection.in_transaction:
            connection.rollback()
        connection.close()

    def _close_all_connections(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()
            except sqlite3.Error:
                pass

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            self._close_thread_connection()
            raise

    def _initialize(self) -> None:
        now = time.time()
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owners "
                "(owner TEXT PRIMARY KEY, pid INTEGER NOT NULL, boot_id TEXT, "
                "last_seen REAL NOT NULL, tenant_id TEXT)"
            )
            owner_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(owners)").fetchall()
            }
            if "tenant_id" not in owner_columns:
                connection.execute("ALTER TABLE owners ADD COLUMN tenant_id TEXT")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS leases "
                "(owner TEXT NOT NULL, lease_key TEXT NOT NULL, bytes INTEGER NOT NULL, "
                "PRIMARY KEY(owner, lease_key))"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS leases_lease_key_unique "
                "ON leases(lease_key)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS claims "
                "(owner TEXT NOT NULL, claim_key TEXT NOT NULL, "
                "claim_kind TEXT NOT NULL, bytes INTEGER NOT NULL, "
                "PRIMARY KEY(owner, claim_key))"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS claims_claim_key_unique "
                "ON claims(claim_key)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tenants "
                "(tenant_id TEXT PRIMARY KEY, weight REAL NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS demands "
                "(owner TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, "
                "requested_bytes INTEGER NOT NULL, expires_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS demands_tenant_idx "
                "ON demands(tenant_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS demands_expiry_idx "
                "ON demands(expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS owners_tenant_idx "
                "ON owners(tenant_id)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('capacity_bytes', ?)",
                (str(self.capacity_bytes),),
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='capacity_bytes'"
            ).fetchone()
            if row is None or int(row[0]) != self.capacity_bytes:
                raise ValueError(
                    "global broker capacity does not match the existing state file"
                )
            if self.resource_id is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) "
                    "VALUES('resource_id', ?)",
                    (self.resource_id,),
                )
                resource = connection.execute(
                    "SELECT value FROM metadata WHERE key='resource_id'"
                ).fetchone()
                if resource is None or resource[0] != self.resource_id:
                    raise ValueError(
                        "global broker state file belongs to a different physical GPU"
                    )
            fair_row = connection.execute(
                "SELECT value FROM metadata WHERE key='fair_mode'"
            ).fetchone()
            if fair_row is None:
                active_leases = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM leases"
                    ).fetchone()[0]
                )
                if self.fair_mode != "off" and active_leases:
                    raise ValueError(
                        "cannot enable fair share while legacy leases are active"
                    )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('fair_mode', ?)",
                    (self.fair_mode,),
                )
            elif fair_row[0] != self.fair_mode:
                raise ValueError(
                    "global broker fair mode does not match the existing state file"
                )
            if self.tenant_id is not None:
                connection.execute(
                    "INSERT INTO tenants(tenant_id, weight) VALUES(?, ?) "
                    "ON CONFLICT(tenant_id) DO NOTHING",
                    (self.tenant_id, self.tenant_weight),
                )
                stored_weight = connection.execute(
                    "SELECT weight FROM tenants WHERE tenant_id=?", (self.tenant_id,)
                ).fetchone()[0]
                if not math.isclose(
                    float(stored_weight), self.tenant_weight,
                    rel_tol=1e-9, abs_tol=1e-12,
                ):
                    raise ValueError(
                        "global broker tenant weight does not match shared configuration"
                    )
            connection.execute(
                "INSERT INTO owners(owner, pid, boot_id, last_seen, tenant_id) "
                "VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(owner) DO UPDATE SET last_seen=excluded.last_seen, "
                "tenant_id=excluded.tenant_id",
                (self.owner_id, os.getpid(), self._boot_id, now, self.tenant_id),
            )
            connection.commit()

    def _prune_dead_owners(self, connection: sqlite3.Connection, now: float) -> None:
        cutoff = now - self.stale_after_seconds
        rows = connection.execute(
            "SELECT owner, pid, boot_id FROM owners "
            "WHERE owner != ? AND last_seen < ?",
            (self.owner_id, cutoff),
        ).fetchall()
        for owner, pid, owner_boot_id in rows:
            boot_changed = (
                self._boot_id is not None
                and owner_boot_id is not None
                and owner_boot_id != self._boot_id
            )
            if not boot_changed and self._process_exists(int(pid)):
                continue
            connection.execute("DELETE FROM leases WHERE owner=?", (owner,))
            connection.execute("DELETE FROM claims WHERE owner=?", (owner,))
            connection.execute("DELETE FROM demands WHERE owner=?", (owner,))
            connection.execute("DELETE FROM owners WHERE owner=?", (owner,))

    def heartbeat(self) -> None:
        self._ensure_open()
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE owners SET last_seen=? WHERE owner=?",
                (now, self.owner_id),
            )
            connection.commit()

    def reserve(self, leases: Mapping[str, int]) -> LeaseDecision:
        self._ensure_open()
        requested = self._validated_leases(leases)
        if not requested:
            return LeaseDecision(
                True, "empty", 0, self.outstanding_bytes(), self.available_bytes()
            )
        started = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            decision = self._reserve_once(requested)
            waited = time.monotonic() - started
            retryable = decision.reason in {
                "global-capacity-exhausted", "fair-share-yield"
            }
            remaining = self.reserve_wait_seconds - waited
            if decision.admit or not retryable or remaining <= 0:
                return replace(
                    decision,
                    retry_count=attempts - 1,
                    waited_seconds=waited,
                )
            time.sleep(min(0.025, remaining))

    def _reserve_once(self, requested: Mapping[str, int]) -> LeaseDecision:
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_dead_owners(connection, now)
            connection.execute(
                "UPDATE owners SET last_seen=? WHERE owner=?", (now, self.owner_id)
            )
            acquired = []
            required = 0
            for key, size in requested.items():
                row = connection.execute(
                    "SELECT owner, bytes FROM leases WHERE lease_key=?", (key,)
                ).fetchone()
                if row is not None:
                    if row[0] == self.owner_id and int(row[1]) == size:
                        continue
                    connection.rollback()
                    raise RuntimeError(f"global broker lease key conflict: {key}")
                claim = connection.execute(
                    "SELECT owner, bytes FROM claims WHERE claim_key=?", (key,)
                ).fetchone()
                if claim is not None:
                    connection.rollback()
                    raise RuntimeError(f"global broker lease key conflict: {key}")
                acquired.append((key, size))
                required += size
            fair = self._fair_state(connection, now)
            used = fair["used"]
            tenant_usage = fair["usage"].get(self.tenant_id, 0)
            target = fair["targets"].get(self.tenant_id)
            starved = tuple(
                tenant for tenant in fair["starved"]
                if tenant != self.tenant_id
            )
            if used + required > self.capacity_bytes:
                decision = LeaseDecision(
                    False,
                    "global-capacity-exhausted",
                    required,
                    used,
                    max(0, self.capacity_bytes - used),
                    tenant_id=self.tenant_id,
                    tenant_usage_bytes=tenant_usage,
                    fair_target_bytes=target,
                    starved_tenants=starved,
                )
                self._record_demand(connection, required, now)
                connection.commit()
                return decision
            fair_yield = (
                self.fair_mode == "enforce"
                and required > 0
                and target is not None
                and tenant_usage + required > target
                and bool(starved)
            )
            if fair_yield:
                decision = LeaseDecision(
                    False,
                    "fair-share-yield",
                    required,
                    used,
                    self.capacity_bytes - used,
                    tenant_id=self.tenant_id,
                    tenant_usage_bytes=tenant_usage,
                    fair_target_bytes=target,
                    starved_tenants=starved,
                )
                self._record_demand(connection, required, now)
                connection.commit()
                return decision
            connection.executemany(
                "INSERT INTO leases(owner, lease_key, bytes) VALUES(?, ?, ?)",
                ((self.owner_id, key, size) for key, size in acquired),
            )
            connection.execute("DELETE FROM demands WHERE owner=?", (self.owner_id,))
            connection.commit()
        outstanding = used + required
        reason = "admitted" if acquired else "already-held"
        if (
            self.fair_mode == "observe"
            and required > 0
            and target is not None
            and tenant_usage + required > target
            and starved
        ):
            reason = "fair-share-observed"
        return LeaseDecision(
            True,
            reason,
            required,
            outstanding,
            self.capacity_bytes - outstanding,
            tuple(key for key, _ in acquired),
            tenant_id=self.tenant_id,
            tenant_usage_bytes=tenant_usage + required,
            fair_target_bytes=target,
            starved_tenants=starved,
        )

    def claim(
        self, key: str, bytes: int, *, kind: str = "prefill"
    ) -> LeaseDecision:
        """Reserve non-mapped capacity that still shares the broker budget."""

        self._ensure_open()
        normalized = str(key)
        if not normalized:
            raise ValueError("broker claim key must be nonempty")
        claim_kind = str(kind)
        if not claim_kind or len(claim_kind) > 64:
            raise ValueError("broker claim kind must be a short nonempty string")
        if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes < 1:
            raise ValueError("broker claim bytes must be a positive integer")

        started = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            decision = self._claim_once(normalized, bytes, claim_kind)
            waited = time.monotonic() - started
            retryable = decision.reason in {
                "global-capacity-exhausted", "fair-share-yield"
            }
            remaining = self.reserve_wait_seconds - waited
            if decision.admit or not retryable or remaining <= 0:
                return replace(
                    decision,
                    retry_count=attempts - 1,
                    waited_seconds=waited,
                )
            time.sleep(min(0.025, remaining))

    def _claim_once(
        self, key: str, size: int, kind: str
    ) -> LeaseDecision:
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_dead_owners(connection, now)
            connection.execute(
                "UPDATE owners SET last_seen=? WHERE owner=?", (now, self.owner_id)
            )
            lease = connection.execute(
                "SELECT owner, bytes FROM leases WHERE lease_key=?", (key,)
            ).fetchone()
            claim = connection.execute(
                "SELECT owner, bytes, claim_kind FROM claims WHERE claim_key=?", (key,)
            ).fetchone()
            if lease is not None:
                connection.rollback()
                raise RuntimeError(f"global broker claim key conflict: {key}")
            if claim is not None:
                if claim[0] != self.owner_id or int(claim[1]) != size or claim[2] != kind:
                    connection.rollback()
                    raise RuntimeError(f"global broker claim key conflict: {key}")
                fair = self._fair_state(connection, now)
                used = int(fair["used"])
                tenant_usage = fair["usage"].get(self.tenant_id, 0)
                connection.commit()
                return LeaseDecision(
                    True, "already-held", 0, used,
                    self.capacity_bytes - used, tenant_id=self.tenant_id,
                    tenant_usage_bytes=tenant_usage,
                    fair_target_bytes=fair["targets"].get(self.tenant_id),
                    starved_tenants=tuple(
                        tenant for tenant in fair["starved"]
                        if tenant != self.tenant_id
                    ),
                )

            fair = self._fair_state(connection, now)
            used = fair["used"]
            tenant_usage = fair["usage"].get(self.tenant_id, 0)
            target = fair["targets"].get(self.tenant_id)
            starved = tuple(
                tenant for tenant in fair["starved"] if tenant != self.tenant_id
            )
            if used + size > self.capacity_bytes:
                decision = LeaseDecision(
                    False, "global-capacity-exhausted", size, used,
                    max(0, self.capacity_bytes - used), tenant_id=self.tenant_id,
                    tenant_usage_bytes=tenant_usage, fair_target_bytes=target,
                    starved_tenants=starved,
                )
                self._record_demand(connection, size, now)
                connection.commit()
                return decision
            fair_yield = (
                self.fair_mode == "enforce"
                and target is not None
                and tenant_usage + size > target
                and bool(starved)
            )
            if fair_yield:
                decision = LeaseDecision(
                    False, "fair-share-yield", size, used,
                    self.capacity_bytes - used, tenant_id=self.tenant_id,
                    tenant_usage_bytes=tenant_usage, fair_target_bytes=target,
                    starved_tenants=starved,
                )
                self._record_demand(connection, size, now)
                connection.commit()
                return decision
            connection.execute(
                "INSERT INTO claims(owner, claim_key, claim_kind, bytes) "
                "VALUES(?, ?, ?, ?)",
                (self.owner_id, key, kind, size),
            )
            connection.execute("DELETE FROM demands WHERE owner=?", (self.owner_id,))
            connection.commit()

        outstanding = used + size
        reason = "admitted"
        if (
            self.fair_mode == "observe"
            and target is not None
            and tenant_usage + size > target
            and starved
        ):
            reason = "fair-share-observed"
        return LeaseDecision(
            True, reason, size, outstanding, self.capacity_bytes - outstanding,
            (key,), tenant_id=self.tenant_id,
            tenant_usage_bytes=tenant_usage + size, fair_target_bytes=target,
            starved_tenants=starved,
        )

    def _record_demand(
        self, connection: sqlite3.Connection, requested: int, now: float
    ) -> None:
        if self.tenant_id is None or requested <= 0:
            return
        connection.execute(
            "INSERT INTO demands(owner, tenant_id, requested_bytes, expires_at) "
            "VALUES(?, ?, ?, ?) "
            "ON CONFLICT(owner) DO UPDATE SET tenant_id=excluded.tenant_id, "
            "requested_bytes=excluded.requested_bytes, expires_at=excluded.expires_at",
            (
                self.owner_id,
                self.tenant_id,
                requested,
                now + self.demand_ttl_seconds,
            ),
        )

    def _fair_state(
        self,
        connection: sqlite3.Connection,
        now: float,
        *,
        prune_expired: bool = True,
    ) -> dict[str, object]:
        fair_mode = connection.execute(
            "SELECT value FROM metadata WHERE key='fair_mode'"
        ).fetchone()
        if fair_mode is None or fair_mode[0] != self.fair_mode:
            raise RuntimeError("global broker fair mode changed while owner is active")
        if prune_expired:
            connection.execute("DELETE FROM demands WHERE expires_at <= ?", (now,))
        used = int(
            connection.execute(
                "SELECT COALESCE((SELECT SUM(bytes) FROM leases), 0) + "
                "COALESCE((SELECT SUM(bytes) FROM claims), 0)"
            ).fetchone()[0]
        )
        usage = {
            str(tenant): int(bytes_used)
            for tenant, bytes_used in connection.execute(
                "SELECT tenant_id, SUM(bytes_used) FROM ("
                "SELECT owners.tenant_id AS tenant_id, leases.bytes AS bytes_used "
                "FROM leases JOIN owners ON owners.owner=leases.owner "
                "UNION ALL "
                "SELECT owners.tenant_id AS tenant_id, claims.bytes AS bytes_used "
                "FROM claims JOIN owners ON owners.owner=claims.owner) "
                "WHERE tenant_id IS NOT NULL GROUP BY tenant_id"
            ).fetchall()
        }
        demand_query = "SELECT tenant_id, SUM(requested_bytes) FROM demands"
        demand_parameters: tuple[float, ...] = ()
        if not prune_expired:
            demand_query += " WHERE expires_at > ?"
            demand_parameters = (now,)
        demand_query += " GROUP BY tenant_id"
        demands = {
            str(tenant): int(bytes_requested)
            for tenant, bytes_requested in connection.execute(
                demand_query, demand_parameters
            ).fetchall()
        }
        active = set(usage) | set(demands)
        if self.tenant_id is not None:
            active.add(self.tenant_id)
        weights = {
            str(tenant): float(weight)
            for tenant, weight in connection.execute(
                "SELECT tenant_id, weight FROM tenants"
            ).fetchall()
            if tenant in active
        }
        missing = active - set(weights)
        if missing:
            raise RuntimeError(
                "active fair-share tenants lack weight metadata: "
                + ", ".join(sorted(missing))
            )
        targets = self._weighted_targets(weights)
        starved = tuple(
            tenant
            for tenant in sorted(demands)
            if targets.get(tenant, 0) > usage.get(tenant, 0)
        )
        return {
            "used": used,
            "usage": usage,
            "demands": demands,
            "targets": targets,
            "starved": starved,
        }

    def _weighted_targets(self, weights: Mapping[str, float]) -> dict[str, int]:
        if not weights:
            return {}
        total_weight = math.fsum(weights.values())
        tenants = sorted(weights)
        targets = {
            tenant: int(
                math.floor(self.capacity_bytes * weights[tenant] / total_weight)
            )
            for tenant in tenants
        }
        remainder = self.capacity_bytes - sum(targets.values())
        for tenant in tenants[:remainder]:
            targets[tenant] += 1
        for tenant in tenants:
            if targets[tenant] == 0 and self.capacity_bytes >= len(tenants):
                donor = max(tenants, key=lambda item: targets[item])
                if targets[donor] > 1:
                    targets[donor] -= 1
                    targets[tenant] = 1
        return targets

    def release(self, keys) -> int:
        self._ensure_open()
        normalized = tuple(sorted({str(key) for key in keys}))
        if not normalized:
            return 0
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            released = 0
            for key in normalized:
                row = connection.execute(
                    "SELECT bytes FROM leases WHERE owner=? AND lease_key=?",
                    (self.owner_id, key),
                ).fetchone()
                if row is None:
                    continue
                released += int(row[0])
                connection.execute(
                    "DELETE FROM leases WHERE owner=? AND lease_key=?",
                    (self.owner_id, key),
                )
            connection.execute(
                "UPDATE owners SET last_seen=? WHERE owner=?",
                (time.time(), self.owner_id),
            )
            connection.commit()
        return released

    def release_claim(self, key: str) -> int:
        self._ensure_open()
        normalized = str(key)
        if not normalized:
            return 0
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT bytes FROM claims WHERE owner=? AND claim_key=?",
                (self.owner_id, normalized),
            ).fetchone()
            released = 0 if row is None else int(row[0])
            if row is not None:
                connection.execute(
                    "DELETE FROM claims WHERE owner=? AND claim_key=?",
                    (self.owner_id, normalized),
                )
            connection.execute(
                "UPDATE owners SET last_seen=? WHERE owner=?",
                (time.time(), self.owner_id),
            )
            connection.commit()
        return released

    def release_all(self) -> int:
        self._ensure_open()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            released = int(
                connection.execute(
                    "SELECT COALESCE((SELECT SUM(bytes) FROM leases WHERE owner=?), 0) "
                    "+ COALESCE((SELECT SUM(bytes) FROM claims WHERE owner=?), 0)",
                    (self.owner_id, self.owner_id),
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM leases WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM claims WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM demands WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM owners WHERE owner=?", (self.owner_id,))
            connection.commit()
        return released

    def outstanding_bytes(self) -> int:
        self._ensure_open()
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE((SELECT SUM(bytes) FROM leases), 0) + "
                    "COALESCE((SELECT SUM(bytes) FROM claims), 0)"
                ).fetchone()[0]
            )

    def available_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.outstanding_bytes())

    def snapshot(self) -> dict[str, object]:
        now = time.time()
        with self._connection() as connection:
            # Reservation prunes expired demands under its write transaction.
            # Snapshot only ignores them so frequent telemetry reads never take
            # a SQLite write lock just to observe a consistent WAL state.
            connection.execute("BEGIN")
            fair = self._fair_state(connection, now, prune_expired=False)
            lease_bytes = int(
                connection.execute(
                    "SELECT COALESCE(SUM(bytes), 0) FROM leases"
                ).fetchone()[0]
            )
            claim_bytes = int(
                connection.execute(
                    "SELECT COALESCE(SUM(bytes), 0) FROM claims"
                ).fetchone()[0]
            )
            claims = {
                str(key): {"kind": str(kind), "bytes": int(size)}
                for key, kind, size in connection.execute(
                    "SELECT claim_key, claim_kind, bytes FROM claims ORDER BY claim_key"
                )
            }
            connection.commit()
        outstanding = int(fair["used"])
        usage = fair["usage"]
        targets = fair["targets"]
        starved = fair["starved"]
        tenant_usage = usage.get(self.tenant_id, 0)
        tenant_target = targets.get(self.tenant_id)
        other_starved = tuple(
            tenant for tenant in starved if tenant != self.tenant_id
        )
        return {
            "capacity_bytes": self.capacity_bytes,
            "outstanding_bytes": outstanding,
            "lease_bytes": lease_bytes,
            "claim_bytes": claim_bytes,
            "claims": claims,
            "available_bytes": self.capacity_bytes - outstanding,
            "fair_mode": self.fair_mode,
            "tenant_id": self.tenant_id,
            "tenant_usage_bytes": tenant_usage,
            "fair_target_bytes": tenant_target,
            "tenant_deficit_bytes": (
                None if tenant_target is None else max(0, tenant_target - tenant_usage)
            ),
            "starved_tenants": other_starved,
            "fair_share_would_yield": bool(
                tenant_target is not None
                and tenant_usage >= tenant_target
                and other_starved
            ),
            "fairness_pressure": bool(
                self.fair_mode == "enforce"
                and tenant_target is not None
                and tenant_usage >= tenant_target
                and other_starved
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM leases WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM claims WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM demands WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM owners WHERE owner=?", (self.owner_id,))
            connection.commit()
        self._closed = True
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.timeout_seconds))
        self._close_all_connections()

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.heartbeat()
            except (OSError, sqlite3.Error):
                pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("global admission broker is closed")

    @staticmethod
    def _validated_leases(leases: Mapping[str, int]) -> dict[str, int]:
        result = {}
        for key, size in leases.items():
            normalized = str(key)
            if not normalized:
                raise ValueError("broker lease key must be nonempty")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("broker lease bytes must be positive integers")
            result[normalized] = size
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
