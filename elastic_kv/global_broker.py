"""Cross-process leases for aggregate CUDA VMM physical backing.

The broker coordinates callers on one host and one device.  It is deliberately
limited to chunks that a ``Backing`` is about to map; model weights, allocator
caches and activations remain outside its capacity.
"""

from __future__ import annotations

import ctypes
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LeaseDecision:
    admit: bool
    reason: str
    required_bytes: int
    outstanding_bytes: int
    available_bytes: int
    acquired_keys: tuple[str, ...] = ()


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


class GlobalAdmissionBroker:
    """SQLite-backed, first-come aggregate VMM leases.

    Every caller has a unique owner token.  A stale owner is reclaimed only
    after its recorded PID is demonstrably gone, or when the machine boot ID
    changed.  A live but hung process therefore keeps its safety reservation;
    the broker can leak capacity, but it never invalidates backing that may
    still be mapped by that process.
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
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = capacity
        self.owner_id = owner_id or f"{os.getpid()}-{uuid.uuid4().hex}"
        self.stale_after_seconds = float(stale_after_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._process_exists = process_exists
        self._boot_id = boot_id if boot_id is not None else _boot_id()
        self.resource_id = None if resource_id is None else str(resource_id)
        self._closed = False
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._initialize()
        if heartbeat_seconds is not None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(heartbeat_seconds,),
                name=f"ekv-broker-{self.owner_id[-8:]}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=self.timeout_seconds, isolation_level=None
        )
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

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
                "last_seen REAL NOT NULL)"
            )
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
            connection.execute(
                "INSERT INTO owners(owner, pid, boot_id, last_seen) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(owner) DO UPDATE SET last_seen=excluded.last_seen",
                (self.owner_id, os.getpid(), self._boot_id, now),
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
                acquired.append((key, size))
                required += size
            used = int(
                connection.execute("SELECT COALESCE(SUM(bytes), 0) FROM leases").fetchone()[0]
            )
            if used + required > self.capacity_bytes:
                connection.rollback()
                return LeaseDecision(
                    False,
                    "global-capacity-exhausted",
                    required,
                    used,
                    max(0, self.capacity_bytes - used),
                )
            connection.executemany(
                "INSERT INTO leases(owner, lease_key, bytes) VALUES(?, ?, ?)",
                ((self.owner_id, key, size) for key, size in acquired),
            )
            connection.commit()
        outstanding = used + required
        return LeaseDecision(
            True,
            "admitted" if acquired else "already-held",
            required,
            outstanding,
            self.capacity_bytes - outstanding,
            tuple(key for key, _ in acquired),
        )

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

    def release_all(self) -> int:
        self._ensure_open()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            released = int(
                connection.execute(
                    "SELECT COALESCE(SUM(bytes), 0) FROM leases WHERE owner=?",
                    (self.owner_id,),
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM leases WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM owners WHERE owner=?", (self.owner_id,))
            connection.commit()
        return released

    def outstanding_bytes(self) -> int:
        self._ensure_open()
        with self._connection() as connection:
            return int(
                connection.execute("SELECT COALESCE(SUM(bytes), 0) FROM leases").fetchone()[0]
            )

    def available_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.outstanding_bytes())

    def snapshot(self) -> dict[str, int]:
        outstanding = self.outstanding_bytes()
        return {
            "capacity_bytes": self.capacity_bytes,
            "outstanding_bytes": outstanding,
            "available_bytes": self.capacity_bytes - outstanding,
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM leases WHERE owner=?", (self.owner_id,))
            connection.execute("DELETE FROM owners WHERE owner=?", (self.owner_id,))
            connection.commit()
        self._closed = True
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.timeout_seconds))

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
