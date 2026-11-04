"""Single-owner byte ledger for experiments, with no device allocator hooks."""


class BudgetLedger:
    """Call from one scheduling thread and use one key per physical allocation.

    External bytes exclude allocations charged here. Reservations describe peak
    incremental bytes, including temporary buffers and overlapping migration
    destinations. Releasing bytes requires confirmation from the allocator.
    """

    def __init__(self, capacity_bytes, safety_margin_bytes=0, external_bytes=0):
        self._validate_bytes(capacity_bytes)
        self._validate_bytes(safety_margin_bytes)
        if safety_margin_bytes > capacity_bytes:
            raise ValueError("Safety margin exceeds capacity")
        self._capacity = capacity_bytes
        self._margin = safety_margin_bytes
        self._allocations = {}
        self._pending = {}
        self.set_external_bytes(external_bytes)

    @staticmethod
    def _validate_bytes(size):
        if type(size) is not int or size < 0:
            raise ValueError("Byte count must be a nonnegative integer")

    def set_external_bytes(self, size):
        self._validate_bytes(size)
        self._external = size

    def _validate_allocation(self, key, category, size):
        self._validate_bytes(size)
        if key in self._allocations:
            raise ValueError("Allocation key already registered")
        if not isinstance(category, str) or not category:
            raise ValueError("Allocation category must be a nonempty string")

    def register(self, key, category, committed_bytes):
        """Record an existing allocation even when observed usage exceeds capacity."""
        self._validate_allocation(key, category, committed_bytes)
        self._allocations[key] = (category, committed_bytes)

    def reserve(self, ticket, peak_increment_bytes):
        self._validate_bytes(peak_increment_bytes)
        if ticket in self._pending:
            raise ValueError("Ticket already reserved")
        if peak_increment_bytes > self._remaining():
            return False
        self._pending[ticket] = peak_increment_bytes
        return True

    def commit(self, ticket, key, category, actual_bytes):
        """Convert a ticket to a new allocation without taking a second charge.

        actual_bytes must cover every still-live byte charged to this ticket.
        Temporary buffers must be gone, or separately charged, before unused
        reserved peak capacity is returned by this call.
        """
        reserved = self._pending[ticket]
        self._validate_allocation(key, category, actual_bytes)
        if actual_bytes > reserved:
            raise ValueError("Actual bytes exceed reserved peak")
        self._allocations[key] = (category, actual_bytes)
        del self._pending[ticket]

    def cancel(self, ticket):
        """Cancel a reservation after any attempted allocation is rolled back."""
        return self._pending.pop(ticket)

    def release(self, key, *, quiescent):
        if quiescent is not True:
            raise ValueError("Release requires explicit confirmation of quiescence")
        return self._allocations.pop(key)[1]

    def _remaining(self):
        committed = sum(size for _, size in self._allocations.values())
        return self._capacity - self._margin - self._external - committed - sum(self._pending.values())

    def snapshot(self):
        by_category = {}
        for category, size in self._allocations.values():
            by_category[category] = by_category.get(category, 0) + size
        remaining = self._remaining()
        return {
            "capacity_bytes": self._capacity,
            "safety_margin_bytes": self._margin,
            "external_bytes": self._external,
            "committed_bytes": sum(by_category.values()),
            "committed_by_category": by_category,
            "pending_bytes": sum(self._pending.values()),
            "pending_by_ticket": dict(self._pending),
            "available_bytes": max(0, remaining),
            "overcommitted_bytes": max(0, -remaining),
            "overcommitted": remaining < 0,
        }
