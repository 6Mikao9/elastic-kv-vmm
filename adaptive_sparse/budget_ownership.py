"""Read-only attribution for the runtime's layered memory budget."""

from collections import Counter
from collections.abc import Mapping


SCHEMA_VERSION = 1


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value, name):
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _physical_broker(snapshot):
    if snapshot is None:
        return {
            "status": "disabled",
            "authority": "none",
        }
    if not isinstance(snapshot, Mapping):
        raise TypeError("physical broker snapshot must be an object")
    capacity = _nonnegative_int(snapshot.get("capacity_bytes"), "capacity_bytes")
    outstanding = _nonnegative_int(
        snapshot.get("outstanding_bytes"), "outstanding_bytes"
    )
    available = _nonnegative_int(
        snapshot.get("available_bytes"), "available_bytes"
    )
    if outstanding > capacity or available != capacity - outstanding:
        raise ValueError("inconsistent physical broker snapshot")
    lease_bytes = _optional_nonnegative_int(
        snapshot.get("lease_bytes"), "lease_bytes"
    )
    claim_bytes = _optional_nonnegative_int(
        snapshot.get("claim_bytes"), "claim_bytes"
    )
    if (lease_bytes is None) != (claim_bytes is None):
        raise ValueError("physical broker lease and claim bytes must be paired")
    if lease_bytes is not None and lease_bytes + claim_bytes != outstanding:
        raise ValueError("physical broker lease/claim partition is inconsistent")

    claims = snapshot.get("claims")
    claim_kinds = Counter()
    if claims is not None:
        if not isinstance(claims, Mapping):
            raise TypeError("physical broker claims must be an object")
        for key, claim in claims.items():
            if not isinstance(key, str) or not isinstance(claim, Mapping):
                raise TypeError("physical broker claim entry is malformed")
            size = _nonnegative_int(claim.get("bytes"), "claim bytes")
            kind = claim.get("kind")
            if not isinstance(kind, str) or not kind:
                raise TypeError("physical broker claim kind is malformed")
            claim_kinds[kind] += size
        if claim_bytes is not None and sum(claim_kinds.values()) != claim_bytes:
            raise ValueError("physical broker claim entries do not match claim bytes")

    result = {
        "status": "enabled",
        "authority": "physical-capacity",
        "capacity_bytes": capacity,
        "outstanding_bytes": outstanding,
        "available_bytes": available,
        "lease_bytes": lease_bytes,
        "claim_bytes": claim_bytes,
        "active_claim_kinds": dict(sorted(claim_kinds.items())),
    }
    for name in (
        "fair_mode",
        "tenant_id",
        "fairness_pressure",
        "fair_share_would_yield",
    ):
        if name in snapshot:
            result[name] = snapshot[name]
    return result


def _request_growth(snapshot, *, mode):
    if snapshot is None:
        return {
            "status": "disabled",
            "mode": mode,
            "authority": "none",
        }
    if not isinstance(snapshot, Mapping):
        raise TypeError("request growth snapshot must be an object")
    outstanding = _nonnegative_int(
        snapshot.get("outstanding_bytes"), "request growth outstanding_bytes"
    )
    active = _nonnegative_int(
        snapshot.get("active_reservations"), "active_reservations"
    )
    result = {
        "status": "enabled",
        "mode": mode,
        "authority": "predictive-advisory",
        "outstanding_bytes": outstanding,
        "active_reservations": active,
    }
    for name in (
        "assessments",
        "would_admit",
        "would_decline",
        "reservations",
        "consumed_bytes",
        "released_bytes",
    ):
        if name in snapshot:
            result[name] = _nonnegative_int(snapshot[name], name)
    declines = snapshot.get("decline_reasons")
    if declines is not None:
        if not isinstance(declines, Mapping):
            raise TypeError("request growth decline reasons must be an object")
        result["decline_reasons"] = {
            str(reason): _nonnegative_int(count, "decline count")
            for reason, count in sorted(declines.items())
        }
    return result


def _vmm_backing(snapshot):
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise TypeError("VMM backing snapshot must be an object")
    return {
        name: _nonnegative_int(snapshot[name], name)
        for name in (
            "backed_bytes",
            "live_bytes",
            "reclaimable_bytes",
            "mapped_chunks",
        )
        if name in snapshot
    }


def _activation(counters):
    if counters is None:
        return {
            "status": "disabled",
            "authority": "physical-temporary-claim",
        }
    if not isinstance(counters, Mapping):
        raise TypeError("activation counters must be an object")
    selected = {}
    for name, value in sorted(counters.items()):
        if not isinstance(name, str):
            continue
        if name.startswith("activation_global_prefill_") and isinstance(
            value, (bool, int, float, str, type(None))
        ):
            selected[name] = value
    return {
        "status": "enabled",
        "authority": "physical-temporary-claim",
        "counters": selected,
    }


def build_budget_ownership_snapshot(
    *,
    physical_broker_snapshot=None,
    request_growth_snapshot=None,
    request_growth_mode="off",
    vmm_backing_snapshot=None,
    activation_counters=None,
):
    """Describe the existing budget owners without creating another ledger.

    The physical broker is authoritative and already includes VMM leases plus
    temporary prefill claims. Request-growth reservations are advisory promises
    made against native free capacity and must not be added to broker usage.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "physical_broker": _physical_broker(physical_broker_snapshot),
        "request_growth": _request_growth(
            request_growth_snapshot, mode=str(request_growth_mode)
        ),
        "vmm_backing": _vmm_backing(vmm_backing_snapshot),
        "activation": _activation(activation_counters),
        "relationship": {
            "physical_broker_is_authoritative": (
                physical_broker_snapshot is not None
            ),
            "request_growth_is_additive_to_physical_usage": False,
            "prefill_claim_is_additive_to_physical_usage": True,
        },
    }
