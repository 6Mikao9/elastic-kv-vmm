"""Opt-in ``kv_hints`` transport compatibility for SGLang 0.5.2/0.5.17.

Neither supported release exposes the upstream ``kv_hints`` transport used by
the future-aware scheduler. This module backports only that request transport.
It does not enable any storage or attention feature.

Installation is deliberately process local and idempotent.  The launcher runs
it before ``http_server`` imports ``GenerateReqInput`` and every spawned
worker repeats it through the existing elastic import hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import time
from types import ModuleType
from typing import Any, Mapping, Optional


PROTOCOL_VERSION = "1"
_SUPPORTED_SGLANG_VERSIONS = ("0.5.2", "0.5.17")
_MARKER = "_ekv_kv_hints_compat"


@dataclass(frozen=True, slots=True)
class KvHintAction:
    action_id: str
    action_type: str
    action_version: str
    payload: Optional[dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class KvHintsEnvelope:
    protocol_version: str
    message_id: str
    actions: tuple[KvHintAction, ...]


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"kv_hints {name} must be a non-empty string")
    return value


def decode_kv_hints_envelope(value: Any) -> KvHintsEnvelope:
    """Validate a wire envelope and return an immutable representation."""
    if isinstance(value, KvHintsEnvelope):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("kv_hints must be an object")

    version = value.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported kv_hints protocol_version {version!r}")
    message_id = _required_string(value.get("message_id"), "message_id")
    raw_actions = value.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("kv_hints actions must be a list")

    actions: list[KvHintAction] = []
    for action in raw_actions:
        if not isinstance(action, Mapping):
            raise ValueError("each kv_hints action must be an object")
        payload = action.get("payload")
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("kv_hints action payload must be an object or null")
        actions.append(
            KvHintAction(
                action_id=_required_string(action.get("action_id"), "action_id"),
                action_type=_required_string(
                    action.get("action_type"), "action_type"
                ),
                action_version=_required_string(
                    action.get("action_version"), "action_version"
                ),
                payload=None if payload is None else dict(payload),
            )
        )
    return KvHintsEnvelope(PROTOCOL_VERSION, message_id, tuple(actions))


def _encode_kv_hints(value: Any) -> Optional[dict[str, Any]]:
    """Convert a validated envelope to an encoding-neutral IPC value."""
    if value is None:
        return None
    envelope = decode_kv_hints_envelope(value)
    return {
        "protocol_version": envelope.protocol_version,
        "message_id": envelope.message_id,
        "actions": [
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "action_version": action.action_version,
                "payload": action.payload,
            }
            for action in envelope.actions
        ],
    }


def _normalize_one_hint(value: Any) -> KvHintsEnvelope:
    return decode_kv_hints_envelope(value)


def _build_generate_input_class(base: type) -> type:
    """Create an appended-field dataclass without copying SGLang's large schema."""

    original_normalize_single = base._normalize_single_inputs
    original_normalize_batch = base._normalize_batch_inputs
    original_getitem = base.__getitem__

    @dataclass
    class GenerateReqInput(base):
        kv_hints: Optional[Any] = None

        def _normalize_single_inputs(self):
            original_normalize_single(self)
            if isinstance(self.kv_hints, list):
                raise ValueError("kv_hints should be one envelope for a single request")
            self.kv_hints = (
                _normalize_one_hint(self.kv_hints)
                if self.kv_hints is not None
                else None
            )

        def _normalize_batch_inputs(self, num):
            original_normalize_batch(self, num)
            hints = self.kv_hints
            if hints is None:
                return
            if isinstance(hints, (Mapping, KvHintsEnvelope)):
                normalized = _normalize_one_hint(hints)
                self.kv_hints = [normalized] * num
            elif isinstance(hints, list):
                if len(hints) != self.batch_size:
                    raise ValueError(
                        "the length of kv_hints should equal the batch size"
                    )
                normalized = [
                    _normalize_one_hint(value) if value is not None else None
                    for value in hints
                ]
                self.kv_hints = normalized * self.parallel_sample_num
            else:
                raise ValueError("kv_hints should be a list or an object")

        def __getitem__(self, index):
            result = original_getitem(self, index)
            result.kv_hints = (
                self.kv_hints[index] if self.kv_hints is not None else None
            )
            return result

    # Pickling resolves through io_struct, which installs this object under the
    # original public name.  Override only those two initialization arguments.
    GenerateReqInput.__module__ = base.__module__
    GenerateReqInput.__qualname__ = base.__qualname__
    GenerateReqInput.__name__ = base.__name__
    setattr(GenerateReqInput, _MARKER, True)
    return GenerateReqInput


def install_generate_input(io_struct: ModuleType | None = None) -> type:
    """Patch the imported HTTP dataclass before http_server resolves it."""
    if io_struct is None:
        import sglang.srt.managers.io_struct as io_struct
    base = io_struct.GenerateReqInput
    if getattr(base, _MARKER, False):
        return base
    compat = _build_generate_input_class(base)
    io_struct.GenerateReqInput = compat
    return compat


def install_tokenized_generate_input(io_struct: ModuleType | None = None) -> type | None:
    """Append a wire-safe field to the 0.5.17 msgspec IPC request struct."""
    if io_struct is None:
        import sglang.srt.managers.io_struct as io_struct
    base = getattr(io_struct, "TokenizedGenerateReqInput", None)
    if base is None or not hasattr(base, "__struct_fields__"):
        return None
    if getattr(base, _MARKER, False):
        return base

    import msgspec

    inherited = set(io_struct.BaseReq.__struct_fields__)
    fields = []
    for item in msgspec.structs.fields(base):
        if item.name in inherited:
            continue
        if item.default is msgspec.NODEFAULT:
            fields.append((item.name, item.type))
        else:
            fields.append((item.name, item.type, item.default))
    fields.append(("kv_hints", Optional[dict[str, Any]], None))
    compat = msgspec.defstruct(
        base.__name__,
        fields,
        bases=(io_struct.BaseReq,),
        tag=True,
        kw_only=True,
        array_like=True,
    )
    compat.__module__ = base.__module__
    compat.__qualname__ = base.__qualname__
    setattr(compat, _MARKER, True)
    io_struct.TokenizedGenerateReqInput = compat
    return compat

def install_tokenizer_bridge() -> None:
    """Carry the hint across the tokenizer-to-scheduler serialization hop."""
    import sglang.srt.managers.io_struct as io_struct
    from sglang.srt.managers import tokenizer_manager

    TokenizerManager = tokenizer_manager.TokenizerManager
    tokenized_class = install_tokenized_generate_input(io_struct)
    if tokenized_class is not None:
        # Avoid rebuilding SGLang's large request struct and keep every worker
        # on the same msgspec schema as the tokenizer process.
        tokenizer_manager.TokenizedGenerateReqInput = tokenized_class

    original_create = TokenizerManager._create_tokenized_object
    if getattr(original_create, "_ekv_kv_hints_bridge", False):
        return

    @wraps(original_create)
    def create_tokenized_object(self, obj, *args, **kwargs):
        tokenized = original_create(self, obj, *args, **kwargs)
        if hasattr(obj, "kv_hints"):
            tokenized.kv_hints = _encode_kv_hints(obj.kv_hints)
        return tokenized

    create_tokenized_object._ekv_kv_hints_bridge = True
    TokenizerManager._create_tokenized_object = create_tokenized_object


def install_scheduler_bridge() -> None:
    """Attach tokenizer metadata to ``Req`` before SchedulePolicy runs."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

    original_req_init = Req.__init__
    if not getattr(original_req_init, "_ekv_kv_hints_bridge", False):
        @wraps(original_req_init)
        def req_init(self, *args, **kwargs):
            original_req_init(self, *args, **kwargs)
            if not hasattr(self, "kv_hints"):
                self.kv_hints = None

        req_init._ekv_kv_hints_bridge = True
        Req.__init__ = req_init

    original_handle = Scheduler.handle_generate_request
    original_add = Scheduler._add_request_to_queue
    if not getattr(original_handle, "_ekv_kv_hints_bridge", False):
        @wraps(original_handle)
        def handle_generate_request(self, recv_req):
            pending = getattr(self, "_ekv_pending_kv_hints", None)
            if pending is None:
                pending = {}
                self._ekv_pending_kv_hints = pending
            raw_hint = getattr(recv_req, "kv_hints", None)
            hint = (
                None
                if raw_hint is None
                else decode_kv_hints_envelope(raw_hint)
            )
            pending[recv_req.rid] = (hint, time.perf_counter())
            return original_handle(self, recv_req)

        handle_generate_request._ekv_kv_hints_bridge = True
        Scheduler.handle_generate_request = handle_generate_request

    if not getattr(original_add, "_ekv_kv_hints_bridge", False):
        @wraps(original_add)
        def add_request_to_queue(self, req, *args, **kwargs):
            pending = getattr(self, "_ekv_pending_kv_hints", {})
            item = pending.pop(req.rid, None)
            if item is not None:
                req.kv_hints, req._ekv_scheduler_recv_time = item
            return original_add(self, req, *args, **kwargs)

        add_request_to_queue._ekv_kv_hints_bridge = True
        Scheduler._add_request_to_queue = add_request_to_queue


def install() -> type:
    """Install both halves of the compatibility layer in this process."""
    import sglang

    version = getattr(sglang, "__version__", None)
    if version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            "kv_hints compatibility is only valid for supported SGLang "
            f"versions {_SUPPORTED_SGLANG_VERSIONS}; got {version}"
        )
    compat = install_generate_input()
    install_tokenizer_bridge()
    install_scheduler_bridge()
    return compat
