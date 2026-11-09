import pickle
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from adaptive_sparse import kv_hints_compat as compat
from adaptive_sparse import sglang_future_hook as hook

ENVELOPE = {
    "protocol_version": "1",
    "message_id": "msg-1",
    "actions": [
        {
            "action_id": "future-1",
            "action_type": "elastic.future",
            "action_version": "1",
            "payload": {"deadline_after_ms": 500},
        }
    ],
}


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@dataclass
class _BaseGenerateInput:
    text: object = None
    rid: object = None
    kv_hints: object = None
    batch_size: int = 0
    is_single: bool = False
    parallel_sample_num: int = 1

    def normalize_batch_and_arguments(self):
        if isinstance(self.text, str):
            self.is_single = True
            self.batch_size = 1
            self.parallel_sample_num = 1
            self._normalize_single_inputs()
        else:
            self.is_single = False
            self.batch_size = len(self.text)
            self._normalize_batch_inputs(self.batch_size)

    def _normalize_single_inputs(self):
        if self.rid is None:
            self.rid = "rid"

    def _normalize_batch_inputs(self, num):
        if self.rid is None:
            self.rid = [f"rid-{i}" for i in range(self.batch_size)]

    def __getitem__(self, index):
        return _BaseGenerateInput(
            text=self.text[index],
            rid=self.rid[index],
            batch_size=1,
            is_single=True,
        )


class KvHintsTransportTest(unittest.TestCase):
    def test_envelope_is_immutable_and_pickle_safe(self):
        envelope = compat.decode_kv_hints_envelope(ENVELOPE)
        self.assertIs(compat.decode_kv_hints_envelope(envelope), envelope)
        self.assertEqual(pickle.loads(pickle.dumps(envelope)), envelope)
        with self.assertRaises(Exception):  # noqa: B017 - frozen dataclass contract
            envelope.message_id = "changed"
        self.assertEqual(compat._encode_kv_hints(envelope), ENVELOPE)

    def test_v4_cohort_metadata_round_trips_to_scheduler_hint(self):
        payload = {
            "protocol_version": "1",
            "message_id": "cohort-message",
            "actions": [
                {
                    "action_id": "cohort-member",
                    "action_type": "elastic.future",
                    "action_version": "4",
                    "payload": {
                        "cohort_id": "batch-0",
                        "cohort_size": 2,
                    },
                }
            ],
        }

        envelope = compat.decode_kv_hints_envelope(payload)
        hint = hook._extract_hint(SimpleNamespace(kv_hints=envelope))

        self.assertEqual(hint.cohort_id, "batch-0")
        self.assertEqual(hint.cohort_size, 2)
        self.assertEqual(compat._encode_kv_hints(envelope), payload)

    def test_rejects_malformed_envelope_and_action(self):
        cases = (
            None,
            {},
            {"protocol_version": "2", "message_id": "m", "actions": []},
            {"protocol_version": "1", "message_id": "", "actions": []},
            {"protocol_version": "1", "message_id": "m", "actions": None},
            {
                "protocol_version": "1",
                "message_id": "m",
                "actions": [{"action_id": "a"}],
            },
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                compat.decode_kv_hints_envelope(value)

    def test_batch_broadcast_and_per_item_hints(self):
        compat_input = compat._build_generate_input_class(_BaseGenerateInput)

        broadcast = compat_input(text=["a", "b"], kv_hints=ENVELOPE)
        broadcast.normalize_batch_and_arguments()
        self.assertEqual(
            [broadcast[i].kv_hints.message_id for i in range(2)],
            ["msg-1", "msg-1"],
        )

        second = dict(ENVELOPE)
        second["message_id"] = "msg-2"
        per_item = compat_input(text=["a", "b"], kv_hints=[ENVELOPE, second])
        per_item.normalize_batch_and_arguments()
        self.assertEqual(
            [per_item[i].kv_hints.message_id for i in range(2)],
            ["msg-1", "msg-2"],
        )

    def test_partial_batch_hint_is_rejected(self):
        compat_input = compat._build_generate_input_class(_BaseGenerateInput)
        request = compat_input(text=["a", "b"], kv_hints=[ENVELOPE])
        with self.assertRaisesRegex(ValueError, "equal the batch size"):
            request.normalize_batch_and_arguments()

    def test_parallel_sampling_expands_one_hint_per_generated_item(self):
        compat_input = compat._build_generate_input_class(_BaseGenerateInput)
        request = compat_input(text=["a", "b"], kv_hints=ENVELOPE)
        request.batch_size = 2
        request.parallel_sample_num = 2
        request._normalize_batch_inputs(4)
        self.assertEqual(len(request.kv_hints), 4)
        self.assertTrue(
            all(hint.message_id == "msg-1" for hint in request.kv_hints)
        )

    def test_empty_hint_objects_are_rejected_in_single_and_batch_forms(self):
        compat_input = compat._build_generate_input_class(_BaseGenerateInput)
        for text, hints in (("a", {}), (["a", "b"], [None, {}])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                compat_input(text=text, kv_hints=hints).normalize_batch_and_arguments()

    def test_hint_survives_fake_http_to_queue_path(self):
        class Req:
            def __init__(self, rid):
                self.rid = rid
                self.left_queue = False

        class TokenizerManager:
            def _create_tokenized_object(self, obj, *args, **kwargs):
                return SimpleNamespace(rid=obj.rid)

        class Scheduler:
            def handle_generate_request(self, recv_req):
                req = Req(recv_req.rid)
                self._add_request_to_queue(req)

            def _add_request_to_queue(self, req):
                req.left_queue = True
                self.queued = req

        io_struct = _module(
            "sglang.srt.managers.io_struct", GenerateReqInput=_BaseGenerateInput
        )
        compat.install_generate_input(io_struct)
        request = io_struct.GenerateReqInput(text="hello", kv_hints=ENVELOPE)
        request.normalize_batch_and_arguments()

        modules = {
            "sglang": _module("sglang"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.managers": _module("sglang.srt.managers"),
            "sglang.srt.managers.io_struct": io_struct,
            "sglang.srt.managers.tokenizer_manager": _module(
                "sglang.srt.managers.tokenizer_manager",
                TokenizerManager=TokenizerManager,
            ),
            "sglang.srt.managers.schedule_batch": _module(
                "sglang.srt.managers.schedule_batch", Req=Req
            ),
            "sglang.srt.managers.scheduler": _module(
                "sglang.srt.managers.scheduler", Scheduler=Scheduler
            ),
        }
        with patch.dict(sys.modules, modules):
            compat.install_tokenizer_bridge()
            compat.install_scheduler_bridge()
            tokenized = TokenizerManager()._create_tokenized_object(request)
            scheduler = Scheduler()
            scheduler.handle_generate_request(tokenized)

        self.assertEqual(scheduler.queued.kv_hints, request.kv_hints)
        self.assertTrue(scheduler.queued.left_queue)
        self.assertGreater(scheduler.queued._ekv_scheduler_recv_time, 0)


class RealSGLang052CompatibilityTest(unittest.TestCase):
    def test_installer_rejects_unknown_sglang_before_patching(self):
        modules = {"sglang": _module("sglang", __version__="dev-unknown")}
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(
            RuntimeError, "only valid for supported SGLang"
        ):
            compat.install()

    def test_real_request_dataclass_accepts_hints(self):
        try:
            import sglang
        except ImportError:
            self.skipTest("SGLang is not installed")
        if sglang.__version__ not in compat._SUPPORTED_SGLANG_VERSIONS:
            self.skipTest(
                f"unsupported real SGLang version: {sglang.__version__}"
            )

        request_class = compat.install()
        request = request_class(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 4},
            rid="real-compat",
            kv_hints=ENVELOPE,
        )
        request.normalize_batch_and_arguments()
        self.assertTrue(request.is_single)
        self.assertEqual(request.kv_hints.message_id, "msg-1")
        if sglang.__version__ == "0.5.17":
            from sglang.srt.managers import io_struct

            self.assertIn(
                "kv_hints", io_struct.TokenizedGenerateReqInput.__struct_fields__
            )


if __name__ == "__main__":
    unittest.main()
