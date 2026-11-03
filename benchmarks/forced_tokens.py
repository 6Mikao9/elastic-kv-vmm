"""Synthetic token replay for trusted, isolated SGLang benchmark endpoints."""

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class ForceTokens(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        if logits.ndim != 2 or custom_param_list is None or len(custom_param_list) != logits.shape[0]:
            raise ValueError("Expected one forced-token sequence per logit row")
        targets = []
        for params in custom_param_list:
            request = params["__req__"]
            ids = params["forced_ids"]
            step = len(request.output_ids)
            if not isinstance(ids, list) or len(ids) != request.sampling_params.max_new_tokens:
                raise ValueError("Forced sequence length must equal max_new_tokens")
            if not 0 <= step < len(ids):
                raise ValueError("Forced sequence exhausted")
            token = ids[step]
            if type(token) is not int or not 0 <= token < logits.shape[1]:
                raise ValueError("Forced token ID outside vocabulary")
            targets.append(token)
        indices = logits.new_tensor(targets, dtype=torch.int64).unsqueeze(1)
        return logits.new_full(logits.shape, float("-inf")).scatter_(1, indices, 0.0)


def forced_ids(input_tokens, output_tokens):
    """Return a deterministic synthetic sequence independent of model outputs."""
    if type(input_tokens) is not int or input_tokens <= 0:
        raise ValueError("Input token count must be positive")
    if type(output_tokens) is not int or not 1 <= output_tokens <= 128:
        raise ValueError("Synthetic replay supports one to 128 generated tokens")
    palette = (256, 367, 494, 583, 722, 742, 952, 967)
    return [palette[(input_tokens + i) % len(palette)] for i in range(output_tokens)]


def forced_payload(input_tokens, output_tokens):
    """Return JSON request fields to merge into an otherwise normal /generate request."""
    ids = forced_ids(input_tokens, output_tokens)
    return {"custom_logit_processor": ForceTokens.to_str(),
            "sampling_params": {"temperature": 0, "ignore_eos": True,
                                "max_new_tokens": len(ids), "custom_params": {"forced_ids": ids}}}


def verify_forced_output(record, expected):
    """Reject partial or divergent token records from controlled replay."""
    actual = record.get("output_ids")
    if (not record.get("output_ids_complete") or not isinstance(actual, list)
            or record.get("output_tokens") != len(expected)
            or any(type(token) is not int for token in actual) or actual != expected):
        raise ValueError("Generated token IDs differ from the complete forced sequence")
