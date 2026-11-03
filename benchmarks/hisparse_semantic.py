"""Four short semantic checks against a local MiniMax chat endpoint."""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


CASES = (
    ("arithmetic", "What is 7 times 8? Reply with only the number.", r"56"),
    ("capital", "What is the capital of France? Reply with only the city name.", r"Paris"),
    ("translation", 'Translate “hello” to simplified Chinese. Reply with only the translation.',
     r"你好|您好"),
    ("sort", "Sort 9, 2, 5 from smallest to largest. Reply with only the three numbers.",
     r"2[\s,，]+5[\s,，]+9"),
)


def check_answer(content, pattern):
    if not isinstance(content, str):
        return False
    answer = content.strip()
    # Some chat endpoints expose a thinking block in content, rather than
    # returning it as a separate reasoning field.
    for opening, closing in (("<think>", "</think>"), ("<mm:think>", "</mm:think>")):
        if answer.startswith(opening) and closing in answer:
            answer = answer.split(closing, 1)[1].strip()
        elif answer.startswith(closing):
            # MiniMax's chat prompt may already contain the opening tag.
            answer = answer[len(closing):].strip()
    answer = answer.rstrip(".。").strip()
    return re.fullmatch(pattern, answer, re.IGNORECASE) is not None


def smoke(base_url, model, timeout):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https")
            or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use a loopback service URL with no credentials or query")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = endpoint.removesuffix("/v1") + "/v1/chat/completions"
    opener = build_opener(ProxyHandler({}))
    checks = []
    for name, prompt, pattern in CASES:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 256, "stream": False}
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with opener.open(request, timeout=timeout) as response:
            choice = json.load(response)["choices"][0]
        message = choice["message"]
        finish = choice["finish_reason"]
        content = message.get("content")
        checks.append({"case": name, "prompt": prompt, "content": content,
                       "reasoning_content": message.get("reasoning_content"),
                       "finish_reason": finish,
                       "passed": finish == "stop" and check_answer(content, pattern)})
    return {"model": model, "status": "complete", "passed": all(c["passed"] for c in checks),
            "checks": checks, "note": "Short semantic smoke test. It does not measure model quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31080")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = smoke(args.url, args.model, args.timeout)
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            print(rendered, end="")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        parser.exit(2, f"Semantic smoke request failed ({type(exc).__name__})\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
