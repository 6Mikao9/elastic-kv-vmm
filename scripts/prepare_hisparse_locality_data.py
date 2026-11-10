"""Download pinned sources and build a private HiSparse workload JSONL."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


SOURCES = {
    "chatbot": (
        "allenai/WildChat-1M", "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        "data/train-00000-of-00014.parquet", "ODC-BY-1.0", 350,
    ),
    "agent": (
        "nebius/SWE-agent-trajectories", "68195a1450865274106246d0d0296a1d6807b88e",
        "data/train-00000-of-00012.parquet", "CC-BY-4.0", 150,
    ),
    "long_document": (
        "zai-org/LongBench-v2", "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "data.json", "Apache-2.0", 500,
    ),
}
CHUNK_BYTES = 1024 * 1024
# LFS OIDs returned by Hugging Face's tree API at the revisions above.
SOURCE_SHA256 = {
    "chatbot": "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c",
    "agent": "5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13",
    "long_document": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
}


def write_private(path, write):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n",
    ) as output:
        temporary = Path(output.name)
        try:
            write(output)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def source_info(category):
    repo, revision, filename, license_name, limit_mib = SOURCES[category]
    return {
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "license": license_name,
        "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
        "limit_bytes": limit_mib * CHUNK_BYTES,
    }


def source_path(directory, category):
    return directory / (category + "-" + Path(SOURCES[category][2]).name)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path, info, expected_sha256):
    request = Request(info["url"], headers={"User-Agent": "hisparse-locality-study/1.0"})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urlopen(request, timeout=90) as response:
                advertised = response.headers.get("Content-Length")
                limit = info["limit_bytes"]
                if advertised is not None and int(advertised) > limit:
                    raise ValueError(f"Source exceeds download limit: {info['url']}")
                total = 0
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError(f"Source exceeds download limit: {info['url']}")
                    output.write(chunk)
                    digest.update(chunk)
                if not total or (advertised is not None and total != int(advertised)):
                    raise ValueError(f"Incomplete download: {info['url']}")
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(f"Source does not match the pinned LFS SHA-256: {info['url']}")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_sources(directory, *, download):
    manifest_path = directory / "sources.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    manifest = {}
    for category in SOURCES:
        info = source_info(category)
        path = source_path(directory, category)
        record = previous.get(category)
        if download and (record is None or not path.exists()):
            print(f"Downloading {category} from {info['repo']}@{info['revision']}", file=sys.stderr)
            fetch(path, info, SOURCE_SHA256[category])
        if not path.is_file():
            raise ValueError(f"Missing {path}. Run the download command first.")
        if path.stat().st_size > info["limit_bytes"]:
            raise ValueError(f"Cached source exceeds size limit: {path}")
        info.update(bytes=path.stat().st_size, sha256=checksum(path))
        if info["sha256"] != SOURCE_SHA256[category]:
            raise ValueError(f"Cached source does not match the pinned LFS SHA-256: {path}")
        if record is None and not download:
            raise ValueError(f"Missing provenance for {path}. Run the download command first.")
        if record is not None and info != record:
            raise ValueError(f"Cached source differs from its recorded provenance: {path}")
        manifest[category] = info
        if download:
            combined = {**previous, **manifest}
            write_private(
                manifest_path, lambda handle: json.dump(combined, handle, indent=2),
            )
    return manifest


def parquet_rows(path, columns):
    try:
        from pyarrow.parquet import ParquetFile
    except ImportError as error:
        raise ValueError("Install pyarrow to read the two Parquet sources.") from error
    parquet = ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        yield from batch.to_pylist()


def message(role, content):
    return {"role": role, "content": content}


def in_range(chars, args):
    return args.min_chars <= chars <= args.max_chars


def chat_candidates(path, args):
    columns = ["conversation", "conversation_hash", "language", "model", "toxic"]
    for row_index, row in enumerate(parquet_rows(path, columns)):
        if row["toxic"] or (args.chat_language and row["language"] != args.chat_language):
            continue
        history = []
        checkpoints = []
        chars = 0
        for position, turn in enumerate(row["conversation"]):
            content = turn["content"] or ""
            role = turn["role"]
            if role not in {"user", "assistant"}:
                break
            if (role == "assistant" and content and history and history[-1]["role"] == "user"
                    and in_range(chars, args) and len(content) >= args.chat_min_reply_chars):
                checkpoints.append({
                    "checkpoint": position,
                    "source_turn_id": turn["turn_identifier"],
                    "messages": list(history),
                    "reference_completion": content,
                    "prompt_chars": chars,
                })
            history.append(message(role, content))
            chars += len(content)
            if chars > args.max_chars:
                break
        if len(checkpoints) >= 2:
            yield f"row-{row_index}", {
                "conversation_hash": row["conversation_hash"],
                "model": row["model"],
                "language": row["language"],
            }, [checkpoints[0], checkpoints[-1]]


def agent_candidates(path, args):
    columns = ["instance_id", "model_name", "trajectory", "target"]
    for row_index, row in enumerate(parquet_rows(path, columns)):
        trajectory = row["trajectory"]
        if isinstance(trajectory, str):
            trajectory = json.loads(trajectory)
        history = []
        checkpoints = []
        chars = 0
        for position, turn in enumerate(trajectory):
            role = turn["role"]
            if role == "ai":
                content = turn["text"] or ""
                if history and history[-1]["role"] == "user" and in_range(chars, args) and content:
                    checkpoints.append({
                        "checkpoint": position,
                        "messages": list(history),
                        "reference_completion": content,
                        "prompt_chars": chars,
                    })
                history.append(message("assistant", content))
            elif role == "system":
                content = turn["system_prompt"] or turn["text"] or ""
                history.append(message("system", content))
            elif role == "user":
                content = turn["text"] or ""
                history.append(message("user", content))
            else:
                break
            chars += len(content)
            if chars > args.max_chars:
                break
        if len(checkpoints) >= 3:
            yield f"row-{row_index}", {
                "instance_id": row["instance_id"],
                "model_name": row["model_name"],
                "resolved": row["target"],
            }, [checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]


def document_candidates(path, args):
    try:
        import ijson
    except ImportError as error:
        raise ValueError("Install ijson to stream the LongBench-v2 JSON array.") from error
    with path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            domain = row["domain"].lower()
            if "document" not in domain:
                continue
            choices = "\n".join(f"{letter}. {row[f'choice_{letter}']}" for letter in "ABCD")
            content = (
                f"Read the following material and answer the question. Explain the evidence "
                f"supporting your answer, then state the letter of the best choice.\n\n"
                f"{row['context']}\n\nQuestion: {row['question']}\n\n{choices}"
            )
            if in_range(len(content), args):
                yield row["_id"], {
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                }, [{"checkpoint": 0, "messages": [message("user", content)],
                      "prompt_chars": len(content)}]


def pick_origins(candidates, args, category):
    # Prefer distinct source groups, with a deterministic fallback if a shard
    # has fewer groups than the requested number of origins.
    best, by_group = [], {}
    eligible_count = 0
    group_key = {"chatbot": "conversation_hash", "agent": "instance_id",
                 "long_document": "sub_domain"}[category]
    for origin_id, metadata, checkpoints in candidates:
        eligible_count += 1
        key = f"{args.seed}/{category}/{origin_id}".encode()
        entry = (hashlib.sha256(key).digest(), origin_id, metadata, checkpoints)
        best.append(entry)
        best.sort(key=lambda candidate: candidate[0])
        del best[args.origins:]
        group = metadata[group_key]
        if group in by_group and entry[0] >= by_group[group][0]:
            continue
        if group not in by_group and len(by_group) == args.origins:
            last = max(by_group, key=lambda key: by_group[key][0])
            if entry[0] >= by_group[last][0]:
                continue
            del by_group[last]
        by_group[group] = entry
    selected = sorted(by_group.values(), key=lambda entry: entry[0])
    selected_ids = {entry[1] for entry in selected}
    selected.extend(entry for entry in best if entry[1] not in selected_ids)
    if len(selected) < args.origins:
        raise ValueError(
            f"{category}: found {eligible_count} eligible origins, need {args.origins}. "
            "Check the character range, chatbot language and reply-length filters.",
        )
    print(f"{category}: selected {args.origins} of {eligible_count} eligible origins",
          file=sys.stderr)
    return selected[:args.origins]


def build(directory, args, manifest):
    candidates = {"chatbot": chat_candidates, "agent": agent_candidates,
                  "long_document": document_candidates}
    modes = {"chatbot": "conversation_replay", "agent": "trajectory_replay",
             "long_document": "reasoned_response_diagnostic"}
    output = []
    for category, load in candidates.items():
        source = {key: manifest[category][key] for key in ("repo", "revision", "filename")}
        for _, origin, metadata, checkpoints in pick_origins(
            load(source_path(directory, category), args), args, category,
        ):
            origin_id = f"{category}/{origin}"
            for checkpoint in checkpoints:
                output.append({
                    "request_id": f"{origin_id}/{checkpoint['checkpoint']}",
                    "origin_id": origin_id,
                    "workload": category,
                    "mode": modes[category],
                    "source": {**source, "origin": origin, **metadata},
                    "max_new_tokens": 512,
                    **checkpoint,
                })
    destination = directory / "workloads.jsonl"

    def write(handle):
        for request in output:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    write_private(destination, write)
    print(f"Wrote {len(output)} requests to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "build", "all"])
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1]
                        / ".container-work" / "locality-study" / "datasets")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--origins", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=12000)
    parser.add_argument("--max-chars", type=int, default=54000)
    parser.add_argument("--chat-min-reply-chars", type=int, default=0,
                        help="optional filter for a longer recorded chatbot reply cohort")
    parser.add_argument("--chat-language")
    args = parser.parse_args()
    if not (0 < args.origins <= 100 and 0 < args.min_chars <= args.max_chars
            and args.chat_min_reply_chars >= 0):
        parser.error("Invalid origin count or character bounds.")
    try:
        args.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        manifest = prepare_sources(args.directory, download=args.command != "build")
        if args.command != "download":
            build(args.directory, args, manifest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
