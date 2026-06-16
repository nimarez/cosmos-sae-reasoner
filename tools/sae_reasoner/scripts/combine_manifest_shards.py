from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import write_jsonl
from ..storage import iter_s3_keys, parse_s3_uri, read_text_uri, s3_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine per-worker manifest JSONL shards.")
    parser.add_argument("--input", action="append", default=[], help="Local path or S3 URI to a worker manifest. Can be repeated.")
    parser.add_argument("--input-prefix", action="append", default=[], help="Local directory or S3 prefix containing *.jsonl worker manifests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-s3-uri", default=None, help="Optional S3 URI for the combined manifest.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate record ids instead of dropping later duplicates.")
    args = parser.parse_args()

    inputs = list(expand_inputs(args.input, args.input_prefix))
    if not inputs:
        raise ValueError("provide at least one --input or --input-prefix")

    records = combine_manifests(inputs, dedupe=not args.no_dedupe)
    write_jsonl(args.output, records)
    manifest_uri = upload_file_to_s3(args.output, args.manifest_s3_uri) if args.manifest_s3_uri else None
    print(
        json.dumps(
            {
                "inputs": inputs,
                "manifest_uri": manifest_uri,
                "num_inputs": len(inputs),
                "num_records": len(records),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def expand_inputs(inputs: list[str], prefixes: list[str]) -> Iterable[str]:
    yield from inputs
    for prefix in prefixes:
        raw = str(prefix)
        if raw.startswith("s3://"):
            bucket, key_prefix = parse_s3_uri(raw)
            for key in iter_s3_keys(bucket, key_prefix, suffix=".jsonl"):
                yield f"s3://{bucket}/{key}"
            continue
        path = Path(raw)
        for child in sorted(path.glob("*.jsonl")):
            yield str(child)


def combine_manifests(inputs: list[str], *, dedupe: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for uri in sorted(inputs):
        for line_no, line in enumerate(read_text_uri(uri).splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{uri}:{line_no}: expected object record")
            record_id = str(record.get("id") or "")
            if dedupe and record_id:
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
            records.append(record)
    records.sort(key=lambda item: str(item.get("id") or ""))
    return records


def upload_file_to_s3(path: Path, uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    s3_client().upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/jsonl; charset=utf-8"},
    )
    return f"s3://{bucket}/{key}"


if __name__ == "__main__":
    raise SystemExit(main())
