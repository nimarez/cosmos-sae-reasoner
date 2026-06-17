from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import write_jsonl
from ..corpus import (
    compact_sidecar_metadata,
    enrich_records,
    fetch_json_sidecars_from_hf_tar,
    merge_sidecar,
    parse_pax_headers,
    parse_tar_header,
    record_assigned,
    sidecar_member_for_media_member,
    sidecar_s3_key,
    source_shard_and_member,
    tar_data_span,
    upload_robotsim_sidecar,
)
from ..storage import parse_s3_uri, read_text_uri, s3_client

DEFAULT_REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair an older RobotSim manifest by adding paired JSON sidecar captions/metadata."
    )
    parser.add_argument("--manifest", required=True, help="Local path or S3 URI for an existing RobotSim manifest JSONL.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-s3-uri", default=None, help="Optional S3 URI for the repaired manifest.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--caption-field", default="caption")
    parser.add_argument("--metadata-mode", choices=["none", "compact", "full"], default="compact")
    parser.add_argument("--sidecar-s3-uri", default=None, help="Optional S3 prefix for uploading full RobotSim JSON sidecars.")
    parser.add_argument("--missing", choices=["keep", "drop", "error"], default="keep")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    records = list(iter_manifest_records(args.manifest))
    repaired, stats = enrich_records(
        records,
        repo_id=args.repo_id,
        caption_field=args.caption_field,
        metadata_mode=args.metadata_mode,
        sidecar_s3_uri=args.sidecar_s3_uri,
        missing=args.missing,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
    )
    write_jsonl(args.output, repaired)
    manifest_uri = upload_file_to_s3(args.output, args.manifest_s3_uri) if args.manifest_s3_uri else None
    print(
        json.dumps(
            {
                **stats,
                "manifest": args.manifest,
                "manifest_uri": manifest_uri,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def iter_manifest_records(uri: str) -> Iterable[dict[str, Any]]:
    for line_no, line in enumerate(read_text_uri(uri).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{uri}:{line_no}: expected object record")
        yield record


def validate_worker_partition(worker_index: int, num_workers: int) -> None:
    from ..corpus import validate_worker_partition as validate

    validate(worker_index=worker_index, num_workers=num_workers)


def upload_file_to_s3(path: Path, uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    s3_client().upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/jsonl; charset=utf-8"},
    )
    return f"s3://{bucket}/{key}"


__all__ = [
    "compact_sidecar_metadata",
    "enrich_records",
    "fetch_json_sidecars_from_hf_tar",
    "merge_sidecar",
    "parse_pax_headers",
    "parse_tar_header",
    "record_assigned",
    "sidecar_member_for_media_member",
    "sidecar_s3_key",
    "source_shard_and_member",
    "tar_data_span",
    "upload_robotsim_sidecar",
]


if __name__ == "__main__":
    raise SystemExit(main())
