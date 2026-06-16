from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import write_jsonl
from ..storage import parse_s3_uri, read_text_uri, s3_client

DEFAULT_REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes"


def main() -> int:
    parser = argparse.ArgumentParser(description="Add RobotSim JSON sidecar captions/metadata to an existing manifest.")
    parser.add_argument("--manifest", required=True, help="Local path or S3 URI for an existing RobotSim manifest JSONL.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-s3-uri", default=None, help="Optional S3 URI for the enriched manifest.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--caption-field", default="caption")
    parser.add_argument("--metadata-mode", choices=["none", "compact", "full"], default="compact")
    parser.add_argument("--sidecar-s3-uri", default=None, help="Optional S3 prefix for uploading full RobotSim JSON sidecars.")
    parser.add_argument("--missing", choices=["keep", "drop", "error"], default="keep")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    records = list(iter_manifest_records(args.manifest))
    enriched, stats = enrich_records(
        records,
        repo_id=args.repo_id,
        caption_field=args.caption_field,
        metadata_mode=args.metadata_mode,
        sidecar_s3_uri=args.sidecar_s3_uri,
        missing=args.missing,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
    )
    write_jsonl(args.output, enriched)
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


def enrich_records(
    records: list[dict[str, Any]],
    *,
    repo_id: str,
    caption_field: str,
    metadata_mode: str,
    missing: str,
    sidecar_s3_uri: str | None = None,
    worker_index: int = 0,
    num_workers: int = 1,
    sidecar_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    sidecar_uploader: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_worker_partition(worker_index, num_workers)
    selected_records = [
        record for idx, record in enumerate(records) if record_assigned(idx, worker_index=worker_index, num_workers=num_workers)
    ]
    by_shard: dict[str, set[str]] = defaultdict(set)
    record_sidecars: list[tuple[dict[str, Any], str | None, str | None]] = []
    for record in selected_records:
        shard, member = source_shard_and_member(record)
        sidecar_member = sidecar_member_for_media_member(member) if member else None
        if shard and sidecar_member:
            by_shard[shard].add(sidecar_member)
        record_sidecars.append((record, shard, sidecar_member))

    fetched: dict[tuple[str, str], dict[str, Any]] = dict(sidecar_lookup or {})
    if sidecar_lookup is None:
        for shard, sidecar_members in sorted(by_shard.items()):
            for member, sidecar in fetch_json_sidecars_from_hf_tar(repo_id, shard, sidecar_members).items():
                fetched[(shard, member)] = sidecar

    enriched: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "selected_records": len(selected_records),
        "output_records": 0,
        "enriched_records": 0,
        "missing_sidecars": 0,
        "missing_captions": 0,
        "uploaded_sidecars": 0,
    }
    for record, shard, sidecar_member in record_sidecars:
        sidecar = fetched.get((shard, sidecar_member)) if shard and sidecar_member else None
        if sidecar is None:
            stats["missing_sidecars"] += 1
            if missing == "drop":
                continue
            if missing == "error":
                raise ValueError(f"missing RobotSim sidecar for record {record.get('id')!r}")
            enriched.append(record)
            continue

        sidecar_path = None
        if sidecar_s3_uri and shard and sidecar_member:
            sidecar_path = upload_robotsim_sidecar(
                sidecar_s3_uri,
                repo_id=repo_id,
                shard=shard,
                sidecar_member=sidecar_member,
                sidecar=sidecar,
                uploader=sidecar_uploader,
            )
            stats["uploaded_sidecars"] += 1

        caption = sidecar.get(caption_field)
        if not isinstance(caption, str) or not caption.strip():
            stats["missing_captions"] += 1
            if missing == "drop":
                continue
            if missing == "error":
                raise ValueError(f"missing {caption_field!r} in RobotSim sidecar for record {record.get('id')!r}")
            enriched.append(
                merge_sidecar(
                    record,
                    sidecar,
                    sidecar_member=sidecar_member,
                    sidecar_path=sidecar_path,
                    metadata_mode=metadata_mode,
                )
            )
            continue

        next_record = merge_sidecar(
            record,
            sidecar,
            sidecar_member=sidecar_member,
            sidecar_path=sidecar_path,
            metadata_mode=metadata_mode,
        )
        next_record["prompt"] = caption.strip()
        enriched.append(next_record)
        stats["enriched_records"] += 1

    stats["output_records"] = len(enriched)
    return enriched, stats


def source_shard_and_member(record: dict[str, Any]) -> tuple[str | None, str | None]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    shard = metadata.get("shard")
    member = metadata.get("member")
    if isinstance(shard, str) and isinstance(member, str):
        return shard, member
    source_uri = metadata.get("source_uri")
    if isinstance(source_uri, str) and source_uri.startswith("hf://dataset/") and "#" in source_uri:
        before, member = source_uri.split("#", 1)
        parts = before.removeprefix("hf://dataset/").split("/", 2)
        if len(parts) == 3:
            return parts[2], member
    return None, None


def sidecar_member_for_media_member(member: str) -> str | None:
    lower = member.lower()
    if lower.endswith(".mp4"):
        return member[:-4] + ".json"
    return None


def merge_sidecar(
    record: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    sidecar_member: str | None,
    sidecar_path: str | None = None,
    metadata_mode: str,
) -> dict[str, Any]:
    next_record = dict(record)
    metadata = dict(next_record.get("metadata") or {})
    metadata["prompt_source"] = "robotsim_sidecar"
    if sidecar_member:
        metadata["sidecar_member"] = sidecar_member
    if sidecar_path:
        metadata["sidecar_path"] = sidecar_path
    if metadata_mode == "compact":
        compact = compact_sidecar_metadata(sidecar)
        if compact:
            metadata["robotsim_sidecar"] = compact
    elif metadata_mode == "full":
        metadata["robotsim_sidecar"] = sidecar
    next_record["metadata"] = metadata
    return next_record


def compact_sidecar_metadata(sidecar: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in sidecar.items():
        if key == "caption":
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            compact[key] = value
    return compact


def upload_robotsim_sidecar(
    sidecar_s3_uri: str,
    *,
    repo_id: str,
    shard: str,
    sidecar_member: str,
    sidecar: dict[str, Any],
    uploader: Any | None = None,
) -> str:
    bucket, prefix = parse_s3_uri(sidecar_s3_uri)
    key = sidecar_s3_key(prefix, repo_id=repo_id, shard=shard, sidecar_member=sidecar_member)
    body = json.dumps(sidecar, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    if uploader is None:
        s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )
    else:
        uploader(bucket, key, body)
    return f"s3://{bucket}/{key}"


def sidecar_s3_key(prefix: str, *, repo_id: str, shard: str, sidecar_member: str) -> str:
    repo_slug = repo_id.replace("/", "--")
    shard_stem = Path(shard).stem
    parts = [
        prefix.rstrip("/"),
        repo_slug,
        safe_s3_path_part(shard_stem),
        *safe_s3_member_parts(sidecar_member),
    ]
    return "/".join(part for part in parts if part)


def safe_s3_member_parts(path: str) -> list[str]:
    return [safe_s3_path_part(part) for part in Path(path).parts if part not in {"", "."}]


def safe_s3_path_part(part: str) -> str:
    return part.replace("%", "%25").replace("/", "%2F").replace(":", "%3A")


def fetch_json_sidecars_from_hf_tar(repo_id: str, shard: str, sidecar_members: Iterable[str]) -> dict[str, dict[str, Any]]:
    try:
        import requests
        from huggingface_hub import hf_hub_url
    except Exception as exc:  # pragma: no cover - optional dependencies
        raise RuntimeError("RobotSim sidecar enrichment requires requests and huggingface_hub.") from exc

    needed = set(sidecar_members)
    if not needed:
        return {}

    session = requests.Session()
    url = hf_hub_url(repo_id, shard, repo_type="dataset")
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    pax_attrs: dict[str, str] = {}
    gnu_long_name: str | None = None
    while needed:
        header = http_range(session, url, offset, offset + 511)
        parsed = parse_tar_header(header)
        if parsed is None:
            break
        raw_name, size, typeflag = parsed
        data_offset = offset + 512
        member_name = pax_attrs.pop("path", None) or gnu_long_name or raw_name
        gnu_long_name = None

        if typeflag == "x":
            text = http_range(session, url, data_offset, data_offset + size - 1).decode("utf-8", "replace") if size else ""
            pax_attrs.update(parse_pax_headers(text))
        elif typeflag == "L":
            text = http_range(session, url, data_offset, data_offset + size - 1).decode("utf-8", "replace") if size else ""
            gnu_long_name = text.split("\0", 1)[0]
        elif member_name in needed:
            data = http_range(session, url, data_offset, data_offset + size - 1) if size else b"{}"
            found[member_name] = json.loads(data.decode("utf-8"))
            needed.remove(member_name)

        offset = data_offset + tar_data_span(size)
    return found


def http_range(session: Any, url: str, start: int, end: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    response.raise_for_status()
    return response.content


def parse_tar_header(block: bytes) -> tuple[str, int, str] | None:
    if len(block) < 512 or all(byte == 0 for byte in block):
        return None
    name = block[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix:
        name = f"{prefix}/{name}"
    size_raw = block[124:136].split(b"\0", 1)[0].strip() or b"0"
    try:
        size = int(size_raw, 8)
    except ValueError:
        size = 0
    typeflag = block[156:157].decode("ascii", "ignore") or "0"
    return name, size, typeflag


def parse_pax_headers(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    index = 0
    while index < len(text):
        space = text.find(" ", index)
        if space < 0:
            break
        try:
            length = int(text[index:space])
        except ValueError:
            break
        record = text[space + 1 : index + length]
        if "=" in record:
            key, value = record.rstrip("\n").split("=", 1)
            attrs[key] = value
        index += length
    return attrs


def tar_data_span(size: int) -> int:
    return ((size + 511) // 512) * 512


def record_assigned(index: int, *, worker_index: int, num_workers: int) -> bool:
    return index % num_workers == worker_index


def validate_worker_partition(worker_index: int, num_workers: int) -> None:
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError("--worker-index must be in [0, num_workers)")


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
