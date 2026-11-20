#!/usr/bin/env python3
"""Create a physically separated pre-sealed model input for development experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed-start", default="2020-01-01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.output.exists() or args.manifest.exists()) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    cutoff = date.fromisoformat(args.sealed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(args.input)
    writer: pq.ParquetWriter | None = None
    rows = 0
    minimum: date | None = None
    maximum: date | None = None
    try:
        for batch in source.iter_batches(batch_size=100_000):
            mask = pc.less(batch.column("month"), pa.scalar(cutoff, type=pa.date32()))
            kept = pa.Table.from_batches([batch]).filter(mask)
            if kept.num_rows == 0:
                continue
            months = kept.column("month")
            batch_min = pc.min(months).as_py()
            batch_max = pc.max(months).as_py()
            minimum = batch_min if minimum is None else min(minimum, batch_min)
            maximum = batch_max if maximum is None else max(maximum, batch_max)
            if writer is None:
                writer = pq.ParquetWriter(args.output, kept.schema, compression="zstd")
            writer.write_table(kept, row_group_size=100_000)
            rows += kept.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or maximum is None or maximum >= cutoff:
        raise RuntimeError("Pre-sealed output failed its date boundary")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Physical exclusion of the sealed period before remote GPU development",
        "sealed_start": args.sealed_start,
        "source": {
            "path": str(args.input.resolve()),
            "size_bytes": args.input.stat().st_size,
            "sha256": sha256(args.input),
        },
        "output": {
            "path": str(args.output.resolve()),
            "size_bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "rows": rows,
            "minimum_month": minimum.isoformat() if minimum else None,
            "maximum_month": maximum.isoformat(),
        },
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
