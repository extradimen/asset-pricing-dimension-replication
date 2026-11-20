#!/usr/bin/env python3
"""Build characteristic-month-indexed excess returns for 74 public pricing assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_teacher_pricing import read_factor_file, read_first_value_weighted_monthly


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolios-25", type=Path, required=True)
    parser.add_argument("--industries-49", type=Path, required=True)
    parser.add_argument("--ff5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names25, returns25 = read_first_value_weighted_monthly(args.portfolios_25, 25, "size_bm::")
    names49, returns49 = read_first_value_weighted_monthly(args.industries_49, 49, "industry::")
    ff5 = read_factor_file(args.ff5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    asset_names = names25 + names49
    characteristic_months = np.arange(np.datetime64("1963-07"), np.datetime64("2020-01"), dtype="datetime64[M]")
    matrix: list[np.ndarray] = []
    for characteristic_month in characteristic_months:
        realized = str(characteristic_month + 1)
        if realized not in returns25 or realized not in returns49 or realized not in ff5:
            raise ValueError(f"Missing official return month {realized}")
        gross = np.concatenate([returns25[realized], returns49[realized]])
        matrix.append(gross - ff5[realized][-1])
    values = np.vstack(matrix).astype(np.float32)
    missing_cells = int((~np.isfinite(values)).sum())
    if (np.isfinite(values).sum(axis=1) < 25).any():
        raise ValueError("Fewer than 25 public pricing assets are available in at least one month")
    arrays: dict[str, pa.Array] = {
        "month": pa.array(characteristic_months.astype("datetime64[D]")),
    }
    arrays.update({f"asset_{index:03d}": pa.array(values[:, index]) for index in range(values.shape[1])})
    metadata = {
        b"asset_names": json.dumps(asset_names).encode(),
        b"timing": b"characteristic month t paired with official excess return in t+1",
    }
    table = pa.table(arrays).replace_schema_metadata(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    print(json.dumps({"rows": table.num_rows, "assets": values.shape[1], "missing_cells": missing_cells, "start": str(characteristic_months[0]), "end": str(characteristic_months[-1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
