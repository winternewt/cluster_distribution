"""Convert simdata/v2 CSVs to Parquet + validate 1-to-1 coverage.

Usage:
    python tools/simdata_to_parquet.py [--validate] [--out-dir OUTDIR]

Produces one .parquet per CSV in OUTDIR (default: simdata/v2_parquet/).
Schema preserved exactly: S_prime float64, N_prime int64, iteration int64
(including placeholder rows where S_prime=-1 and N_prime=-1).
Compression: zstd level 9.

Validation: round-trips each file and asserts byte-identical DataFrame.

TODO: once upstream scripts are updated to read parquet, retire the CSV
      counterparts. All read paths should prefer simdata/v2_parquet/ and
      fall back to simdata/v2/ only for legacy compatibility.
      Mark each caller with: # TODO(parquet): switch to parquet reader
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SIM_DIR = Path("simdata/v2")
DEFAULT_OUT = Path("simdata/v2_parquet")

# Exact schema matching the CSVs
SCHEMA = pa.schema([
    pa.field("S_prime", pa.float64()),
    pa.field("N_prime", pa.int64()),
    pa.field("iteration", pa.int64()),
])


def csv_to_parquet(csv_path: Path, out_path: Path) -> dict:
    """Convert one CSV to parquet. Returns stats dict."""
    df = pd.read_csv(csv_path, dtype={"S_prime": "float64", "N_prime": "int64", "iteration": "int64"})
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd", compression_level=9)
    return {
        "eps": csv_path.stem.replace("simulation_data_N10000_radius100_eps", ""),
        "rows": len(df),
        "real_rows": int((df.S_prime != -1).sum()),
        "csv_bytes": csv_path.stat().st_size,
        "parquet_bytes": out_path.stat().st_size,
    }


def validate(csv_path: Path, parquet_path: Path) -> bool:
    """Assert round-trip equality. Returns True or raises AssertionError."""
    df_csv = pd.read_csv(csv_path, dtype={"S_prime": "float64", "N_prime": "int64", "iteration": "int64"})
    df_pq = pd.read_parquet(parquet_path)
    df_pq = df_pq[df_csv.columns]  # ensure column order
    pd.testing.assert_frame_equal(df_csv.reset_index(drop=True),
                                  df_pq.reset_index(drop=True),
                                  check_exact=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="round-trip validate each file")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default=None, help="single eps value to process (e.g. 1.20)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(SIM_DIR.glob("simulation_data_N10000_radius100_eps*.csv"))
    if args.only:
        csvs = [c for c in csvs if f"eps{args.only}" in c.stem]
    if not csvs:
        print("No CSV files found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)

    total_csv = total_pq = 0
    errors = []
    for i, csv_path in enumerate(csvs, 1):
        eps = csv_path.stem.replace("simulation_data_N10000_radius100_eps", "")
        pq_path = out_dir / f"eps_{eps}.parquet"
        try:
            stats = csv_to_parquet(csv_path, pq_path)
            total_csv += stats["csv_bytes"]
            total_pq += stats["parquet_bytes"]
            ratio = stats["csv_bytes"] / stats["parquet_bytes"]
            if args.validate:
                validate(csv_path, pq_path)
                ok = " [validated]"
            else:
                ok = ""
            print(f"[{i:3d}/{len(csvs)}] eps={eps:5s}  "
                  f"rows={stats['rows']:>8,}  real={stats['real_rows']:>7,}  "
                  f"csv={stats['csv_bytes']//1024:>6}K  pq={stats['parquet_bytes']//1024:>5}K  "
                  f"ratio={ratio:.1f}x{ok}")
        except Exception as e:
            print(f"ERROR eps={eps}: {e}", file=sys.stderr)
            errors.append(eps)

    print(f"\nTotal CSV: {total_csv/1e9:.2f} GB  ->  Parquet: {total_pq/1e9:.2f} GB  "
          f"({total_csv/total_pq:.1f}x compression)")
    if errors:
        print(f"FAILED: {errors}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
