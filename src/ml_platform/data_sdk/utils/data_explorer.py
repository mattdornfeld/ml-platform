"""Open a pandas-backed IPython session for local or MinIO Parquet data."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as fs
from IPython import embed


def _minio_filesystem() -> fs.S3FileSystem:
    """Build a PyArrow S3 filesystem for the configured MinIO endpoint."""
    endpoint = os.environ.get("S3_ENDPOINT", os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000"))
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return fs.S3FileSystem(
        access_key=os.environ.get("MINIO_ROOT_USER", "minio"),
        secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "miniosecret"),
        endpoint_override=parsed.netloc,
        scheme=parsed.scheme,
    )


def _parquet_source(directory: str | Path) -> tuple[str | list[str], fs.FileSystem | None]:
    """Resolve a local directory or an s3/s3a URI into a PyArrow dataset source."""
    value = str(directory)
    parsed = urlparse(value)
    if parsed.scheme in {"s3", "s3a"}:
        path = f"{parsed.netloc}{parsed.path}".strip("/")
        if not path:
            raise ValueError("An S3/MinIO URI must include a bucket and table path.")
        return path, _minio_filesystem()
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Parquet directory does not exist: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found below: {root}")
    return [str(file) for file in files], None


def load_partitioned_parquet(directory: str | Path) -> pd.DataFrame:
    """Load every local or MinIO Parquet file below ``directory`` into pandas."""
    source, filesystem = _parquet_source(directory)
    dataset = ds.dataset(source, filesystem=filesystem, format="parquet")
    return dataset.to_table().to_pandas()


def parse_args() -> argparse.Namespace:
    """Parse a local path or s3a:// MinIO table URI to explore."""
    parser = argparse.ArgumentParser(
        description="Load local or MinIO partitioned Parquet into pandas and open IPython."
    )
    parser.add_argument(
        "directory",
        help="Local Parquet directory or MinIO URI, e.g. s3a://warehouse/market_data/sp500-v3.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested table and open an IPython REPL with it in scope."""
    args = parse_args()
    frame = load_partitioned_parquet(args.directory)
    embed(
        header="Data explorer: `frame` and `data` contain the loaded pandas DataFrame.",
        user_ns={"data": frame, "directory": args.directory, "frame": frame, "pd": pd},
    )


if __name__ == "__main__":
    main()
