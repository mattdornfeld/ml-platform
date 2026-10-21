# PyTorch Lightning MNIST tutorial

Create the shared uv-managed virtual environment and install the project:

```bash
make configure
```

Run the complete test suite:

```bash
make test
```

The reusable `MNISTDataModule` and `MNISTClassifier` live in
`src/ml_platform/sdk/mnist.py`. The `ml-platform-train` command remains
available after `workon ml-platform`.

## Yahoo Finance hourly OHLCV data

The `ml-platform-download-ohlcv` command writes a daily-partitioned Apache
Iceberg table backed by Parquet. Configure `dataset_type`, `symbols`, and the
`time_range.start_date` / `time_range.end_date` bounds in [configs/datasets/sp500.yaml](configs/datasets/sp500.yaml).
The `sp500` variant in [configs/variants/sp500.yaml](configs/variants/sp500.yaml)
uses Hydra's `defaults` composition to place that configuration under
`datasets`. The checked-in dates request Yahoo Finance's maximum current
hourly history: 2024-10-20 through the exclusive end date 2026-10-20. The
default `sp500` result is materialized under `data/market_data/hourly_ohlcv`.
The data files include a `market_session` column with `pre_market`,
`regular`, or `after_hours`; extended-hours bars are always requested.

After `workon ml-platform`, run the configured download with:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@21 ml-platform-download-ohlcv --variant sp500
```

The table uses Iceberg's `days(timestamp)` transform, UTC session time,
Zstandard compression, and a `symbol` plus `timestamp` merge key. Yahoo Finance
restricts intraday history. Spark 4.1 currently requires Java 17 through 24.

## Long/short labels

The `ml-platform-extract-long-short-labels` command reads the hourly OHLCV
Iceberg table with Spark and writes daily-partitioned labels to
`data/labels/<name>_v<version>`. The composed
variant supplies [configs/labels/hourly_triple_barrier.yaml](configs/labels/hourly_triple_barrier.yaml).
For each event, the writer calculates trailing hourly-return volatility, then
uses future `High` and `Low` bars to find which is hit first: a profit-taking
barrier (`1`), stop-loss barrier (`-1`), or vertical time barrier (`0`). A bar
that touches both price barriers is recorded as `ambiguous` with label `0`.
Rows without a complete vertical horizon are omitted.

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@21 ml-platform-extract-long-short-labels --variant sp500
```

## Market features

`ml-platform-create-feature` applies a configured PySpark transformation
to the Iceberg market-data table and writes a versioned feature table under
`data/features/<name>_v<version>`, with Iceberg time partitions below its
`data` directory. The supplied configurations are `log_return`,
`trailing_volatility_1d`, `trailing_volatility_7d`, and `relative_volume`.
Trailing features use preceding bars only. Each YAML file in `configs/features` maps a versioned `FeatureConfig`
to one transform module under `src/ml_platform/data_sdk/features` and records its output dtype,
description, parameters, datasets, and free-form metadata. Each dataset
declares its dataset ID and number of preceding physical partitions; the current
partition is always included. The generic SDK harness loads the declared
dependency windows, then injects them into the transform constructor; it does not
contain feature-specific logic. The application
configuration in `configs/feature_materialization/default.yaml` controls
`max_concurrent_spark_jobs`; every YAML file in `configs/features` runs as a
separate Spark job. The harness records deterministic content hashes for every
dataset partition used by each feature-output partition. A changed or removed
parent marks only its dependent output partitions stale; those partitions are
deleted, recomputed, and given a replacement dependency manifest.

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@21 ml-platform-create-feature --config default
```

## Parquet explorer

Open an IPython session with every partitioned Parquet file below a directory
loaded into the pandas `frame` and `data` variables:

```bash
ml-platform-parquet-explorer data/market_data/hourly_ohlcv/data
```

## Kubernetes materialization environment

The development deployment runs the control plane, PostgreSQL, and MinIO in the
`ml-platform` namespace. Iceberg's Hadoop catalog writes its warehouse to
the `warehouse` bucket in MinIO; it does not write dataset partitions to the
API pod filesystem. The API creates `SparkApplication` custom resources with
the Kubernetes Python client and waits for the Spark Operator to report a
terminal state.

Run `scripts/bootstrap-minikube.sh` from the repository root to create the
Minikube cluster, install the Spark Operator, build `ml-platform:latest`
in Minikube's image store, and apply the versioned manifests. Access the API
locally with `kubectl -n ml-platform port-forward service/ml-control-plane 8000:8000`.

The control-plane client is available as the `ml_platform.cli` module:

```bash
python -m ml_platform.cli --help
```

The MinIO credentials in `deploy/kubernetes/minio.yaml` are development-only.
Replace that Secret through the deployment environment before using a shared
cluster.
