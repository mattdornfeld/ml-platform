# Materialization control plane

The PostgreSQL control plane records the intent and outcome of materializing Iceberg-backed datasets and features. Spark owns table schemas and writes; PostgreSQL stores resource definitions, work state, completed partition receipts, and lineage.

The initial database schema is [src/ml_platform/api/sql/schema.sql](../src/ml_platform/api/sql/schema.sql).

## Resources

Datasets and features are distinct resources, each identified by an immutable `(name, version)` pair. A dataset has a typed `dataset_type` and an opaque `config JSONB` document. The initial supported type is `market_data`. Spark and Iceberg own the dataset's physical schema.

Feature registration is normalized rather than stored as an opaque configuration. A feature has `dtype`, `frequency`, ordered `feature_keys`, `transform`, `description`, `parameters`, and `validations` fields. Its `datasets` payload creates rows in `feature_dataset_dependencies`; each row identifies an existing `(dataset_name, dataset_version)` and its `num_lookback_partitions`. Foreign keys prevent registering a feature dependency on a dataset version that does not exist.

A new dataset or feature configuration version creates a new resource version rather than mutating an existing one.

## Jobs and partition update requests

A `jobs` row represents one materialization dispatch. Its `type` is either `dataset_materialization` or `feature_materialization` and its status reports aggregate job progress.

One job may have many partition update requests:

- `partition_update_requests` targets a dataset partition.
- `feature_partition_update_requests` targets a feature partition.

Each request has a UUID `partition_id`, the logical UTC `timestamp` to materialize, a `partition_status`, and the owning job and resource version. The request UUID identifies that particular attempt to produce a partition. Corrections or rematerializations create new requests rather than modifying a previous completed result.

The dataset and feature partition endpoints accept a `timestamps` array. Every value must be timezone-aware UTC and aligned to the start of an hour: minutes, seconds, and microseconds must be zero. The API rejects an invalid timestamp with a validation error; it never silently rounds a requested partition.

## HTTP errors and validation

Every current route documents each of its error status codes with complete JSON examples in OpenAPI.

| Status | Meaning | Response body |
| --- | --- | --- |
| `422` | A path parameter or request body failed validation. | A `detail` array containing the field location, message, error type, invalid input, and validation context. |
| `501` | The route is declared, but its database-backed service is not implemented. | `{"detail": "Database-backed materialization services have not been implemented."}` |

The OpenAPI `422` examples cover every declared constraint: resource names and versions, required and unknown request properties, feature dtype and frequency, feature dependencies and lookbacks, transform and description fields, UUID path parameters, content hashes, and every partition timestamp rule. When database-backed behavior adds `404` or `409` responses, those statuses must also have complete examples before the routes are exposed.

Partition request status follows this lifecycle:

- A newly persisted request is `pending`.
- After Spark commits the corresponding Iceberg partition and the API records its receipt, the request becomes `created`.
- If the job fails before the partition receipt is recorded, the request becomes `failed`.
- If the job is cancelled before the partition receipt is recorded, the request becomes `cancelled`.

`created`, `failed`, and `cancelled` are terminal states. The API must reject transitions out of them.

The API validates that dataset requests belong only to dataset jobs and feature requests belong only to feature jobs. This validation intentionally lives in the API, not in PostgreSQL triggers.

## Materialization lifecycle

The normal lifecycle is:

1. Create a versioned dataset or feature resource.
2. Create a job and its partition update requests.
3. Dispatch a Spark job for the requests.
4. Spark writes the requested partitions in an Iceberg transaction.
5. After a successful Iceberg commit, the worker sends one top-level JSON array to `POST /partitions/complete`. Each receipt carries its own request UUID, resource type, row count, full committed schema, content hash, snapshot ID, and callback token. Feature and label receipts also carry their pinned `dataset_partition_ids`.
6. In one PostgreSQL transaction, the API validates the entire batch, creates its partition results and lineage, marks requests `created`, and updates completed jobs.

A `dataset_partitions`, `feature_partitions`, or `label_partitions` row is a receipt for a successful Iceberg write. It is keyed by the request UUID and records the content hash, snapshot, row count, schema, and creation time. Its resource identity, timestamp, job, and lifecycle status come from the corresponding request. Existing historical receipts predate row-count collection; the migration preserves their unknown counts without inventing values, while requiring complete new receipts.

The API rejects duplicate partition IDs, invalid tokens, ineligible or deleted resources, mismatched pinned dependency plans, and inconsistent schemas within the batch or against completed receipts in the same Iceberg table. Any failure rolls back the entire batch. An identical retry succeeds without changing state; a conflicting receipt returns `409`. Locking the resource definition during validation serializes concurrent schema checks for that table.

## Kueue admission and partition eligibility

The API owns data-partition ordering and eligibility. Kueue owns Kubernetes resource admission after the API has released a job. Kueue does not make dataset or feature lineage decisions.

### Durable state

`jobs.status` has only four values:

- `submitted`: the nonterminal state, covering an API-held workload, a Kueue-pending workload, a Kueue-admitted workload, or a running Spark application;
- `succeeded`: every target update request has a successful completion receipt;
- `failed`: the Spark application reaches a terminal failure state before every target receipt is recorded; and
- `cancelled`: the API cancels the job before completion.

The API response augments this durable state by querying Kubernetes for the Kueue Workload and SparkApplication state. It does not copy Kueue admission or Spark execution states into PostgreSQL.

### Submission and hold

The API creates every SparkApplication with the Kueue LocalQueue label and this admission gate:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: materialization
  annotations:
    kueue.x-k8s.io/admission-gated-by: api.ml-control-plane.io/blocked
```

The gate is added at creation time, which Kueue requires. While present, Kueue does not begin quota reservation. The API creates the job and all of its partition update requests before creating the SparkApplication. Multiple submitted jobs may have update requests for the same logical resource partition.

### API-controlled FIFO release

The API evaluates submitted jobs in `(created_at, id)` order. For a candidate, it obtains short-lived PostgreSQL advisory transaction locks for all target resource-partition keys in sorted order. It then confirms that no older nonterminal job and no Kueue-admitted or running job targets any same partition.

If a conflict exists, the API leaves `api.ml-control-plane.io/blocked` in place. Otherwise, it removes that one admission gate from the SparkApplication. Kueue then applies its own `StrictFIFO` resource queue and quota policy before unsuspending the SparkApplication.

The advisory locks prevent concurrent API scheduler workers from releasing conflicting jobs at the same time. They are not durable ownership records. Durable ordering comes from the `jobs.created_at` ordering and the update-request rows. Unrelated jobs can be released concurrently.

### Completion and cancellation

Workers complete update requests only after an Iceberg commit. When every target callback is complete, the API marks the job `succeeded`. It marks a job `failed` when the SparkApplication reaches a terminal failure state before all callbacks finish. Cancellation keeps the gate in place when possible, deletes or suspends the SparkApplication, marks unfinished requests `cancelled`, and marks the job `cancelled`.

The local deployment installs Kueue's SparkApplication integration, which is currently an upstream alpha feature. `deploy/kubernetes/kueue-values.yaml` enables its feature gate and framework registration; `deploy/kubernetes/kueue.yaml` defines the `materialization` LocalQueue and its resource quota.

## Iceberg consistency

Every dataset, feature, and label table records the same resource definition in Iceberg table properties: `resource.type`, `resource.name`, `resource.version`, `resource.frequency`, `resource.description`, `resource.dependencies`, and `resource.validations`. Frequency is the table's logical partition cadence (`daily` or `hourly`); dependencies and validations are JSON strings. The shared materializer writes these properties when creating a table and backfills existing tables on their next materialization. Features and labels use canonical layouts inferred from their resource metadata: `timestamp`, ordered `feature_keys`, and respectively `feature_value` or `label_value`. Their frequency determines the Iceberg partition transform. Dataset transforms retain ownership of arbitrary schemas and partition specs.

An Iceberg snapshot belongs to an Iceberg table, not a single partition. The resource identity held by the update request identifies that table, while `snapshot_id` identifies the table state that committed the result.

Iceberg and PostgreSQL cannot share one atomic transaction. The Spark worker commits Iceberg first, then reports success to the API. If the callback fails after a successful Iceberg write, reconciliation can locate the Iceberg commit and complete the PostgreSQL receipt. The API must never create a successful partition receipt before the Iceberg transaction succeeds.

## Feature lineage

`feature_partition_update_request_dataset_dependencies` records the immutable input plan for each pending feature-partition request. `feature_partition_dataset_dependencies` records the same exact dataset partition results once the feature receipt has completed. Both reference `dataset_partitions`, not dataset update requests, so planned or completed lineage cannot refer to an unmaterialized, failed, or cancelled dataset request.

The dependency-parent index supports finding feature outputs affected by a corrected dataset partition. The feature materializer can compare the current parent result hashes and snapshots with recorded lineage to determine which feature partitions need replacement.

## Partition revisions, freshness, and idempotency

Dataset and feature partitions are revision histories. A logical partition is identified by its resource `(name, version)` and timestamp; a completed update request and its partition receipt identify one immutable revision of that logical partition. The current revision is the newest successful receipt for that logical identity. Older receipts remain available for lineage and Iceberg time-travel audits.

### Dataset requests

A normal dataset materialization request is **create-missing**. If a successful current receipt already exists for a requested logical dataset partition, the API returns that receipt and does not fetch or overwrite source data. Replacing source data is explicit through a refresh or forced-correction request. This avoids silently changing the source-of-record history used by features, labels, backtests, and models.

An explicit dataset refresh creates a new dataset-partition revision only after its Spark job commits an Iceberg snapshot. The API then treats that receipt as current while retaining the old revision for reproducibility.

### Feature requests

A normal feature materialization request is **ensure-current**. For each requested feature partition, the API resolves the dependency range from the feature's `datasets` metadata, then selects the current successful revision of every required dataset partition from PostgreSQL. It compares those exact dataset partition IDs with the dependency rows recorded for the latest successful feature partition revision.

| Condition | API action |
| --- | --- |
| No completed feature partition exists | Create a feature job. |
| Completed feature dependencies equal current dataset revisions | Return the existing feature receipt; do not run Spark. |
| Any required dataset partition has a newer revision | Create a replacement feature job for that output partition. |
| A required dataset partition is missing or non-terminal | Do not schedule the feature partition until its inputs are available. |

The comparison uses partition revision IDs as the authoritative identity, with `content_hash` and Iceberg `snapshot_id` retained as integrity and audit data. A changed upstream partition therefore invalidates only feature output partitions whose inferred lookback ranges include that input.

The API should expose three request modes:

- `ensure_current` (default): create missing output and replace only stale output;
- `create_only`: return existing output without replacement; and
- `force`: recompute regardless of dependency equality.

The API enforces a single active request for a logical feature partition. A repeated request joins the active job when its planned dependency revisions match, rather than creating a second queued write. This is separate from Kueue admission: PostgreSQL decides whether work is necessary and deduplicates it; Kueue only controls when eligible work runs.

## Deletion and retention

Foreign keys intentionally use restrictive deletion behavior. A job, request, resource definition, or partition receipt with dependents cannot be hard-deleted accidentally.

Soft deletion is not yet modeled. When deletion endpoints are added, use an explicit API-managed deletion marker such as `deleted_at` rather than deleting rows. Retained records preserve job history, Iceberg snapshot references, and feature lineage.

## API responsibilities

The database enforces data types, primary keys, foreign keys, and non-null columns. The FastAPI service is responsible for:

- validating request resource type against job type;
- validating legal status transitions;
- updating `updated_at` when mutable rows change;
- atomically recording a successful partition receipt and request status;
- providing idempotent completion callbacks;
- enforcing soft deletion when it is introduced; and
- adding indexes when implemented API queries demonstrate a need.

## Feature partition materialization

Feature materialization separates control-plane decisions from user-defined Spark transformations. The API decides which output partitions to create, pins exact upstream dataset-partition revisions, and persists that immutable plan. The Spark worker reads only those pinned Iceberg inputs, invokes the transform, writes Iceberg output, and reports the committed result.

### API request and job creation

A client requests materialization for a registered `(feature_name, feature_version)` and a set of logical feature partition timestamps. The API:

1. Validates each timestamp against the feature's `frequency`.
2. Creates one `feature_partition_update_requests` row per requested partition with status `pending`.
3. Creates a `feature_materialization` job containing those requests.
4. Dispatches a Spark worker with the job ID and partition request IDs.

The API request IDs, rather than a YAML glob or a scan for missing output, define the work for the Spark job. A job may not materialize unrequested feature partitions.

### Dependency resolution

The worker loads the registered feature metadata, including its transform path, parameters, and `feature_dataset_dependencies` rows. Each dependency declares:

- the dataset `(name, version)`;
- `input_name`, the name of the transform constructor argument; and
- `num_lookback_partitions`, the number of preceding available source partitions required for input coverage and lineage.

For every requested feature partition, the API resolves the requested output partition and the preceding calendar dataset partitions implied by `num_lookback_partitions` before creating a job. Dataset materialization writes an empty receipt for a day with no market rows, so every required calendar partition remains part of the reproducible input range. The API persists the planned exact completed dataset partition IDs with the request; those immutable receipts retain the corresponding content hashes and Iceberg snapshots. The worker receives the immutable plan.

The worker reads the resolved Iceberg input partitions into Spark and injects each as a `Dataset`, which carries both the filtered `DataFrame` and its metadata. The dataset dependency must exist in the registry before feature creation; the PostgreSQL foreign key enforces this requirement.

### Transform contract

A feature transform is a frozen dataclass implementing `FeatureTransform`. Its constructor receives a named `Dataset` for each declared `input_name` and typed values from the feature `parameters` object.

```python
@dataclass(frozen=True)
class TrailingVolatilityTransform(FeatureTransform):
    market_data: Dataset
    lookback_bars: int

    def __call__(self) -> DataFrame:
        ...
```

The materialization harness verifies that declared dependency names exactly match dataset constructor arguments, validates parameter names and types, and invokes the transform. The transform can perform arbitrary Spark operations, but it does not choose input partitions, write output storage, or create lineage records.

The transform returns a Spark `DataFrame` with exactly these logical columns, where the middle columns are the registered ordered `feature_keys` (currently `[symbol]`):

```text
timestamp, symbol, feature_value
```

The harness always enforces the following invariants before any Iceberg write:

- the returned column set is exactly `timestamp`, the configured feature keys, and `feature_value`;
- `timestamp`, every feature key, and `feature_value` are non-null, and `feature_value` is castable to the registered `dtype`;
- each `(timestamp, feature_keys)` tuple is unique; and
- the emitted output-partition set contains no partition outside the requested output-partition set.

The feature YAML or API `validations` field is a list of tagged `ValidationConfig` objects:

```yaml
validations:
  - type: minimum_rows
    minimum_rows: 1
    temporal_filter:
      type: all_partitions
  - type: finite_feature_value
  - type: feature_keys_from_input
```

`MaterializationValidator` is the shared pre-write contract for every materializer. It always validates required or exact columns, non-null persisted values, unique logical keys, unexpected output partitions, and static transform layout declarations. Each declared validation independently contributes its check. `minimum_rows` applies to each requested partition selected by its `temporal_filter`, which defaults to `all_partitions`; partitions outside that filter may complete empty. `finite_feature_value` rejects `NaN`, positive infinity, and negative infinity after the dtype cast. `feature_keys_from_input` rejects feature-key values that do not occur in the injected input datasets.

### Commit and completion

For each successful worker write:

1. Spark writes only the requested feature partitions to the versioned feature Iceberg table in one Iceberg transaction.
2. The worker computes committed per-partition content hashes and row counts, reads the table schema and Iceberg `snapshot_id`, then sends all receipts in one `POST /partitions/complete` batch.
3. The API validates every receipt and its exact completed dataset partition IDs against the immutable input plan before writing any database rows.
4. In one PostgreSQL transaction, the API creates all `feature_partitions` receipts, inserts `feature_partition_dataset_dependencies` lineage rows, and changes their request statuses from `pending` to `created`.

If a worker fails before this completion transaction, its unfinished requests become `failed`. If its job is cancelled before completion, unfinished requests become `cancelled`. A completion callback is idempotent only when its existing receipt has the same content hash, snapshot ID, row count, schema, and lineage; a differing callback is a conflict.

### Responsibilities

| Component | Responsibilities |
| --- | --- |
| API | Resource registration, validation, request/job lifecycle, completion receipt, lineage persistence, and idempotency. |
| Spark materialization harness | Read API-pinned input partitions, construct `Dataset` inputs, validate and invoke transforms, filter requested outputs, write Iceberg, and report completion. |
| Feature transform | Compute feature values from injected Spark datasets and return `timestamp`, `symbol`, and `feature_value`. |
| Iceberg | Store dataset and feature rows atomically and expose committed snapshots for partition receipts. |
