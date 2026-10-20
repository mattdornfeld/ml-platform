# Materialization control plane

The PostgreSQL control plane records the intent and outcome of materializing Iceberg-backed datasets and features. Spark owns table schemas and writes; PostgreSQL stores resource definitions, work state, completed partition receipts, and lineage.

The initial database schema is [src/api/sql/schema.sql](../src/api/sql/schema.sql).

## Resources

Datasets and features are distinct resources, each identified by an immutable `(name, version)` pair. A dataset has an opaque `config JSONB` document. Spark and Iceberg own the dataset's physical schema.

Feature registration is normalized rather than stored as an opaque configuration. A feature has `dtype`, `frequency`, `feature_key`, `transform`, `description`, `parameters`, and `validations` fields. Its `datasets` payload creates rows in `feature_dataset_dependencies`; each row identifies an existing `(dataset_name, dataset_version)` and its `num_lookback_partitions`. Foreign keys prevent registering a feature dependency on a dataset version that does not exist.

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
5. After a successful Iceberg commit, the worker calls the API with the request UUID, content hash, and Iceberg snapshot ID.
6. In one PostgreSQL transaction, the API creates the partition result and marks its request `created`.

A `dataset_partitions` or `feature_partitions` row is therefore a receipt for a successful Iceberg write. It is keyed by the request UUID and stores only `content_hash`, `snapshot_id`, and `created_at`; its dataset or feature identity, timestamp, job, and lifecycle status are obtained by joining to the corresponding request.

The API must make completion callbacks idempotent. A repeated callback with the same hash and snapshot may succeed without changing state. A callback for an existing request with different materialization metadata must fail with a conflict.

## Iceberg consistency

An Iceberg snapshot belongs to an Iceberg table, not a single partition. The resource identity held by the update request identifies that table, while `snapshot_id` identifies the table state that committed the result.

Iceberg and PostgreSQL cannot share one atomic transaction. The Spark worker commits Iceberg first, then reports success to the API. If the callback fails after a successful Iceberg write, reconciliation can locate the Iceberg commit and complete the PostgreSQL receipt. The API must never create a successful partition receipt before the Iceberg transaction succeeds.

## Feature lineage

`feature_partition_dataset_dependencies` records the exact completed dataset partition results consumed by a completed feature partition. It references `dataset_partitions`, not dataset update requests, so lineage cannot refer to an unmaterialized, failed, or cancelled dataset request.

The dependency-parent index supports finding feature outputs affected by a corrected dataset partition. The feature materializer can compare the current parent result hashes and snapshots with recorded lineage to determine which feature partitions need replacement.

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

Feature materialization separates control-plane decisions from user-defined Spark transformations. The API decides which output partitions to create and which registered feature version applies. The Spark worker resolves and reads required inputs, invokes the transform, writes Iceberg output, and reports the committed result.

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

For every requested feature partition, the worker resolves the requested output partition and its preceding *available* input dataset partitions. It does not use calendar subtraction: weekends, holidays, and incomplete market sessions may leave calendar partitions empty. The resolution result contains exact completed dataset partition IDs, content hashes, and Iceberg snapshots.

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

The transform returns a Spark `DataFrame` with exactly these logical columns, where the middle column is the registered `feature_key` (currently `symbol`):

```text
timestamp, symbol, feature_value
```

The harness always enforces the following invariants before any Iceberg write:

- the returned column set is exactly `timestamp`, the configured feature key, and `feature_value`;
- `timestamp`, the feature key, and `feature_value` are non-null, and `feature_value` is castable to the registered `dtype`;
- each `(timestamp, feature_key)` pair is unique; and
- the emitted output-partition set is exactly the requested output-partition set. Missing and extra partitions both fail the job.

The feature YAML or API `validations` object configures the remaining checks:

```yaml
validations:
  min_rows_per_partition: 1
  require_finite_feature_value: true
  require_feature_keys_from_input: true
```

`min_rows_per_partition` defaults to `1`. When enabled (the default), `require_finite_feature_value` rejects `NaN`, positive infinity, and negative infinity after the dtype cast. When enabled (the default), `require_feature_keys_from_input` rejects feature-key values that do not occur in the injected input datasets. The partition-set check is deliberately not configurable: a worker must never silently omit or add a requested partition.

### Commit and completion

For each successful worker write:

1. Spark writes only the requested feature partitions to the versioned feature Iceberg table in one Iceberg transaction.
2. The worker computes the output content hash and obtains the committed Iceberg `snapshot_id`.
3. The worker calls the API completion endpoint with the feature partition request ID, hash, snapshot ID, and exact completed dataset partition IDs used as inputs.
4. In one PostgreSQL transaction, the API creates the `feature_partitions` receipt, inserts `feature_partition_dataset_dependencies` lineage rows, and changes the request status from `pending` to `created`.

If a worker fails before this completion transaction, its unfinished requests become `failed`. If its job is cancelled before completion, unfinished requests become `cancelled`. A completion callback is idempotent only when its existing receipt has the same content hash, snapshot ID, and lineage; a differing callback is a conflict.

### Responsibilities

| Component | Responsibilities |
| --- | --- |
| API | Resource registration, validation, request/job lifecycle, completion receipt, lineage persistence, and idempotency. |
| Spark materialization harness | Resolve exact input partitions, construct `Dataset` inputs, validate and invoke transforms, filter requested outputs, write Iceberg, and report completion. |
| Feature transform | Compute feature values from injected Spark datasets and return `timestamp`, `symbol`, and `feature_value`. |
| Iceberg | Store dataset and feature rows atomically and expose committed snapshots for partition receipts. |
