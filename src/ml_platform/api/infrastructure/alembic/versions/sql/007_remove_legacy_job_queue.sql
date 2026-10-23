-- Kueue holds submitted Kubernetes workloads; PostgreSQL retains only durable job outcomes.
DROP TABLE IF EXISTS job_queue;
