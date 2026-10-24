# A self-contained single-node ML Platform Kubernetes cluster.
#
# The image builds the two local workload images and stores OCI archives where
# K3s imports them before scheduling workloads. End-to-end tests therefore
# start one image and wait only for its Docker health status.

FROM ghcr.io/astral-sh/uv:0.9.25 AS uv

FROM maven:3.9.11-eclipse-temurin-21 AS spark_jars
WORKDIR /build
COPY deploy/docker/spark-deps/pom.xml ./pom.xml
RUN mvn --batch-mode dependency:copy-dependencies \
    -DincludeScope=runtime \
    -DoutputDirectory=/jars

FROM apache/spark:4.1.3-python3 AS platform_image
COPY --from=uv /uv /uvx /bin/
COPY --from=spark_jars /jars /opt/spark/jars/
USER root
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv python install 3.12 --install-dir /opt/python \
 && python_path="$(find /opt/python -path '*/bin/python3.12' -type f -print -quit)" \
 && uv venv --python "$python_path" /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps . \
 && uv pip install --python /opt/venv/bin/python \
      'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'psycopg[binary]>=3.2' \
      'pydantic>=2.0' 'kubernetes>=32.0' 'omegaconf>=2.3' \
      'simple-classproperty>=4.0.2' 'exchange-calendars>=4.13,<5' \
      'pyarrow>=26.0.0' 'ipython>=9.0' \
 && mkdir -p /tmp/.ivy2 /tmp/.cache \
 && chown -R 185:0 /app /opt/venv /tmp/.ivy2 /tmp/.cache
ENV PATH=/opt/venv/bin:${PATH}
ENV PYTHONPATH=/app/src:/opt/spark/python/lib/pyspark.zip:/opt/spark/python/lib/py4j-0.10.9.9-src.zip
ENV PYSPARK_PYTHON=/opt/venv/bin/python
ENV HOME=/tmp
USER 185

FROM golang:1.24-alpine AS minio_builder
RUN apk add --no-cache git
RUN GOBIN=/out go install github.com/minio/minio@RELEASE.2025-04-22T22-12-26Z \
 && GOBIN=/out go install github.com/minio/mc@RELEASE.2025-04-16T18-13-26Z

FROM alpine:3.21 AS minio_image
COPY --from=minio_builder /out/minio /usr/local/bin/minio
COPY --from=minio_builder /out/mc /usr/local/bin/mc
ENTRYPOINT ["/usr/local/bin/minio"]

FROM gcr.io/go-containerregistry/crane:debug AS crane

FROM alpine:3.21 AS workload_archives
RUN apk add --no-cache tar
COPY --from=crane /ko-app/crane /usr/local/bin/crane
COPY --from=platform_image / /rootfs/platform
COPY --from=minio_image / /rootfs/minio
RUN mkdir /images \
 && tar -C /rootfs/platform -cf /tmp/platform.tar . \
 && crane mutate apache/spark:4.1.3-python3 \
      --append /tmp/platform.tar \
      --env PATH=/opt/venv/bin:/opt/java/openjdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      --env PYTHONPATH=/app/src:/opt/spark/python/lib/pyspark.zip:/opt/spark/python/lib/py4j-0.10.9.9-src.zip \
      --env PYSPARK_PYTHON=/opt/venv/bin/python \
      --env HOME=/tmp \
      --user 185 \
      --workdir /app \
      --entrypoint /opt/entrypoint.sh \
      --tag ml-platform:latest \
      --output /images/ml-platform.tar \
 && tar -C /rootfs/minio -cf /tmp/minio.tar . \
 && crane mutate alpine:3.21 \
      --append /tmp/minio.tar \
      --entrypoint /usr/local/bin/minio \
      --tag ml-platform-minio:latest \
      --output /images/ml-platform-minio.tar

FROM rancher/k3s:v1.31.4-k3s1

COPY deploy/kubernetes /opt/ml-platform/kubernetes
COPY --from=workload_archives /images /var/lib/rancher/k3s/agent/images
COPY deploy/docker/k3s/k3s.yaml /etc/rancher/k3s/config.yaml
COPY deploy/docker/k3s/entrypoint.sh /usr/local/bin/ml-platform-k3s
COPY deploy/docker/k3s/healthcheck.sh /usr/local/bin/ml-platform-k3s-healthcheck
RUN chmod 755 /usr/local/bin/ml-platform-k3s \
 && chmod 755 /usr/local/bin/ml-platform-k3s-healthcheck

HEALTHCHECK --interval=2s --timeout=1s --start-period=10s --retries=90 \
  CMD ["/usr/local/bin/ml-platform-k3s-healthcheck"]

ENTRYPOINT ["/usr/local/bin/ml-platform-k3s"]
CMD ["server", "--disable", "traefik"]
