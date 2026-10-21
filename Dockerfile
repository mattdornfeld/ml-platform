FROM ghcr.io/astral-sh/uv:0.9.25 AS uv

FROM maven:3.9.11-eclipse-temurin-21 AS spark_jars
WORKDIR /build
COPY deploy/docker/spark-deps/pom.xml ./pom.xml
RUN mvn --batch-mode dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=/jars

FROM apache/spark:4.1.3-python3
COPY --from=uv /uv /uvx /bin/
COPY --from=spark_jars /jars /opt/spark/jars/
USER root
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv python install 3.12 --install-dir /opt/python \
 && uv venv --python /opt/python/cpython-3.12.12-linux-aarch64-gnu/bin/python3.12 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps . \
 && uv pip install --python /opt/venv/bin/python \
      'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'psycopg[binary]>=3.2' \
      'pydantic>=2.0' 'kubernetes>=32.0' 'omegaconf>=2.3' 'hydra-core>=1.3' \
 && mkdir -p /tmp/.ivy2 /tmp/.cache \
 && chown -R 185:0 /app /opt/venv /tmp/.ivy2 /tmp/.cache
ENV PATH=/opt/venv/bin:${PATH}
ENV PYTHONPATH=/app/src
ENV PYSPARK_PYTHON=/opt/venv/bin/python
ENV HOME=/tmp
USER 185
