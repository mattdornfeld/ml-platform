#!/usr/bin/env bash
set -euo pipefail

minikube start --driver=docker --cpus=8 --memory=49152
minikube image build -t ml-platform:latest .
minikube image build -t ml-platform-minio:latest -f deploy/docker/minio/Dockerfile .
deploy/kubernetes/install.sh
