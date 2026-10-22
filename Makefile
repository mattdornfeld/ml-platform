UV ?= uv
PYTHON_VERSION ?= 3.12
VIRTUALENVWRAPPER_CONFIG ?= $(HOME)/.zsh_configs/uv-virtualenvwrapper.sh
VENV_NAME ?= ml-platform
WORKON_HOME ?= $(HOME)/.virtualenvs
VENV_PATH := $(WORKON_HOME)/$(VENV_NAME)
SPARK_JAVA_HOME ?= $(shell brew --prefix openjdk@21 2>/dev/null || /usr/libexec/java_home 2>/dev/null)

.PHONY: configure venv lock test ui headlamp minio minio-console

## Create the named environment with mkvirtualenv, then install locked dependencies.
configure: venv
	UV_PROJECT_ENVIRONMENT="$(VENV_PATH)" $(UV) sync --locked --python $(PYTHON_VERSION)

## Create the environment using mkvirtualenv and associate this directory with it.
venv:
	@zsh -c 'source "$(VIRTUALENVWRAPPER_CONFIG)"; if [[ ! -d "$(VENV_PATH)" ]]; then mkvirtualenv --python "$(PYTHON_VERSION)" "$(VENV_NAME)"; fi; workon "$(VENV_NAME)"; setvirtualenvproject "$(CURDIR)"'
	@printf 'Activate and enter this project with: workon %s\n' "$(VENV_NAME)"

## Refresh uv.lock after changing project dependencies.
lock: venv
	UV_PROJECT_ENVIRONMENT="$(VENV_PATH)" $(UV) lock

## Run pytest, including production mypy checks and available Spark/Iceberg integration tests.
test: configure
	JAVA_HOME="$(SPARK_JAVA_HOME)" UV_PROJECT_ENVIRONMENT="$(VENV_PATH)" $(UV) run --locked pytest -v

## Serve the control-plane dashboard from the local Minikube cluster.
ui:
	kubectl -n ml-platform port-forward service/ml-control-plane 8001:8000

## Serve the in-cluster Headlamp Kubernetes UI.
headlamp:
	kubectl -n headlamp port-forward service/headlamp 8002:80

## Serve the public in-cluster MinIO warehouse endpoint.
minio:
	kubectl -n ml-platform port-forward service/minio 9004:9000

## Serve the authenticated in-cluster MinIO Console.
minio-console:
	kubectl -n ml-platform port-forward service/minio 9003:9001
