UV ?= uv
PYTHON_VERSION ?= 3.12
VIRTUALENVWRAPPER_CONFIG ?= $(HOME)/.zsh_configs/uv-virtualenvwrapper.sh
VENV_NAME ?= pytorch-tutorial
WORKON_HOME ?= $(HOME)/.virtualenvs
VENV_PATH := $(WORKON_HOME)/$(VENV_NAME)
SPARK_JAVA_HOME ?= $(shell brew --prefix openjdk@21 2>/dev/null || /usr/libexec/java_home 2>/dev/null)

.PHONY: configure venv lock test

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

## Run the complete test suite, including the Spark/Iceberg integration test.
test: configure
	JAVA_HOME="$(SPARK_JAVA_HOME)" UV_PROJECT_ENVIRONMENT="$(VENV_PATH)" $(UV) run --locked python -m unittest discover -s tests -v
