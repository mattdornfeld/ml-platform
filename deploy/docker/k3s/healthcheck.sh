#!/bin/sh
# Docker reports healthy only after the platform bootstrap completed.
set -eu

test -f /run/ml-platform/bootstrap-complete
