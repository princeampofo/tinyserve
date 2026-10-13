#!/usr/bin/env bash
# Delete the local tinyserve cluster (and everything in it).
set -euo pipefail
kind delete cluster --name tinyserve
