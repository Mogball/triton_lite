#!/usr/bin/env bash

# Auto-generated build helper that compiles the Mojo kernels into shared
# libraries and then compiles the C-based Python bindings into `bindings.so`.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Compile Mojo kernels → shared libraries (*.so)
# ---------------------------------------------------------------------------

magic run mojo build --emit shared-lib kernels/make_4d_causal_mask.mojo
magic run mojo build --emit shared-lib kernels/apply_rotary_emb.mojo

# ---------------------------------------------------------------------------
# 2. Compile C Python extension
# ---------------------------------------------------------------------------

gcc -shared -fPIC $(python3-config --includes) bindings.c -o bindings.so -ldl

echo "[build_bindings.sh] Build complete: bindings.so + kernel shared libs generated."
