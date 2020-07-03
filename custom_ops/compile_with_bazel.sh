#!/usr/bin/env bash
# to test the compilation of a single op
op="compute_keys"
# bazel clean
bazel build 'tfg_custom_ops/'$op":python/ops/_"$op"_ops.so" --verbose_failures
bazel test 'tfg_custom_ops/'$op:$op'_ops_py_test'