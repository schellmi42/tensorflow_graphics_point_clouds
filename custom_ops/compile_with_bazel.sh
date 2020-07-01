#!/usr/bin/env bash
# to test the compilation of a single op
op="find_neighbors"
bazel clean
bazel build 'tfg_custom_ops/'$op":python/ops/_"$op"_ops.so" --verbose_failures