#!/bin/bash
# for dir in ./tfg_custom_ops/*;
# do
#   if [[ $dir == *"shared" ]]
#   then
#     echo 'skip' $dir
#   else
#     ebazel build 'tfg_custom_ops/'$dir":python/ops/_"$dir"_ops.so"
#   fi
# done
op="compute_pdf"
bazel build 'tfg_custom_ops/'$op":python/ops/_"$op"_ops.so" --verbose_failures