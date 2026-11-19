#!/usr/bin/env bash
#
# run_matmul.sh - matrix multiplication measurement series.
#
# Wrapper: all the logic is in run_series.sh, only the task is chosen here.
# This guarantees that all three tasks are measured with THE SAME protocol
# (shuffled order, alternating languages within a point, environment check,
# result check) - otherwise their results could not be compared with each
# other.
#
# Arguments are passed through as-is, see ./run_series.sh --help

exec "$(dirname "${BASH_SOURCE[0]}")/run_series.sh" --task matmul "$@"
