#!/usr/bin/env bash
#
# run_wordfreq.sh - word frequency measurement series.
#
# Wrapper: all the logic is in run_series.sh, only the task is chosen here.
# Arguments are passed through as-is, see ./run_series.sh --help

exec "$(dirname "${BASH_SOURCE[0]}")/run_series.sh" --task wordfreq "$@"
