# m10dis: shared paths for every runner script (sourced, not executed).
# Override any of these in the environment or in m10dis.local.env at the repo root.
M10DIS_HOME=${M10DIS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
[[ -f $M10DIS_HOME/m10dis.local.env ]] && source "$M10DIS_HOME/m10dis.local.env"
M10DIS_RUNS=${M10DIS_RUNS:-$M10DIS_HOME/runs}            # one folder per run (logs, manifests, results)
M10DIS_DATA=${M10DIS_DATA:-/mnt/nvme}                     # fast local disk for model files and caches
M10DIS_MODEL=${M10DIS_MODEL:-$M10DIS_DATA/models/Qwen3.8-Flash-Next-FP8}
M10DIS_BENCH_DATA=${M10DIS_BENCH_DATA:-$M10DIS_DATA/benchmarks}
M10DIS_STATE=${M10DIS_STATE:-$M10DIS_DATA/m10dis-state}          # machine-specific prepared inputs (docs/RUNNING.md)
export M10DIS_HOME M10DIS_RUNS M10DIS_DATA M10DIS_MODEL M10DIS_BENCH_DATA M10DIS_STATE
