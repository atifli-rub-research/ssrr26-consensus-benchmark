"""Default simulation, evaluation, and consensus strategy parameters."""

# Default output directory for the YAML pipeline.
OUTPUT_FILES_DIR = "output_files"
MAX_UAV_COUNT = 10

# Monte-Carlo: multiple runs with reproducible start tiles, then average metrics.
# Set ENABLE_MONTE_CARLO_AVERAGING = True to activate (uses NUM_SIMULATION_RUNS).
ENABLE_MONTE_CARLO_AVERAGING = True
NUM_SIMULATION_RUNS = 10


def effective_simulation_runs(cli_runs: int | None = None) -> int:
    """Return number of runs: CLI override, else 1 or NUM_SIMULATION_RUNS if averaging enabled."""
    if cli_runs is not None:
        return max(1, int(cli_runs))
    if ENABLE_MONTE_CARLO_AVERAGING:
        return NUM_SIMULATION_RUNS
    return 1

# WMSR: max Byzantine agents tolerated (must satisfy n > 3f)
WMSR_F = 1

# DBBS: weight of social belief in [0, 1]
DBBS_MU = 0.5

# MLES
MLES_LAMBDA = 0.1

# CDCI
CDCI_GAMMA = 0.8
CDCI_RHO = 0.02
CDCI_ALPHA = 0.1
CDCI_SIGMA = 0.02

# Opinion mapping from detector confidence
NO_DETECTION_OPINION = -1.0

# All mine slots: consensus_value >= threshold → mine accepted at that slot
DECISION_THRESHOLD = -0.6

# Max mine slots per tile (consensus per detection rank)
MAX_MINE_SLOTS_PER_TILE = 5

# Instance metrics: greedy IoU matching (tile-local boxes)
INSTANCE_IOU_THRESHOLD = 0.5

# Extra consensus iterations per tile after all messages (0 = only on receive)
POST_STEP_CONSENSUS_ROUNDS = 0
