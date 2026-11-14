"""Profile a single episode of BayesAgent with cProfile."""

import cProfile
from pathlib import Path
import pstats
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agents.bayes import BayesAgent
from beamsim.env_api import BeamlineEnv, evaluate_env


def main():
    env = BeamlineEnv(seed=0, budget=40, randomization="medium")
    agent = BayesAgent(n_particles=50_000, chunk_size=10_000, seed=0)

    pr = cProfile.Profile()
    pr.enable()
    agent.solve(env)
    pr.disable()

    score = evaluate_env(env)
    print(f"Episode finished: Sf={score.sf:.4f}, shots={score.shots_used}")

    ps = pstats.Stats(pr).sort_stats("cumtime")
    print("\n--- TOP 20 BY CUMULATIVE TIME ---")
    ps.print_stats(20)

    print("\n--- TOP 20 BY TOTAL (INTERNAL) TIME ---")
    ps.sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
