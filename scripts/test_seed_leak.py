"""Check for seed leak between BeamlineSim and BayesAgent across seeds 0-9."""

import time
from beamsim.env_api import BeamlineEnv, evaluate_env
from agents.bayes import BayesAgent


def main():
    print(f"{'Seed':<6} | {'(a) Same Seed':<16} | {'(b) Indep (+1M+3)':<22} | {'Diff':<8}")
    print("-" * 58)

    rewards_a = []
    rewards_b = []

    for s in range(10):
        # (a) agent seed = env seed
        env_a = BeamlineEnv(seed=s, budget=60, randomization="medium")
        agent_a = BayesAgent(n_particles=40_000, roughening=0.02, seed=s)
        agent_a.solve(env_a)
        sc_a = evaluate_env(env_a)
        rew_a = sc_a.reward
        rewards_a.append(rew_a)

        # (b) independent agent seed = env seed + 1_000_003
        env_b = BeamlineEnv(seed=s, budget=60, randomization="medium")
        agent_b = BayesAgent(n_particles=40_000, roughening=0.02, seed=s + 1_000_003)
        agent_b.solve(env_b)
        sc_b = evaluate_env(env_b)
        rew_b = sc_b.reward
        rewards_b.append(rew_b)

        diff = rew_b - rew_a
        print(f"{s:<6d} | {rew_a:<16.4f} | {rew_b:<22.4f} | {diff:+7.4f}")

    print("-" * 58)
    mean_a = sum(rewards_a) / len(rewards_a)
    mean_b = sum(rewards_b) / len(rewards_b)
    print(f"{'Mean':<6} | {mean_a:<16.4f} | {mean_b:<22.4f} | {mean_b - mean_a:+7.4f}")


if __name__ == "__main__":
    main()
