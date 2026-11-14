# The challenge (AfterQuery track, Junction 2026)

We are building a **reinforcement-learning environment** that challenges today's best AI agents,
demonstrates real failure modes, and has a reward that reflects real progress.

Required: a clear goal, actions the agent can take, and a checkable reward. We run a frontier
agent (OpenCode harness + Gemini) and compare it with a **stronger reference run** we can
demonstrate and reproduce. We must show where the agent falls short and why. The difficulty must
come from the decisions the agent has to make — it must have the information and tools needed to
succeed. Judges will actively try to reward-hack.

Optional stretch / tiebreaker: train a smaller model on experience from the environment and show
it improves.

## Judging rubric (100 p)
- **The gap (30 p):** frontier agent consistently scores below our reference run; both
  reproducible across repeated runs.
- **Meaningful difficulty (30 p):** we can explain the agent's main mistakes and tie them to
  decisions in the task. Zero points if the gap comes from withheld information, trick wording,
  impossible timing or broken tools.
- **Reward integrity (25 p):** reward measures task success and survives exploit attempts.
  Doing nothing or stopping early earns little or nothing; doing the task well is the best way
  to score.
- **Craft (15 p):** rules and behaviour make sense for the setting; judges reproduce the
  evaluation from the README with **one command**.

## How judges run it
10-minute demo, then they rerun with the standard harness: **our reference run, three agent runs,
a do-nothing baseline, and a stop-early baseline**, plus 15 minutes of exploit hunting.

## Required stack (from organizers)
- Framework: **Harbor** (https://github.com/harbor-framework/harbor). The environment is a Harbor
  task: `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/`, `solution/`.
- Harness: OpenCode, e.g. `harbor run -p ./my-task -a opencode -m <provider/model>`.
  The agent connects to our environment over **MCP**.
- Reward: the verifier writes **one number in [0, 1]** to `/logs/verifier/reward.txt`.
  Partial credit is fine.
- Submission must include at least one Harbor run log with its score, and a short README:
  what the agent does, why it's hard, how it's scored.
- No AfterQuery inference credits: we use our own Google (Gemini), OpenAI and Modal credits.
