import os
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from ecoli_bioreactor_v2 import CoupledEColiBioreactorEnv, solve_reference_policy


# Automatically picks up GEMINI_API_KEY from your environment
client = genai.Client()

SYSTEM_PROMPT = """You are an expert bioprocess control engineer operating a fed-batch E. coli bioreactor producing bioethanol.
Your goal is to maximize terminal net profit (Bioethanol revenue minus Glucose, Inducer, and Utility costs) without killing the culture (viability must stay >= 0.20).

Key coupled dynamics to manage:
1. Biphasic Strategy: Grow biomass aerobically at ~37°C first (low pdc/adhB), then shift to production at ~31°C (higher pdc/adhB) so enzymes fold properly.
2. Stoichiometric Balance: pdc converts Pyruvate -> toxic Acetaldehyde; adhB converts Acetaldehyde -> Ethanol. Keep adh_expression >= pdc_expression to prevent acetaldehyde spikes.
3. Weak-Acid Amplification: Low pH (< 6.8) exponentially increases acetate toxicity. Use base_addition proportionally to hold pH near 7.0 without overshooting above 7.8.
4. Oxygen Mass Transfer (kLa): At high biomass, dissolved O2 drops even at max o2_sparge. Throttle glucose_feed if DO drops below 25% to prevent acetate overflow.
5. Thermal Runaway & Maintenance: High metabolism generates heat. Watch `scheduled_cooling_maintenance_hours` and pre-cool ahead of maintenance windows."""


class BioreactorAction(BaseModel):
    reasoning: str = Field(description="Brief 1-2 sentence analysis of current state trends, coupled risks, and chosen setpoints.")
    glucose_feed: float = Field(description="Glucose feed rate in g/L/hr within action_bounds.")
    o2_sparge: float = Field(description="Aeration and agitation intensity between 0.0 and 1.0.")
    pdc_expression: float = Field(description="Pyruvate decarboxylase (pdc) expression level between 0.0 and 2.0.")
    adh_expression: float = Field(description="Alcohol dehydrogenase (adhB) expression level between 0.0 and 2.0.")
    cooling_power: float = Field(description="Cooling jacket valve power between 0.0 and 1.0.")
    base_addition: float = Field(description="Alkaline buffer (NH4OH/NaOH) dosing rate between 0.0 and 1.0.")
    harvest: bool = Field(description="Set true only if ending the batch early to lock in profit.")


def classify_failure(trajectory: List[Dict[str, Any]], crash_reason: Optional[str]) -> str:
    """Categorizes trajectory outcomes into concrete failure modes."""
    if crash_reason is None:
        final_reward = trajectory[-1]["reward"]
        if final_reward <= 0:
            return "SURVIVED_UNPROFITABLE (Over-spent substrates/utilities)"
        return "SUCCESS"

    reason_lower = crash_reason.lower()
    if "thermal" in reason_lower:
        return "CRASH_THERMAL_RUNAWAY"
    if "acidification" in reason_lower:
        return "CRASH_ACIDIFICATION_WEAK_ACID"
    if "alkaline" in reason_lower:
        return "CRASH_ALKALINE_OVERDOSE"
    if "acetaldehyde" in reason_lower:
        return "CRASH_ACETALDEHYDE_IMBALANCE (pdc > adhB)"
    if "acetate" in reason_lower:
        return "CRASH_HYPOXIA_OVERFLOW (O2 bottleneck)"
    if "burden" in reason_lower:
        return "CRASH_METABOLIC_BURDEN (Premature/excess induction)"
    return f"CRASH_OTHER ({crash_reason})"


def run_gemini_benchmark(
    model_id: str = "gemini-3.8-flash",
    difficulties: List[int] = [1, 2, 3, 4, 5],
    save_path: str = "gemini_trajectories.json",
):
    results_summary = []
    all_logs = []

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=BioreactorAction,
    )

    for diff in difficulties:
        print(f"\n--- Running Difficulty {diff} with {model_id} ---")
        ref = solve_reference_policy(difficulty=diff)
        print(f"Reference Oracle Reward: {ref['reference_reward']:.2f} | Final EtOH: {ref['final_state']['bioethanol_g_L']:.2f} g/L")

        env = CoupledEColiBioreactorEnv(difficulty=diff)
        obs = env.reset()

        # Stateful chat preserves multi-step trajectory momentum across the episode
        chat = client.chats.create(model=model_id, config=config)
        trajectory = []
        total_reward = 0.0

        while not obs["terminated"]:
            prompt = f"Current Bioreactor Observation:\n{json.dumps(obs, indent=2)}\n\nProvide your next control action."
            try:
                response = chat.send_message(prompt)
                parsed_action: BioreactorAction = response.parsed
                action_dict = parsed_action.model_dump()
            except Exception as e:
                print(f"Gemini API error at step {obs['step']}: {e}")
                break

            next_obs, reward, done, _ = env.step(action_dict)
            trajectory.append({
                "step": obs["step"],
                "obs": obs["state"],
                "reasoning": action_dict["reasoning"],
                "action": {k: v for k, v in action_dict.items() if k != "reasoning"},
                "reward": reward,
            })
            obs = next_obs
            total_reward += reward
            time.sleep(0.15)  # Rate-limit cushion

        outcome_category = classify_failure(trajectory, obs["crash_reason"])
        final_st = obs["state"]

        summary_row = {
            "difficulty": diff,
            "model": model_id,
            "gemini_reward": round(total_reward, 2),
            "reference_reward": ref["reference_reward"],
            "gap": round(ref["reference_reward"] - total_reward, 2),
            "steps_survived": f"{obs['step']}/{env.cfg.max_steps}",
            "final_etoh_g_L": final_st["bioethanol_g_L"],
            "outcome": outcome_category,
        }
        results_summary.append(summary_row)
        all_logs.append({"summary": summary_row, "trajectory": trajectory})

        print(
            f"Gemini Result -> Reward: {total_reward:>6.2f} | Survived: {summary_row['steps_survived']} | "
            f"EtOH: {final_st['bioethanol_g_L']:>5.2f} g/L | Outcome: {outcome_category}"
        )

    with open(save_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"\nSaved full trajectories and reasoning traces to {save_path}")
    return results_summary


if __name__ == "__main__":
    # Current Gemini API model IDs to compare:
    # - "gemini-3.8-flash" (fast, strong agentic baseline)
    # - "gemini-3.1-pro-preview" (deeper multi-step reasoning)
    run_gemini_benchmark(model_id="gemini-3.8-flash", difficulties=[1, 2, 3, 4, 5])
