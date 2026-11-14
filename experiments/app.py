import os
import json
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from ecoli_bioreactor_v2 import CoupledEColiBioreactorEnv, solve_reference_policy
from run_gemini_agent import SYSTEM_PROMPT, BioreactorAction, classify_failure

st.set_page_config(page_title="E. coli Bioethanol RL Benchmark", layout="wide")
st.title("🧬 Fed-Batch E. coli Bioethanol Bioreactor: LLM Breaking-Point Benchmark")
st.caption("Testing frontier LLM multi-step control against a closed-loop SciPy MPC oracle under coupled thermal, pH, and kLa bottlenecks.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Experiment Configuration")
    difficulty = st.slider("Difficulty Tier", min_value=1, max_value=5, value=3)
    model_id = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    
    # Uses Streamlit Cloud secret if set, otherwise lets user/judge paste a key
    default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    
    run_btn = st.button("🚀 Run Live Benchmark", type="primary", use_container_width=True)

# --- Main Execution ---
if run_btn:
    if not api_key:
        st.error("Please enter a Gemini API key in the sidebar (or configure GEMINI_API_KEY in Streamlit Secrets).")
        st.stop()

    col1, col2 = st.columns(2)

    # 1. Run Reproducible Reference Oracle
    with st.spinner(f"Solving closed-loop reference policy (Difficulty {difficulty})..."):
        ref_out = solve_reference_policy(difficulty=difficulty)
        ref_df = pd.DataFrame([
            {"step": t["obs"]["step"], **t["obs"]["state"]} for t in ref_out["trajectory"]
        ])

    with col1:
        st.subheader("🎯 Reference Oracle (SciPy MPC)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Net Profit", f"${ref_out['reference_reward']:.2f}")
        m2.metric("Final Bioethanol", f"{ref_out['final_state']['bioethanol_g_L']:.2f} g/L")
        m3.metric("Final Viability", f"{ref_out['final_state']['viability']:.2f}")

    # 2. Run Live Gemini Agent
    with col2:
        st.subheader(f"🤖 {model_id} Agent")
        progress_bar = st.progress(0)
        status_text = st.empty()

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=BioreactorAction,
        )

        env = CoupledEColiBioreactorEnv(difficulty=difficulty)
        obs = env.reset()
        chat = client.chats.create(model=model_id, config=config)

        trajectory = []
        total_reward = 0.0

        while not obs["terminated"]:
            status_text.text(f"Simulating Hour {obs['step']} / {env.cfg.max_steps}...")
            prompt = f"Current Bioreactor Observation:\n{json.dumps(obs, indent=2)}\n\nProvide your next control action."
            response = chat.send_message(prompt)
            action_dict = response.parsed.model_dump()

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
            progress_bar.progress(min(1.0, obs["step"] / env.cfg.max_steps))

        status_text.empty()
        outcome = classify_failure(trajectory, obs["crash_reason"])
        final_st = obs["state"]

        g1, g2, g3 = st.columns(3)
        g1.metric("Net Profit", f"${total_reward:.2f}", delta=f"{total_reward - ref_out['reference_reward']:.2f}")
        g2.metric("Final Bioethanol", f"{final_st['bioethanol_g_L']:.2f} g/L")
        g3.metric("Survived", f"{obs['step']}/{env.cfg.max_steps} hrs")

        if outcome == "SUCCESS":
            st.success(f"Outcome: {outcome}")
        else:
            st.error(f"Outcome: {outcome}")
            if obs["crash_reason"]:
                st.warning(f"Diagnostic: {obs['crash_reason']}")

    # --- Trajectory Comparison Charts ---
    st.divider()
    st.subheader("📈 Coupled Bioreactor Trajectories: Agent vs. Reference")
    agent_df = pd.DataFrame([{"step": t["step"], **t["obs"]} for t in trajectory])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Bioethanol Yield (g/L)**")
        chart_etoh = pd.DataFrame({
            "Reference": ref_df.set_index("step")["bioethanol_g_L"],
            "Gemini Agent": agent_df.set_index("step")["bioethanol_g_L"],
        })
        st.line_chart(chart_etoh)

    with c2:
        st.markdown("**Culture Viability & Dissolved O2**")
        chart_health = pd.DataFrame({
            "Agent Viability": agent_df.set_index("step")["viability"],
            "Ref Viability": ref_df.set_index("step")["viability"],
        })
        st.line_chart(chart_health)

    with c3:
        st.markdown("**Agent Temperature (°C) & pH**")
        chart_phys = pd.DataFrame({
            "Temp (°C)": agent_df.set_index("step")["temperature_C"],
            "pH x 5": agent_df.set_index("step")["pH"] * 5.0,
        })
        st.line_chart(chart_phys)

    # --- Step-by-Step Reasoning Log ---
    st.subheader("🧠 Agent Step-by-Step Reasoning & Control Log")
    log_rows = [
        {
            "Hour": t["step"],
            "EtOH (g/L)": t["obs"]["bioethanol_g_L"],
            "Temp (°C)": t["obs"]["temperature_C"],
            "pH": t["obs"]["pH"],
            "DO (%)": t["obs"]["dissolved_o2_pct"],
            "Viability": t["obs"]["viability"],
            "Feed / Cool / Base": f"{t['action']['glucose_feed']:.1f} / {t['action']['cooling_power']:.2f} / {t['action']['base_addition']:.2f}",
            "pdc / adhB": f"{t['action']['pdc_expression']:.2f} / {t['action']['adh_expression']:.2f}",
            "Agent Reasoning": t["reasoning"],
        }
        for t in trajectory
    ]
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True)
