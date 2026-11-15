import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types
from ecoli_bioreactor_v2 import CoupledEColiBioreactorEnv, solve_reference_policy
from run_gemini_agent import SYSTEM_PROMPT, BioreactorAction, classify_failure

st.set_page_config(page_title="E. coli Bioethanol Process Optimizer", layout="wide")
st.title("🧬 Fed-Batch E. coli Bioethanol Bioreactor: Growth & Titer Optimizer")
st.caption(
    "Interactive bioprocess simulator for optimizing cell biomass growth and bioethanol titer "
    "under coupled thermal, pH, weak-acid, and oxygen mass-transfer (kLa) dynamics."
)

# --- Sidebar: Process & Optimization Controls ---
with st.sidebar:
    st.header("1. Reactor Configuration")
    difficulty = st.slider("Bioreactor Complexity Tier", min_value=1, max_value=5, value=3)
    opt_mode = st.radio(
        "Control & Optimization Mode",
        [
            "Interactive Biphasic Optimizer (Manual Setpoints)",
            "SciPy MPC Auto-Optimizer (Reference Baseline)",
            "Gemini AI Agent vs. SciPy Optimizer",
        ],
    )

    env_preview = CoupledEColiBioreactorEnv(difficulty=difficulty)
    max_h = env_preview.cfg.max_steps
    max_g = env_preview.cfg.max_glucose_feed

    if opt_mode == "Interactive Biphasic Optimizer (Manual Setpoints)":
        st.header("2. Biphasic Process Setpoints")
        switch_hr = st.slider("Induction Switch Time (Hour)", 1, max_h - 1, int(max_h * 0.45))

        with st.expander("Phase I: Aerobic Cell Growth", expanded=True):
            g_feed_1 = st.slider("Growth Glucose Feed (g/L/hr)", 0.0, float(max_g), round(max_g * 0.45, 1), 0.5)
            o2_1 = st.slider("Growth O2 Sparge", 0.0, 1.0, 0.90, 0.05)
            cool_1 = st.slider("Growth Jacket Cooling", 0.0, 1.0, 0.35, 0.05)
            base_1 = st.slider("Growth Base Titration", 0.0, 1.0, 0.05, 0.02)

        with st.expander("Phase II: Bioethanol Production", expanded=True):
            g_feed_2 = st.slider("Prod. Glucose Feed (g/L/hr)", 0.0, float(max_g), round(max_g * 0.65, 1), 0.5)
            o2_2 = st.slider("Prod. O2 Sparge", 0.0, 1.0, 0.75, 0.05)
            pdc_2 = st.slider("pdc Expression (Pyruvate -> AcAld)", 0.0, 2.0, 1.25, 0.05)
            adh_2 = st.slider("adhB Expression (AcAld -> EtOH)", 0.0, 2.0, 1.50, 0.05)
            cool_2 = st.slider("Prod. Jacket Cooling", 0.0, 1.0, 0.60, 0.05)
            base_2 = st.slider("Prod. Base Titration", 0.0, 1.0, 0.25, 0.02)

    elif opt_mode == "Gemini AI Agent vs. SciPy Optimizer":
        st.header("2. Gemini Agent Settings")
        model_id = st.selectbox("Gemini Model", ["gemini-3.8-flash", "gemini-4-argon", "gemini-3.5-flash-lite"])
        try:
            default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        except Exception:
            default_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input("Gemini API Key", value=default_key, type="password")

    run_btn = st.button("▶ Run Bioreactor Simulation", type="primary", use_container_width=True)


def run_manual_simulation() -> dict:
    env = CoupledEColiBioreactorEnv(difficulty=difficulty)
    obs = env.reset()
    records = [{**obs["state"], "Time (hr)": 0, "Step Reward": 0.0, "Action Phase": "Initial"}]
    total_reward = 0.0

    for step in range(env.cfg.max_steps):
        if step < switch_hr:
            phase = "Phase I: Growth"
            act = {
                "glucose_feed": g_feed_1,
                "o2_sparge": o2_1,
                "pdc_expression": 0.05,
                "adh_expression": 0.10,
                "cooling_power": cool_1,
                "base_addition": base_1,
                "harvest": False,
            }
        else:
            phase = "Phase II: Production"
            act = {
                "glucose_feed": g_feed_2,
                "o2_sparge": o2_2,
                "pdc_expression": pdc_2,
                "adh_expression": adh_2,
                "cooling_power": cool_2,
                "base_addition": base_2,
                "harvest": (step == env.cfg.max_steps - 1),
            }

        obs, r, done, _ = env.step(act)
        total_reward += r
        records.append({
            **obs["state"],
            "Time (hr)": obs["step"],
            "Step Reward": round(r, 2),
            "Action Phase": phase,
        })
        if done:
            break

    return {
        "df": pd.DataFrame(records),
        "reward": total_reward,
        "crash_reason": obs["crash_reason"],
        "final_state": obs["state"],
        "steps": obs["step"],
        "max_steps": env.cfg.max_steps,
    }


def run_scipy_simulation() -> dict:
    ref_out = solve_reference_policy(difficulty=difficulty)
    rows = []
    for item in ref_out["trajectory"]:
        rows.append({
            **item["obs"]["state"],
            "Time (hr)": item["obs"]["step"],
            "Step Reward": round(item["reward"], 2),
        })
    # Append final terminal state
    last_step = len(rows)
    rows.append({
        **ref_out["final_state"],
        "Time (hr)": last_step,
        "Step Reward": round(ref_out["reference_reward"], 2),
    })
    return {
        "df": pd.DataFrame(rows),
        "reward": ref_out["reference_reward"],
        "crash_reason": ref_out["crash_reason"],
        "final_state": ref_out["final_state"],
        "steps": last_step,
        "max_steps": max_h,
    }


# --- Run Simulation (Auto-runs on first load so dashboard is immediately populated) ---
if run_btn or "last_run" not in st.session_state:
    if opt_mode == "Interactive Biphasic Optimizer (Manual Setpoints)":
        st.session_state["last_run"] = {"primary": run_manual_simulation(), "label": "Manual Biphasic Run"}
    elif opt_mode == "SciPy MPC Auto-Optimizer (Reference Baseline)":
        with st.spinner("Optimizing biphasic growth & induction setpoints via SciPy..."):
            st.session_state["last_run"] = {"primary": run_scipy_simulation(), "label": "SciPy MPC Optimizer"}
    else:
        if not api_key:
            st.error("Enter a Gemini API Key in the sidebar to run the Gemini AI Agent.")
            st.stop()
        with st.spinner("Running SciPy Reference + Gemini Agent..."):
            ref_sim = run_scipy_simulation()
            env = CoupledEColiBioreactorEnv(difficulty=difficulty)
            obs = env.reset()
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=BioreactorAction,
            )
            chat = client.chats.create(model=model_id, config=config)
            rows = [{**obs["state"], "Time (hr)": 0, "Reasoning": "Initial state"}]
            traj = []
            tot_r = 0.0
            while not obs["terminated"]:
                resp = chat.send_message(f"Current Bioreactor Observation:\n{json.dumps(obs)}\n\nNext control action:")
                act = resp.parsed.model_dump()
                next_obs, r, done, _ = env.step(act)
                traj.append({"obs": obs["state"], "action": act, "reward": r})
                obs = next_obs
                tot_r += r
                rows.append({**obs["state"], "Time (hr)": obs["step"], "Reasoning": act["reasoning"]})
            st.session_state["last_run"] = {
                "primary": {
                    "df": pd.DataFrame(rows),
                    "reward": tot_r,
                    "crash_reason": obs["crash_reason"],
                    "final_state": obs["state"],
                    "steps": obs["step"],
                    "max_steps": env.cfg.max_steps,
                    "outcome": classify_failure(traj, obs["crash_reason"]),
                },
                "reference": ref_sim,
                "label": f"Gemini Agent ({model_id})",
            }

res = st.session_state["last_run"]
sim = res["primary"]
df = sim["df"]
final_st = sim["final_state"]

# --- Top KPI Banner ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Final Bioethanol Titer", f"{final_st['bioethanol_g_L']:.2f} g/L")
k2.metric("Peak Cell Biomass", f"{df['biomass_g_L'].max():.2f} g/L")
k3.metric("Final Culture Viability", f"{final_st['viability']*100:.1f}%")
k4.metric("Net Batch Profit", f"${sim['reward']:.2f}")
k5.metric("Batch Duration", f"{sim['steps']} / {sim['max_steps']} hrs")

if sim["crash_reason"]:
    st.error(f"Batch Terminated Early — {sim['crash_reason']}")
else:
    st.success(f"Batch Completed Successfully ({res['label']})")

# --- 1. Logarithmic Growth & Product Spectrum Chart ---
st.subheader("📈 Logarithmic Growth & Metabolite Spectrum (Semi-Log Scale)")
log_cols = st.multiselect(
    "Select Species to Plot on Logarithmic Axis (g/L):",
    options=[
        "biomass_g_L",
        "bioethanol_g_L",
        "glucose_g_L",
        "pyruvate_g_L",
        "acetaldehyde_g_L",
        "acetate_g_L",
    ],
    default=["biomass_g_L", "bioethanol_g_L", "glucose_g_L", "acetate_g_L", "acetaldehyde_g_L"],
)

fig_log = go.Figure()
palette = {
    "biomass_g_L": "#2ca02c",
    "bioethanol_g_L": "#1f77b4",
    "glucose_g_L": "#ff7f0e",
    "pyruvate_g_L": "#9467bd",
    "acetaldehyde_g_L": "#d62728",
    "acetate_g_L": "#8c564b",
}

for col in log_cols:
    # Clamp near-zero values to 1e-3 g/L so log10 axis renders cleanly without dropping points
    y_safe = np.maximum(df[col].astype(float), 1e-3)
    fig_log.add_trace(
        go.Scatter(
            x=df["Time (hr)"],
            y=y_safe,
            mode="lines+markers",
            name=col,
            line=dict(width=2.5, color=palette.get(col)),
        )
    )

fig_log.update_layout(
    yaxis_type="log",
    yaxis_title="Concentration (g/L — Log Scale)",
    xaxis_title="Batch Time (Hours)",
    height=430,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_log, use_container_width=True)

# --- 2. Physico-Chemical Environment Charts (pH, Temperature, Dissolved O2) ---
st.subheader("🌡️ Bioreactor Environment: Temperature, pH & Dissolved Oxygen")
c1, c2, c3 = st.columns(3)

with c1:
    fig_temp = go.Figure(go.Scatter(x=df["Time (hr)"], y=df["temperature_C"], mode="lines+markers", line=dict(color="#e377c2")))
    fig_temp.add_hline(y=41.5, line_dash="dash", line_color="red", annotation_text="Lethal Heat (41.5°C)")
    fig_temp.update_layout(title="Temperature (°C)", xaxis_title="Hour", yaxis_title="°C", height=280)
    st.plotly_chart(fig_temp, use_container_width=True)

with c2:
    fig_ph = go.Figure(go.Scatter(x=df["Time (hr)"], y=df["pH"], mode="lines+markers", line=dict(color="#17becf")))
    fig_ph.add_hrect(y0=6.8, y1=7.2, fillcolor="green", opacity=0.12, line_width=0, annotation_text="Optimal pH")
    fig_ph.update_layout(title="Broth pH", xaxis_title="Hour", yaxis_title="pH", height=280)
    st.plotly_chart(fig_ph, use_container_width=True)

with c3:
    fig_do = go.Figure(go.Scatter(x=df["Time (hr)"], y=df["dissolved_o2_pct"], mode="lines+markers", line=dict(color="#bcbd22")))
    fig_do.add_hline(y=20.0, line_dash="dash", line_color="orange", annotation_text="Hypoxia Threshold")
    fig_do.update_layout(title="Dissolved Oxygen (%)", xaxis_title="Hour", yaxis_title="DO %", height=280)
    st.plotly_chart(fig_do, use_container_width=True)

# --- 3. Structured Process Data Tables ---
st.subheader("📋 Bioprocess Time-Series Data Tables")
tab_all, tab_sub_met, tab_env = st.tabs([
    "Complete Batch Log",
    "Substrate & Metabolites",
    "pH, Temperature & Oxygen Levels",
])

with tab_all:
    st.dataframe(df, use_container_width=True)

with tab_sub_met:
    sub_met_cols = [
        "Time (hr)",
        "biomass_g_L",
        "glucose_g_L",
        "bioethanol_g_L",
        "pyruvate_g_L",
        "acetaldehyde_g_L",
        "acetate_g_L",
        "viability",
    ]
    st.dataframe(df[sub_met_cols], use_container_width=True)

with tab_env:
    env_cols = [
        "Time (hr)",
        "temperature_C",
        "pH",
        "dissolved_o2_pct",
        "acetate_g_L",
        "viability",
    ]
    st.dataframe(df[env_cols], use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Full Simulation CSV", data=csv_bytes, file_name="ecoli_bioreactor_simulation.csv", mime="text/csv")
