from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from data_module import CASES, COUNTRY_PROFILES, SECTOR_STATS, TREATY_BASIS_STATS, calculate_historical_rates
from memo_generator import MemoGenerator
from simulation_engine import BehavioralModule, DisputeProfile, EnforcementPathway, SimulationEngine

st.set_page_config(
    page_title="ISDS Claim-to-Recovery Decision System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --app-bg: #f6f8fb;
            --panel-bg: #ffffff;
            --sidebar-bg: #e9edf3;
            --border: #cfd7e3;
            --text-primary: #122033;
            --text-secondary: #2f3e55;
            --text-muted: #5a6b84;
            --accent: #3f6f9b;
            --positive: #2f8d68;
            --negative: #b34747;
            --warning: #b5822d;
            --chart-bg: #ffffff;
            --gridline: #d8e0eb;
            --legend-text: #334861;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #0f1724;
                --panel-bg: #151f2f;
                --sidebar-bg: #111a28;
                --border: #2b3a4f;
                --text-primary: #e7edf7;
                --text-secondary: #c3cfde;
                --text-muted: #93a4ba;
                --accent: #6ea0d1;
                --positive: #5fbf98;
                --negative: #e07e7e;
                --warning: #d8b46a;
                --chart-bg: #151f2f;
                --gridline: #2a3950;
                --legend-text: #c3cfde;
            }
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp { background-color: var(--app-bg); color: var(--text-primary); }
        [data-testid="stAppViewContainer"] { background-color: var(--app-bg); color: var(--text-primary); }

        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * { color: var(--text-secondary); }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background-color: color-mix(in srgb, var(--panel-bg) 82%, var(--sidebar-bg) 18%) !important;
            border-color: var(--border) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] input[type="text"],
        [data-testid="stSidebar"] input[type="number"],
        [data-testid="stSidebar"] textarea {
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] input::placeholder,
        [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder {
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small {
            color: var(--text-muted) !important;
        }

        .dds-insignia {
            font-size: 0.76rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
        }

        .dds-meta {
            font-size: 0.73rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
            margin-bottom: 0.7rem;
        }

        .subtitle {
            color: var(--text-secondary);
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }

        .section-header {
            font-size: 1.03rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 1rem;
            margin-bottom: 0.45rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.35rem;
        }

        .analyst-note {
            border-left: 3px solid var(--accent);
            background: color-mix(in srgb, var(--panel-bg) 90%, var(--accent) 10%);
            padding: 0.75rem 0.9rem;
            border-radius: 0.35rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            min-height: 110px;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
            font-size: 0.76rem !important;
        }

        div[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            white-space: normal;
            border: 1px solid var(--border);
            border-radius: 6px 6px 0 0;
            color: var(--text-secondary);
            flex: 1 1 0;
            justify-content: center;
            padding: 0.6rem 0.75rem 0.56rem;
            font-size: 0.88rem;
            line-height: 1.3;
            text-align: center;
        }
        .stTabs [aria-selected="true"] {
            background: color-mix(in srgb, var(--accent) 26%, var(--panel-bg) 74%);
            color: var(--text-primary) !important;
        }
        @media (max-width: 1180px) {
            .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
            .stTabs [data-baseweb="tab"] {
                flex: 1 1 auto;
                font-size: 0.82rem;
                padding: 0.52rem 0.55rem;
            }
        }

        [data-testid="stHeader"] {
            background-color: color-mix(in srgb, var(--app-bg) 86%, var(--panel-bg) 14%) !important;
            border-bottom: 1px solid var(--border);
        }
        [data-testid="stHeader"] *,
        [data-testid="stToolbar"] *,
        [data-testid="stDecoration"] * {
            color: var(--text-secondary) !important;
        }
        [data-testid="stHeader"] button:hover,
        [data-testid="stToolbar"] button:hover {
            color: var(--text-primary) !important;
        }
        [data-testid="stStatusWidget"] * {
            color: var(--text-secondary) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_usd(amount: Optional[float], default: str = "N/A") -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return default
    if abs(amount) >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    return f"${amount:,.0f}"


def fmt_pct(v: Optional[float], default: str = "N/A") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return f"{v:.1%}"


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def analyst_note(text: str) -> None:
    st.markdown(f'<div class="analyst-note">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_cases_df() -> pd.DataFrame:
    rows = []
    for c in CASES:
        rows.append(
            {
                "Case Name": c["case_name"],
                "Country": c["respondent_state"],
                "Investor Nationality": c["investor_nationality"],
                "Sector": c["sector"],
                "Treaty Basis": c["treaty_basis"],
                "Year Filed": c["year_filed"],
                "Year Decided": c["year_decided"],
                "Outcome": c["outcome"],
                "Amount Claimed (USD)": c["amount_claimed_usd"],
                "Amount Awarded (USD)": c["amount_awarded_usd"],
                "Annulment Attempted": c["annulment_attempted"],
                "Annulment Outcome": c["annulment_outcome"] or "N/A",
                "Enforcement Status": c["enforcement_status"],
                "Notes": c["notes"],
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    resolved = df[~df["Outcome"].isin(["Pending", "Discontinued"])]
    n_resolved = len(resolved)
    win_rate = float((resolved["Outcome"] == "Investor Win").mean()) if n_resolved > 0 else np.nan

    award_df = df.dropna(subset=["Amount Claimed (USD)", "Amount Awarded (USD)"]).copy()
    award_df = award_df[(award_df["Amount Claimed (USD)"] > 0) & (award_df["Amount Awarded (USD)"] >= 0)]
    avg_ratio = float((award_df["Amount Awarded (USD)"] / award_df["Amount Claimed (USD)"]).mean()) if len(award_df) else np.nan

    return {
        "cases": int(len(df)),
        "investor_win_rate": win_rate,
        "avg_award_claim_ratio": avg_ratio,
        "total_claimed_usd": float(df["Amount Claimed (USD)"].fillna(0).sum()),
        "total_awarded_usd": float(df["Amount Awarded (USD)"].fillna(0).sum()),
    }


def apply_plotly_template() -> None:
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, sans-serif", color="#2f3e55"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=["#3f6f9b", "#2f8d68", "#b34747", "#b5822d", "#6d88ad"],
            margin=dict(l=80, r=35, t=55, b=60),
            legend=dict(font=dict(size=12), orientation="h", y=-0.2),
            xaxis=dict(gridcolor="#d8e0eb", zeroline=False),
            yaxis=dict(gridcolor="#d8e0eb", zeroline=False),
        )
    )
    pio.templates["dds"] = template
    pio.templates.default = "dds"


def build_case_chart(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("Country", as_index=False).size().sort_values("size")
    fig = px.bar(agg, x="size", y="Country", orientation="h", labels={"size": "Cases", "Country": ""})
    fig.update_traces(marker_color="#3f6f9b", text=agg["size"], textposition="outside", cliponaxis=False)
    fig.update_layout(title="Evidence Base by Respondent State", height=400, margin=dict(l=160, r=30, t=60, b=45))
    return fig


def build_outcome_chart(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("Outcome", as_index=False).size().sort_values("size", ascending=False)
    fig = px.bar(agg, x="Outcome", y="size", color="Outcome")
    fig.update_layout(title="Outcome Composition", showlegend=False, height=340, margin=dict(l=50, r=20, t=60, b=80))
    fig.update_xaxes(tickangle=-20)
    return fig


def build_recovery_chart(dist: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dist * 100, nbinsx=50, marker_color="#3f6f9b"))
    mean_v = float(dist.mean() * 100)
    fig.add_vline(x=mean_v, line_dash="dash", annotation_text=f"Mean {mean_v:.1f}%")
    fig.update_layout(title="Recovery Distribution", xaxis_title="Recovery Rate (% of claim)", yaxis_title="Draw count", height=340)
    return fig


_inject_theme()
apply_plotly_template()

st.title("ISDS Claim-to-Recovery Decision System")
st.markdown(
    '<div class="subtitle">Structured decision support for pre-award and post-award pathway selection under enforcement uncertainty.</div>',
    unsafe_allow_html=True,
)

cases_df = load_cases_df()
years = cases_df["Year Filed"].dropna().astype(int)
data_window = f"{years.min()}–{years.max()}" if not years.empty else "N/A"

with st.sidebar:
    st.markdown('<div class="dds-insignia">Designing Decision Systems</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dds-meta">Data window: {data_window}</div>', unsafe_allow_html=True)
    stage = st.radio("Analytical Stage", ["Pre-award framing", "Post-award realization"], horizontal=False)
    st.divider()
    st.markdown("**Case framing inputs**")
    sel_state = st.selectbox("Respondent state", sorted(COUNTRY_PROFILES.keys()), index=0)
    sel_sector = st.selectbox("Sector", sorted(SECTOR_STATS.keys()), index=0)
    sel_treaty = st.selectbox("Treaty basis", sorted(TREATY_BASIS_STATS.keys()), index=0)
    sel_investor = st.selectbox("Investor nationality", sorted(cases_df["Investor Nationality"].dropna().unique().tolist()))
    sel_claim = st.number_input("Claim amount (USD)", min_value=1_000_000, max_value=30_000_000_000, value=500_000_000, step=10_000_000, format="%d")
    invest_type = st.text_input("Investment type", value="greenfield investment")
    n_sims = st.slider("Simulation draws", 1000, 50000, 10000, 1000)
    use_seed = st.checkbox("Fixed seed", True)
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42) if use_seed else None

    st.divider()

# workflow tabs
steps = st.tabs(
    [
        "Step 1 · Define Case",
        "Step 2 · Inspect Evidence",
        "Step 3 · Inspect Assumptions",
        "Step 4 · Run Scenario",
        "Step 5 · Compare Pathways",
        "Step 6 · Recommendation",
        "Step 7 · Export Rationale",
    ]
)

case_profile = DisputeProfile(
    respondent_state=sel_state,
    investor_nationality=sel_investor,
    sector=sel_sector,
    treaty_basis=sel_treaty,
    amount_claimed_usd=float(sel_claim),
    investment_type=invest_type,
)

with steps[0]:
    section_header("Analytical Summary")
    c1, c2, c3, c4 = st.columns(4)
    country_stats = calculate_historical_rates(sel_state)
    c1.metric("Claimed amount", fmt_usd(float(sel_claim)))
    c2.metric("Historical investor win rate", fmt_pct(country_stats.get("investor_win_rate")))
    c3.metric("Historical settlement rate", fmt_pct(country_stats.get("settlement_rate")))
    c4.metric("Historical resolved cases", f"{country_stats.get('resolved_cases', 0):,}")

    analyst_note(
        f"This case is framed as a {sel_treaty.lower()} dispute in the {sel_sector.lower()} sector against {sel_state}. "
        f"The current stage is **{stage}**, which controls emphasis in the downstream pathway comparison and recommendation."
    )

with steps[1]:
    section_header("Key Evidence")
    f1, f2, f3 = st.columns(3)
    with f1:
        countries = st.multiselect("Filter countries", sorted(cases_df["Country"].unique().tolist()))
    with f2:
        sectors = st.multiselect("Filter sectors", sorted(cases_df["Sector"].unique().tolist()))
    with f3:
        outcomes = st.multiselect("Filter outcomes", sorted(cases_df["Outcome"].unique().tolist()))

    filt = cases_df.copy()
    if countries:
        filt = filt[filt["Country"].isin(countries)]
    if sectors:
        filt = filt[filt["Sector"].isin(sectors)]
    if outcomes:
        filt = filt[filt["Outcome"].isin(outcomes)]

    s = summarize(filt)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered cases", f"{s['cases']:,}")
    m2.metric("Investor win rate", fmt_pct(s["investor_win_rate"]))
    m3.metric("Avg award/claim ratio", fmt_pct(s["avg_award_claim_ratio"]))
    m4.metric("Disclosed claimed amount", fmt_usd(s["total_claimed_usd"]))

    tdf = filt[[
        "Case Name", "Country", "Sector", "Treaty Basis", "Year Filed", "Outcome", "Amount Claimed (USD)", "Amount Awarded (USD)", "Enforcement Status"
    ]].copy()
    st.dataframe(
        tdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount Claimed (USD)": st.column_config.NumberColumn(format="$%,.0f"),
            "Amount Awarded (USD)": st.column_config.NumberColumn(format="$%,.0f"),
            "Year Filed": st.column_config.NumberColumn(format="%d"),
        },
        height=360,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_case_chart(filt), use_container_width=True)
    with c2:
        st.plotly_chart(build_outcome_chart(filt), use_container_width=True)

with steps[2]:
    section_header("Model Assumptions")
    analyst_note(
        "Assumptions are shown before simulation to support interpretability: calibrated jurisdiction rate, award-to-claim distribution, "
        "annulment process, enforcement friction, and settlement discount logic from the embedded engine."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input assumptions**")
        st.write(f"- Respondent state: {sel_state}")
        st.write(f"- Sector: {sel_sector}")
        st.write(f"- Treaty basis: {sel_treaty}")
        st.write(f"- Investor nationality: {sel_investor}")
        st.write(f"- Amount claimed: {fmt_usd(float(sel_claim))}")
        st.write(f"- Simulation draws: {n_sims:,}")
    with col2:
        cp = COUNTRY_PROFILES.get(sel_state, {})
        st.markdown("**Country posture assumptions**")
        st.write(f"- Enforcement friction level: {cp.get('enforcement_friction_level', 'N/A')}")
        st.write(f"- Voluntary compliance history: {cp.get('voluntary_compliance_history', 'N/A')}")
        disc = cp.get("settlement_discount_range", (0.1, 0.3))
        st.write(f"- Settlement discount range: {disc[0]:.0%}–{disc[1]:.0%}")
        st.write(f"- WGI rule of law percentile: {cp.get('wgi_rule_of_law', 'N/A')}")

with steps[3]:
    section_header("Scenario Execution")
    run = st.button("Run scenario", type="primary", use_container_width=False)
    if run:
        engine = SimulationEngine(case_profile, n_simulations=n_sims, seed=int(seed) if seed is not None else None)
        with st.spinner("Running Monte Carlo scenario..."):
            full = engine.run_full_simulation()
            rec = engine.simulate_recovery_rate()
            awd = engine.simulate_award_to_claim_ratio()
            tml = engine.simulate_enforcement_timeline()

        st.session_state["scenario"] = {
            "full": full,
            "recovery_dist": rec["distribution"],
            "award_dist": awd["distribution"],
            "timeline_dist": tml["timeline_months_distribution"],
            "profile": case_profile,
            "country_profile": COUNTRY_PROFILES.get(sel_state, {}),
        }
        st.success("Scenario completed.")

    if "scenario" in st.session_state:
        full = st.session_state["scenario"]["full"]
        sm = full["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jurisdictional success", fmt_pct(sm["jurisdictional_success_prob"]))
        c2.metric("Expected recovery", fmt_usd(sm["expected_recovery_usd"]))
        c3.metric("Expected recovery rate", fmt_pct(sm["expected_recovery_fraction"]))
        c4.metric("Expected time to recovery", f"{sm['expected_years_to_recovery']:.1f} years")

        st.plotly_chart(build_recovery_chart(st.session_state["scenario"]["recovery_dist"]), use_container_width=True)

with steps[4]:
    section_header("Scenario Comparison")
    if "scenario" not in st.session_state:
        st.info("Run Step 4 first to compare pathways.")
    else:
        sc = st.session_state["scenario"]
        sm = sc["full"]["summary"]
        ep = EnforcementPathway(sc["country_profile"], float(sm["expected_recovery_usd"]), sel_state)
        jurisdictions = ep.map_jurisdictions()

        rows = []
        for i, j in enumerate(jurisdictions, start=1):
            rows.append(
                {
                    "Priority": i,
                    "Jurisdiction": j["jurisdiction"],
                    "Success Probability": j["success_probability"],
                    "Timeline (months)": int(round(j["timeline_months"])),
                    "Estimated Cost (USD)": j["costs_estimate_usd"],
                }
            )
        jdf = pd.DataFrame(rows)
        st.dataframe(
            jdf,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Success Probability": st.column_config.ProgressColumn(format="%.0%", min_value=0, max_value=1),
                "Estimated Cost (USD)": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

        fig = px.bar(jdf.sort_values("Success Probability"), x="Success Probability", y="Jurisdiction", orientation="h")
        fig.update_traces(texttemplate="%{x:.0%}", textposition="outside", cliponaxis=False)
        fig.update_layout(title="Post-award enforcement pathway comparison", height=360, margin=dict(l=180, r=25, t=60, b=45))
        st.plotly_chart(fig, use_container_width=True)

        if stage == "Pre-award framing":
            analyst_note("Pre-award emphasis: treat pathway scores as bargaining leverage and scenario conditioning, not immediate execution instructions.")
        else:
            analyst_note("Post-award emphasis: pathway ranking indicates near-term sequencing for recognition, discovery, and attachment actions.")

with steps[5]:
    section_header("Decision Rationale")
    if "scenario" not in st.session_state:
        st.info("Run Step 4 before reviewing recommendations.")
    else:
        sc = st.session_state["scenario"]
        full = sc["full"]
        sm = full["summary"]

        confidence_band = sm["expected_recovery_fraction"]
        if confidence_band >= 0.45:
            confidence = "High"
        elif confidence_band >= 0.25:
            confidence = "Moderate"
        else:
            confidence = "Cautious"

        if stage == "Pre-award framing":
            rec_line = "Proceed with arbitration preparation while targeting structured settlement windows tied to enforceable jurisdictions."
        else:
            rec_line = "Proceed with phased enforcement in top-ranked jurisdictions while preserving settlement optionality."

        st.metric("Recommendation confidence qualifier", confidence)
        analyst_note(
            f"**Recommendation:** {rec_line} "
            f"Expected recovery is {fmt_pct(sm['expected_recovery_fraction'])} ({fmt_usd(sm['expected_recovery_usd'])}) "
            f"with an expected realization horizon of {sm['expected_years_to_recovery']:.1f} years."
        )

        st.markdown("**Key drivers**")
        st.write(f"- Jurisdictional success probability: {fmt_pct(sm['jurisdictional_success_prob'])}")
        st.write(f"- Enforcement probability: {fmt_pct(sm['enforcement_prob'])}")
        st.write(f"- Annulment net risk: {fmt_pct(sm['annulment_net_risk'])}")
        st.write(f"- Sovereign friction score: {sm['sovereign_friction_score']:.0f}/100")

with steps[6]:
    section_header("Export Rationale")
    if "scenario" not in st.session_state:
        st.info("Run Step 4 first to enable rationale export.")
    else:
        sc = st.session_state["scenario"]
        ep = EnforcementPathway(sc["country_profile"], float(sc["full"]["summary"]["expected_recovery_usd"]), sel_state)
        mg = MemoGenerator(sc["full"], sc["profile"], sc["country_profile"], ep)
        memo = mg.generate_full_memo()
        csv_data = mg.generate_csv_export()

        export_stamp = datetime.utcnow().strftime("%Y%m%d")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download decision rationale (.txt)",
                data=memo.encode("utf-8"),
                file_name=f"ISDS_Decision_Rationale_{sel_state}_{export_stamp}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download model output table (.csv)",
                data=csv_data.encode("utf-8"),
                file_name=f"ISDS_Model_Output_{sel_state}_{export_stamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with st.expander("Preview rationale", expanded=False):
            st.code(memo[:7000] + "\n\n[Preview truncated]", language=None)
