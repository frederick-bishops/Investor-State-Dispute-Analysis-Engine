# Investor-State Dispute Analysis Engine

**Post-Award Enforcement and Settlement Simulator for African States**

A production-grade Streamlit application that models the full post-award path — from tribunal decision to actual recovery — for investor-state disputes involving Sub-Saharan African states.

## Features

### 1. Case Database Explorer
Interactive filterable database of 37+ real ICSID/UNCITRAL cases involving African respondent states across 13 countries. Includes outcome statistics, award-to-claim ratios, and timeline visualizations.

### 2. Dispute Simulator (Core Feature)
Monte Carlo simulation engine for user-defined dispute profiles:
- **Jurisdictional success probability** — Beta-adjusted Bernoulli trials calibrated to 57% historical upholding rate
- **Award-to-claim ratio** — Bracket-weighted Monte Carlo with sector-specific scaling
- **Annulment risk** — Log-normal delay distributions with country-adjusted rates
- **Enforcement timeline** — Composite model factoring multi-jurisdiction proceedings
- **Recovery rate** — 5-factor multiplicative model producing probabilistic forecasts

### 3. Enforcement Pathway Mapper
Jurisdiction-specific attachment strategies across 6 enforcement venues (New York, London, Paris, The Hague, Stockholm, Australia). Includes sovereign immunity scoring, asset attachability analysis, and recommended sequencing.

### 4. Decision Engine
Probabilistic decision tree visualization with optimal sequencing (annulment first vs. parallel attachments), settlement threshold calculator, and downloadable advisory memos.

### 5. Country Risk Profiles
Governance metrics (WGI Rule of Law, Corruption, Government Effectiveness) for 22 African states with enforcement friction classifications and comparative analysis.

### 6. Behavioral Analysis
- **Overclaiming bias** — Anchoring effect quantification
- **State delay incentive** — NPV decay under enforcement uncertainty
- **Settlement zone (ZOPA)** — Rational concession analysis under loss aversion
- **Prospect theory** — Kahneman-Tversky valuation asymmetries

## Data Sources (All Public)
- [ICSID Caseload Statistics 2025](https://icsid.worldbank.org)
- [UNCTAD Investment Policy Hub](https://investmentpolicy.unctad.org)
- [italaw.com](https://www.italaw.com)
- [World Bank WGI DataBank](https://databank.worldbank.org)
- [World Justice Project Rule of Law Index](https://worldjusticeproject.org)
- Public Citizen, Transnational Institute, BIICL reports

## Installation & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment to Streamlit Community Cloud

1. Push this directory to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set `app.py` as the main file
5. Deploy — Streamlit will install from `requirements.txt` automatically

## Project Structure

```
isds_app/
├── app.py                  # Main Streamlit application (1,790 lines)
├── data_module.py          # Structured case database & reference data (1,616 lines)
├── simulation_engine.py    # Monte Carlo & Bayesian simulation engine (1,792 lines)
├── memo_generator.py       # Advisory brief generator (603 lines)
├── requirements.txt        # Python dependencies
├── __init__.py             # Package marker
└── README.md               # This file
```

## Architecture

- **data_module.py**: 37 coded ISDS cases, 22 country profiles with WGI scores, 6 enforcement jurisdictions, BIT network, sector/treaty statistics
- **simulation_engine.py**: `SimulationEngine` (Monte Carlo), `BehavioralModule` (anchoring, prospect theory, ZOPA), `EnforcementPathway` (jurisdiction mapping, asset scoring, sequencing)
- **memo_generator.py**: `MemoGenerator` producing structured 10-section advisory memoranda
- **app.py**: 6-tab Streamlit dashboard with Plotly visualizations, custom CSS, and export capabilities

## Technical Notes

- All simulations use `numpy.random` with configurable seeds for reproducibility
- `scipy.stats` powers the Beta, log-normal, and custom distributions
- Plotly renders all interactive charts (optimized for Streamlit)
- `st.cache_data` / `st.cache_resource` for performance optimization
- Custom CSS provides a professional teal/navy color scheme

## Calibration

Key parameters are calibrated to published empirical data:
- 57% historical award upholding rate (ICSID 2025)
- 44.7% annulment application rate, ~5% success rate (BIICL 2021)
- Sector-specific award-to-claim brackets (ICSID Caseload Statistics)
- Country-specific enforcement friction from WGI/WJP scores
- Settlement discount ranges from Kluwer Arbitration Blog / Freshfields analysis
