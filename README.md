# ISDS Claim-to-Recovery Decision System

## Overview
A Streamlit-based analytical tool for structured decision support in investor–state dispute settlement (ISDS). The app models the path from claim framing to realized recovery using a staged workflow and probabilistic simulation.

## What it does
- Guides users through a decision sequence from case definition to recommendation
- Translates dispute characteristics into a distribution of plausible recovery outcomes
- Surfaces enforcement, settlement, and institutional friction considerations
- Supports comparison of strategic pathways and export of rationale

## Core features
- Staged workflow: define case → inspect evidence → inspect assumptions → run scenario → compare pathways → recommendation → export rationale
- Pre-award framing and post-award realization views
- Monte Carlo simulation of recovery outcomes
- Pathway comparison for enforcement and settlement strategies
- Structured recommendation with confidence qualifiers
- Evidence tables and assumption transparency

## Methodology
The analytical engine models recovery as a sequence of probabilistic stages, including jurisdictional success, award-to-claim ratio, annulment survival, enforcement success, and settlement retention. These are combined using Monte Carlo simulation to produce a distribution of outcomes rather than a single estimate. Behavioral overlays (e.g., anchoring and settlement range) support interpretation and strategy.

## Data / inputs
- Manually coded ISDS case data
- Global arbitration base rates
- Country-level governance and legal proxies
- User inputs: respondent state, sector, treaty basis, investor nationality, claim size

## Stack
- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Plotly

## Notes / limitations
- Data coverage is limited and partially sparse
- Some variables rely on proxy measures rather than direct observation
- The model simplifies dependencies between factors
- Outputs are intended for calibration and decision support, not prediction or legal advice
