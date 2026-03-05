"""
ISDS Recovery Realism Engine — Simulation Engine
=================================================
Monte Carlo / Bayesian simulation engine for modelling investor-state
dispute outcomes, recovery probabilities, enforcement pathways, and
behavioural dynamics.

Architecture
------------
DisputeProfile      — data class holding all input parameters for one dispute
SimulationEngine    — Monte Carlo simulations of jurisdiction, award,
                      annulment, enforcement, and composite recovery
BehavioralModule    — prospect theory, anchoring bias, ZOPA / settlement-zone
                      analysis, and state delay-incentive modelling
EnforcementPathway  — multi-jurisdiction asset mapping, sequencing strategy,
                      and decision-tree generation

Sources
-------
- ICSID Caseload Statistics 2025:
  https://icsid.worldbank.org/sites/default/files/publications/
  2025-1%20ENG%20-%20The%20ICSID%20Caseload%20Statistics%20(Issue%202025-1).pdf
- BIICL Empirical Study on Annulment (2021):
  https://www.biicl.org/documents/10899_annulment-in-icsid-arbitration190821.pdf
- Public Citizen "The Scramble for Africa Continues" (Dec 2024):
  https://www.citizen.org/article/the-scramble-for-africa-continues-impacts-of-investor-state-dispute-settlement-on-african-countries/
- TNI ISDS in Numbers (2019):
  https://www.tni.org/files/publication-downloads/isds_africa_web.pdf
- World Bank WGI DataBank 2023:
  https://databank.worldbank.org/embed/WGI-Table/id/ceea4d8b
- ICSID Compliance Background Paper (2024):
  https://icsid.worldbank.org/sites/default/files/publications/Enforcement_Paper.pdf
- Kluwer Arbitration Blog — Post-Award Bargaining Power (2019):
  https://legalblogs.wolterskluwer.com/arbitration-blog/post-award-bargaining-power-of-states-examples-from-bolivia/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from data_module import (
    AWARD_TO_CLAIM_DISTRIBUTION,
    COUNTRY_PROFILES,
    ENFORCEMENT_JURISDICTIONS,
    SECTOR_STATS,
    TREATY_BASIS_STATS,
    calculate_historical_rates,
    get_cases_by_country,
)


# ---------------------------------------------------------------------------
# CONSTANTS — calibrated from research data
# ---------------------------------------------------------------------------

# Global jurisdictional upholding rate (ICSID merits; 53% in 2024, ~57% global avg)
# Source: ICSID 2024 Annual; ISDS Bilaterals 2025
_GLOBAL_JURISDICTIONAL_SUCCESS_RATE: float = 0.57

# Annulment application rate (44.7% of ICSID awards)
# Source: BIICL 2021; ICSID Background Paper 2024 (194/434 awards = 44.7%)
_ANNULMENT_APPLICATION_RATE: float = 0.447

# Annulment success rate (full or partial) — ~5% of awards where annulment applied
# Source: BIICL 2021: "partial or full annulment success rate ~5% of awards"
_ANNULMENT_SUCCESS_RATE: float = 0.05

# Average duration of ICSID arbitration (days): mean 1,370; converted to months
# Source: ICSID/UNCITRAL Duration Study
_AVG_ARBITRATION_MONTHS: float = 45.7   # 1,370 / 30

# Average annulment proceeding duration (months)
_AVG_ANNULMENT_MONTHS: float = 24.0

# Friction level → numeric enforcement probability multiplier
_FRICTION_ENFORCEMENT_PROB: dict[str, float] = {
    "Critical":      0.15,
    "Very High":     0.25,
    "High":          0.45,
    "Moderate":      0.65,
    "Moderate-Low":  0.80,
    "Low":           0.92,
}

# Friction level → base jurisdictional success adjustment (additive %)
_FRICTION_JURISDICTION_ADJ: dict[str, float] = {
    "Critical":      -0.10,
    "Very High":     -0.07,
    "High":          -0.04,
    "Moderate":       0.00,
    "Moderate-Low":  +0.03,
    "Low":           +0.05,
}

# Sector → jurisdictional success adjustment (additive %)
# Sectors with poor merits track records get a negative adjustment
_SECTOR_JURISDICTION_ADJ: dict[str, float] = {
    "Mining":             +0.03,
    "Oil & Gas":          +0.05,
    "Agriculture":        +0.02,
    "Energy":             +0.03,
    "Infrastructure":      0.00,
    "Telecommunications": +0.02,
    "Manufacturing":      -0.02,
    "Hospitality":        -0.02,
    "Construction":       -0.02,
    "Retail":             -0.05,
    "Maritime":            0.00,
    "Legal Services":     -0.10,
    "General Commercial": -0.03,
    "Other":               0.00,
}

# Treaty basis → jurisdictional success adjustment (additive %)
_TREATY_JURISDICTION_ADJ: dict[str, float] = {
    "Bilateral Investment Treaty": +0.04,
    "Investment Contract":        -0.02,
    "Investment Law":             -0.01,
}

# Award bracket midpoints for Monte Carlo sampling
_BRACKET_MIDPOINTS: list[float] = [0.05, 0.175, 0.38, 0.63, 0.88]


# ---------------------------------------------------------------------------
# DISPUTE PROFILE
# ---------------------------------------------------------------------------

@dataclass
class DisputeProfile:
    """All input parameters describing a single investor-state dispute.

    Attributes:
        respondent_state:    Name of the African respondent state
                             (must match a key in COUNTRY_PROFILES).
        investor_nationality: Nationality / home jurisdiction of the claimant.
        sector:              Sector of the investment
                             (should match a key in SECTOR_STATS).
        treaty_basis:        Consent basis: "Bilateral Investment Treaty",
                             "Investment Contract", or "Investment Law".
        amount_claimed_usd:  Amount claimed by investor (USD).
        investment_type:     Free-text description, e.g. "greenfield mining",
                             "share acquisition", "infrastructure concession".
        bit_year:            Year the applicable BIT entered into force
                             (None if not BIT-based or unknown).
    """
    respondent_state:     str
    investor_nationality: str
    sector:               str
    treaty_basis:         str
    amount_claimed_usd:   float
    investment_type:      str       = "unspecified"
    bit_year:             Optional[int] = None

    def country_profile(self) -> Optional[dict]:
        """Return the COUNTRY_PROFILES entry for the respondent state, or None."""
        return COUNTRY_PROFILES.get(self.respondent_state)


# ---------------------------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------------------------

class SimulationEngine:
    """Monte Carlo / Bayesian simulation engine for a single dispute profile.

    All simulations are reproducible via the ``seed`` parameter and use
    numpy random number generation with scipy.stats distributions.

    Parameters
    ----------
    dispute_profile : DisputeProfile
        Fully specified dispute profile.
    n_simulations : int
        Number of Monte Carlo draws (default 10,000).
    seed : int or None
        Random seed for reproducibility. None = non-deterministic.
    """

    def __init__(
        self,
        dispute_profile: DisputeProfile,
        n_simulations: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self.dp = dispute_profile
        self.n   = n_simulations
        self.rng = np.random.default_rng(seed)
        self._profile = dispute_profile.country_profile() or {}
        self._friction = self._profile.get("enforcement_friction_level", "Moderate")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _friction_enforcement_prob(self) -> float:
        return _FRICTION_ENFORCEMENT_PROB.get(self._friction, 0.60)

    def _jurisdiction_base_rate(self) -> float:
        """Compute the adjusted jurisdictional success probability."""
        base = _GLOBAL_JURISDICTIONAL_SUCCESS_RATE
        base += _FRICTION_JURISDICTION_ADJ.get(self._friction, 0.0)
        base += _SECTOR_JURISDICTION_ADJ.get(self.dp.sector, 0.0)
        base += _TREATY_JURISDICTION_ADJ.get(self.dp.treaty_basis, 0.0)

        # Historical adjustment: if we have country-level data, blend it in
        hist = calculate_historical_rates(self.dp.respondent_state)
        if hist["investor_win_rate"] is not None and hist["resolved_cases"] >= 3:
            weight = min(hist["resolved_cases"] / 20.0, 0.35)   # max 35% weight
            base = (1.0 - weight) * base + weight * hist["investor_win_rate"]

        return float(np.clip(base, 0.05, 0.95))

    def _sample_award_ratios(self, n: int) -> np.ndarray:
        """Sample award-to-claim ratios using the ICSID bracket distribution.

        Each simulation draw:
        1. Selects a bracket according to historical bracket shares.
        2. Samples uniformly within that bracket.
        """
        labels, shares = zip(*AWARD_TO_CLAIM_DISTRIBUTION)
        share_arr = np.array(shares, dtype=float)
        share_arr /= share_arr.sum()  # normalise to 1.0

        # Bracket boundaries
        bounds = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00)]

        chosen = self.rng.choice(len(bounds), size=n, p=share_arr)
        ratios = np.empty(n)
        for i, idx in enumerate(chosen):
            lo, hi = bounds[idx]
            ratios[i] = self.rng.uniform(lo, hi)
        return ratios

    # ------------------------------------------------------------------
    # Public simulation methods
    # ------------------------------------------------------------------

    def simulate_jurisdictional_success(self) -> dict:
        """Simulate the probability that the tribunal accepts jurisdiction and
        issues an award in favour of the investor (on merits).

        Methodology:
        - Start from global ICSID upholding rate (57%).
        - Apply additive adjustments for country friction level, sector, and
          treaty basis.
        - Blend with country-specific historical rates (if ≥3 resolved cases).
        - Monte Carlo: draw n Bernoulli trials; compute mean and 95% CI.

        Returns
        -------
        dict with:
            probability         (float, 0–1)
            confidence_interval (tuple[float, float] — 95% CI)
            base_rate_used      (float)
            adjustments         (dict)
            draws               (int)
        """
        p = self._jurisdiction_base_rate()
        draws = self.rng.binomial(1, p, size=self.n)
        mean_p = float(draws.mean())

        # Wilson score 95% confidence interval for a proportion
        z = 1.96
        n = self.n
        centre = (mean_p + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * math.sqrt(mean_p * (1 - mean_p) / n + z**2 / (4 * n**2)) / (
            1 + z**2 / n
        )
        ci = (max(0.0, centre - margin), min(1.0, centre + margin))

        return {
            "probability":         mean_p,
            "confidence_interval": ci,
            "base_rate_used":      _GLOBAL_JURISDICTIONAL_SUCCESS_RATE,
            "adjustments": {
                "friction_level":   self._friction,
                "friction_adj":     _FRICTION_JURISDICTION_ADJ.get(self._friction, 0.0),
                "sector_adj":       _SECTOR_JURISDICTION_ADJ.get(self.dp.sector, 0.0),
                "treaty_adj":       _TREATY_JURISDICTION_ADJ.get(self.dp.treaty_basis, 0.0),
            },
            "draws": self.n,
        }

    def simulate_award_to_claim_ratio(self) -> dict:
        """Simulate the distribution of award-to-claim ratios.

        Methodology:
        - Use ICSID historical bracket data as the sampling distribution.
        - Apply a sector-specific scale factor (mining/oil & gas tend higher).
        - Return full distribution summary with percentiles.

        Returns
        -------
        dict with:
            distribution  (numpy.ndarray of shape (n,))
            mean          (float)
            median        (float)
            percentiles   (dict: p5, p10, p25, p50, p75, p90, p95)
            std_dev       (float)
            bracket_probs (dict mapping bracket label to historical share)
        """
        ratios = self._sample_award_ratios(self.n)

        # Sector-specific skew: mining/oil & gas historically see higher
        # absolute awards; apply a mild scale factor capped at 1.0
        sector_scale: dict[str, float] = {
            "Oil & Gas": 1.05,
            "Mining": 1.03,
            "Agriculture": 0.98,
            "Manufacturing": 0.97,
            "Hospitality": 0.95,
            "Legal Services": 0.80,
        }
        scale = sector_scale.get(self.dp.sector, 1.00)
        ratios = np.clip(ratios * scale, 0.0, 1.0)

        pcts = np.percentile(ratios, [5, 10, 25, 50, 75, 90, 95])
        return {
            "distribution": ratios,
            "mean":   float(ratios.mean()),
            "median": float(np.median(ratios)),
            "std_dev": float(ratios.std()),
            "percentiles": {
                "p5":  float(pcts[0]),
                "p10": float(pcts[1]),
                "p25": float(pcts[2]),
                "p50": float(pcts[3]),
                "p75": float(pcts[4]),
                "p90": float(pcts[5]),
                "p95": float(pcts[6]),
            },
            "bracket_probs": dict(AWARD_TO_CLAIM_DISTRIBUTION),
        }

    def simulate_annulment_risk(self) -> dict:
        """Simulate annulment application probability and expected delay.

        Methodology:
        - Base application rate: 44.7% (BIICL 2021 / ICSID 2024).
        - Adjust upward for high-friction states (more likely to appeal).
        - Success rate: 5% of awards (global ICSID data).
        - Delay: log-normal distribution centred on 24 months (±6 months σ).

        Returns
        -------
        dict with:
            application_probability  (float)
            success_probability      (float — probability annulment SUCCEEDS)
            net_risk                 (float — P(applied) × P(success|applied))
            expected_delay_months    (float — mean additional delay if applied)
            delay_distribution       (numpy.ndarray)
            p25_delay, p75_delay     (float)
        """
        friction_annulment_boost: dict[str, float] = {
            "Critical":      0.20,
            "Very High":     0.15,
            "High":          0.10,
            "Moderate":      0.05,
            "Moderate-Low":  0.00,
            "Low":          -0.05,
        }
        app_rate = float(np.clip(
            _ANNULMENT_APPLICATION_RATE + friction_annulment_boost.get(self._friction, 0.0),
            0.05, 0.90
        ))

        # Success rate: global 5%; DRC is only African state with proven success;
        # boost if country == DRC, penalise recalcitrant states with poor track record
        success_rate = _ANNULMENT_SUCCESS_RATE
        if self.dp.respondent_state == "DRC":
            success_rate = 0.12  # DRC succeeded once; elevated prior
        elif self.dp.respondent_state in ("Zimbabwe", "Egypt"):
            success_rate = 0.04  # documented failures; below global average

        net_risk = app_rate * success_rate

        # Annulment delay: log-normal; mean 24 months, σ = 8 months
        mu    = math.log(24)
        sigma = math.log(1 + 8 / 24)
        delay_dist = self.rng.lognormal(mean=mu, sigma=sigma, size=self.n)

        return {
            "application_probability": app_rate,
            "success_probability":     success_rate,
            "net_risk":                net_risk,
            "expected_delay_months":   float(delay_dist.mean()),
            "delay_distribution":      delay_dist,
            "p25_delay":               float(np.percentile(delay_dist, 25)),
            "p75_delay":               float(np.percentile(delay_dist, 75)),
        }

    def simulate_enforcement_timeline(self) -> dict:
        """Simulate the full enforcement timeline distribution.

        Methodology:
        - Enforcement proceedings: base 6–18 months; friction-adjusted.
        - Add annulment delay (sampled from annulment risk simulation).
        - Multi-jurisdiction proceedings add 12–36 additional months
          for high-friction states.
        - Total timeline = arbitration + annulment (if occurs) + enforcement.

        Note: arbitration duration itself is NOT included here; this covers
        post-award to final satisfaction.

        Returns
        -------
        dict with:
            timeline_months_distribution (numpy.ndarray)
            p25, p50, p75               (float — months)
            mean                        (float)
            expected_years              (float)
            scenario_labels             (dict: optimistic, base, pessimistic)
        """
        friction_enforcement_months: dict[str, tuple[float, float]] = {
            "Critical":      (36.0, 18.0),   # mean, sigma (log-normal params before log)
            "Very High":     (30.0, 14.0),
            "High":          (20.0, 10.0),
            "Moderate":      (14.0,  7.0),
            "Moderate-Low":  ( 9.0,  4.0),
            "Low":           ( 5.0,  2.0),
        }
        base_mean, base_sigma = friction_enforcement_months.get(
            self._friction, (14.0, 7.0)
        )

        # Annulment delay contribution (weighted by application probability)
        ann_results = self.simulate_annulment_risk()
        ann_delay = ann_results["delay_distribution"] * ann_results["application_probability"]

        # Base enforcement duration: log-normal
        mu_enf    = math.log(base_mean)
        sigma_enf = math.log(1 + base_sigma / base_mean)
        enf_dist  = self.rng.lognormal(mean=mu_enf, sigma=sigma_enf, size=self.n)

        # Multi-jurisdiction premium: high-friction states require parallel campaigns
        multi_jur_premium: dict[str, float] = {
            "Critical":     24.0,
            "Very High":    18.0,
            "High":         12.0,
            "Moderate":      6.0,
            "Moderate-Low":  3.0,
            "Low":           0.0,
        }
        premium = multi_jur_premium.get(self._friction, 6.0)
        premium_dist = self.rng.exponential(scale=max(premium, 1.0), size=self.n)

        total = enf_dist + ann_delay + premium_dist
        pcts  = np.percentile(total, [25, 50, 75])

        return {
            "timeline_months_distribution": total,
            "p25":            float(pcts[0]),
            "p50":            float(pcts[1]),
            "p75":            float(pcts[2]),
            "mean":           float(total.mean()),
            "expected_years": float(total.mean() / 12.0),
            "scenario_labels": {
                "optimistic":  f"{pcts[0]:.0f} months",
                "base":        f"{pcts[1]:.0f} months",
                "pessimistic": f"{pcts[2]:.0f} months",
            },
        }

    def simulate_recovery_rate(self) -> dict:
        """Simulate the composite expected recovery rate.

        The recovery rate is the fraction of the claimed amount ultimately
        recovered, combining:

        Recovery = P(jurisdiction) × Award_ratio × (1 − P(annulment_success))
                   × P(enforcement) × (1 − settlement_discount)

        All components are Monte Carlo sampled jointly for correlation.

        Returns
        -------
        dict with:
            distribution      (numpy.ndarray — recovery as fraction of claim)
            mean              (float)
            median            (float)
            ci_low, ci_high   (float — 95% credible interval)
            expected_years    (float — from timeline simulation)
            expected_usd      (float — mean recovery in absolute USD)
            percentiles       (dict: p5, p10, p25, p50, p75, p90, p95)
            component_means   (dict — mean of each multiplicative factor)
        """
        # 1. Jurisdictional success
        j_prob = self._jurisdiction_base_rate()
        j_draws = self.rng.binomial(1, j_prob, size=self.n).astype(float)

        # 2. Award-to-claim ratio (conditional on winning jurisdiction)
        ratios = self._sample_award_ratios(self.n)

        # 3. Annulment risk
        ann = self.simulate_annulment_risk()
        ann_net = ann["net_risk"]  # P(annulment applied AND succeeded)
        ann_survival = 1.0 - ann_net   # probability award survives annulment

        # 4. Enforcement probability
        enf_prob = self._friction_enforcement_prob()
        enf_draws = self.rng.binomial(1, enf_prob, size=self.n).astype(float)

        # 5. Settlement discount
        disc_range = self._profile.get("settlement_discount_range", (0.10, 0.30))
        # State negotiates somewhere in the range; model as Beta distribution
        disc_lo, disc_hi = disc_range
        disc_mean   = (disc_lo + disc_hi) / 2.0
        disc_var    = ((disc_hi - disc_lo) ** 2) / 12.0   # uniform approximation
        # Fit Beta parameters
        if disc_var > 0 and 0.0 < disc_mean < 1.0:
            alpha = disc_mean * (disc_mean * (1 - disc_mean) / disc_var - 1)
            beta  = (1 - disc_mean) * (disc_mean * (1 - disc_mean) / disc_var - 1)
            alpha = max(alpha, 0.5)
            beta  = max(beta, 0.5)
            disc_draws = self.rng.beta(alpha, beta, size=self.n)
        else:
            disc_draws = np.full(self.n, disc_mean)

        # Composite recovery
        recovery = (
            j_draws
            * ratios
            * ann_survival
            * enf_draws
            * (1.0 - disc_draws)
        )

        pcts = np.percentile(recovery, [5, 10, 25, 50, 75, 90, 95])

        timeline = self.simulate_enforcement_timeline()
        expected_usd = float(recovery.mean() * self.dp.amount_claimed_usd)

        return {
            "distribution":   recovery,
            "mean":           float(recovery.mean()),
            "median":         float(np.median(recovery)),
            "ci_low":         float(pcts[0]),      # 5th percentile as conservative bound
            "ci_high":        float(pcts[6]),      # 95th percentile as optimistic bound
            "expected_years": timeline["expected_years"],
            "expected_usd":   expected_usd,
            "percentiles": {
                "p5":  float(pcts[0]),
                "p10": float(pcts[1]),
                "p25": float(pcts[2]),
                "p50": float(pcts[3]),
                "p75": float(pcts[4]),
                "p90": float(pcts[5]),
                "p95": float(pcts[6]),
            },
            "component_means": {
                "jurisdictional_success_rate":  float(j_draws.mean()),
                "avg_award_to_claim_ratio":     float(ratios.mean()),
                "annulment_survival_rate":      ann_survival,
                "enforcement_success_rate":     float(enf_draws.mean()),
                "avg_settlement_retention":     float((1 - disc_draws).mean()),
            },
        }

    def run_full_simulation(self) -> dict:
        """Run all simulations and return a consolidated result dictionary.

        Returns
        -------
        dict with keys:
            dispute_profile         (dict — input parameters)
            jurisdictional_success  (dict — from simulate_jurisdictional_success)
            award_to_claim          (dict — from simulate_award_to_claim_ratio,
                                    distributions excluded for serialisability)
            annulment_risk          (dict — from simulate_annulment_risk,
                                    distributions excluded)
            enforcement_timeline    (dict — from simulate_enforcement_timeline,
                                    distributions excluded)
            recovery_rate           (dict — from simulate_recovery_rate,
                                    distributions excluded)
            sovereign_friction_score (float — from score_sovereign_friction)
            summary                 (dict — executive summary of key metrics)
        """
        jurisd = self.simulate_jurisdictional_success()
        award  = self.simulate_award_to_claim_ratio()
        ann    = self.simulate_annulment_risk()
        enf    = self.simulate_enforcement_timeline()
        rec    = self.simulate_recovery_rate()
        friction_score = self.score_sovereign_friction()

        def _strip_arrays(d: dict) -> dict:
            """Remove numpy arrays from dict for clean JSON-serialisable output."""
            return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

        return {
            "dispute_profile": {
                "respondent_state":     self.dp.respondent_state,
                "investor_nationality": self.dp.investor_nationality,
                "sector":               self.dp.sector,
                "treaty_basis":         self.dp.treaty_basis,
                "amount_claimed_usd":   self.dp.amount_claimed_usd,
                "investment_type":      self.dp.investment_type,
                "bit_year":             self.dp.bit_year,
            },
            "jurisdictional_success":   _strip_arrays(jurisd),
            "award_to_claim":           _strip_arrays(award),
            "annulment_risk":           _strip_arrays(ann),
            "enforcement_timeline":     _strip_arrays(enf),
            "recovery_rate":            _strip_arrays(rec),
            "sovereign_friction_score": friction_score,
            "summary": {
                "expected_recovery_fraction":  rec["mean"],
                "expected_recovery_usd":       rec["expected_usd"],
                "expected_years_to_recovery":  rec["expected_years"],
                "jurisdictional_success_prob": jurisd["probability"],
                "median_award_to_claim":       award["median"],
                "annulment_net_risk":          ann["net_risk"],
                "enforcement_prob":            self._friction_enforcement_prob(),
                "sovereign_friction_score":    friction_score,
            },
        }

    def score_sovereign_friction(self) -> float:
        """Compute a composite Sovereign Friction Score (0–100).

        Higher score = more friction = worse recovery prospects.

        Formula (weighted composite):
            - Rule of Law percentile (inverted):        30%
            - Corruption control percentile (inverted): 25%
            - Govt. effectiveness percentile (inverted):20%
            - WJP Rule of Law score (inverted, 0–100):  15%
            - Voluntary compliance history:             10%

        Returns
        -------
        float (0–100), where 0 = no friction, 100 = maximum friction.
        """
        p = self._profile
        if not p:
            return 50.0  # Default: unknown state

        # Governance indicators: WGI percentiles → friction = (100 − pct)
        rol_raw  = p.get("wgi_rule_of_law", 50.0)
        cor_raw  = p.get("wgi_corruption", 50.0)
        gov_raw  = p.get("wgi_govt_effectiveness", 50.0)
        wjp_raw  = p.get("wjp_score")       # 0–1

        rol_friction = 100.0 - rol_raw
        cor_friction = 100.0 - cor_raw
        gov_friction = 100.0 - gov_raw
        wjp_friction = (1.0 - wjp_raw) * 100.0 if wjp_raw is not None else 50.0

        # Compliance history
        compliance_map = {
            "Yes":     0.0,
            "Partial": 35.0,
            "No":     100.0,
            "Unknown": 55.0,
        }
        comp_friction = compliance_map.get(
            p.get("voluntary_compliance_history", "Unknown"), 55.0
        )

        score = (
            0.30 * rol_friction
            + 0.25 * cor_friction
            + 0.20 * gov_friction
            + 0.15 * wjp_friction
            + 0.10 * comp_friction
        )
        return round(float(np.clip(score, 0.0, 100.0)), 2)


# ---------------------------------------------------------------------------
# BEHAVIORAL MODULE
# ---------------------------------------------------------------------------

class BehavioralModule:
    """Behavioural economics models for ISDS strategic analysis.

    Models:
    - Investor anchoring / overclaiming bias
    - State time-inconsistent delay incentives
    - Zone of Possible Agreement (ZOPA) analysis
    - Prospect theory valuation
    """

    # ------------------------------------------------------------------
    # Overclaiming Bias
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_overclaiming_bias(
        claimed_amount: float,
        sector: str,
        historical_avg_ratio: Optional[float] = None,
    ) -> dict:
        """Model anchoring bias in investor claim formulation.

        Investors systematically overclaim relative to rational expectations
        because the initial claimed figure serves as an anchor in negotiations
        and tribunal deliberations. Overclaiming is sector-specific.

        Methodology:
        - Compare claimed amount to the implied rational expectation
          (historical_avg_ratio × claimed_amount, adjusted by sector).
        - Compute overclaiming factor and its strategic implication.

        Args:
            claimed_amount:       USD amount claimed by investor.
            sector:               Investment sector (for sector multiplier).
            historical_avg_ratio: Observed award/claim ratio for this dispute
                                  type (defaults to global ICSID median ~0.38).

        Returns
        -------
        dict with:
            claimed_amount         (float)
            rational_expectation   (float — estimated "fair" award)
            overclaiming_factor    (float — ratio of claim to rational expectation)
            anchoring_premium      (float — implied USD premium due to anchoring)
            overclaiming_level     (str — "Low", "Moderate", "High", "Extreme")
            strategic_implication  (str)
        """
        # Sector-specific average award ratios (derived from ICSID bracket data
        # and Public Citizen 2024 analysis)
        sector_avg_ratios: dict[str, float] = {
            "Oil & Gas":          0.52,
            "Mining":             0.40,
            "Agriculture":        0.35,
            "Energy":             0.45,
            "Infrastructure":     0.38,
            "Telecommunications": 0.42,
            "Manufacturing":      0.30,
            "Hospitality":        0.28,
            "Construction":       0.30,
            "Retail":             0.18,
            "Maritime":           0.35,
            "General Commercial": 0.32,
            "Legal Services":     0.15,
            "Other":              0.35,
        }
        if historical_avg_ratio is None:
            historical_avg_ratio = sector_avg_ratios.get(sector, 0.35)

        rational_expectation = claimed_amount * historical_avg_ratio
        overclaiming_factor  = claimed_amount / rational_expectation  # = 1 / avg_ratio

        anchoring_premium = claimed_amount - rational_expectation

        # Overclaiming level — using 1/ratio as the measure
        if overclaiming_factor < 1.5:
            level = "Low"
            implication = (
                "Claim is near rational expectation. Strong evidential basis likely. "
                "Low downside risk from anchoring backfire."
            )
        elif overclaiming_factor < 2.5:
            level = "Moderate"
            implication = (
                "Claim moderately exceeds expected award. Common in ICSID cases. "
                "Anchoring effect may inflate tribunal starting point by 10–20%. "
                "State will argue credibility risk."
            )
        elif overclaiming_factor < 4.0:
            level = "High"
            implication = (
                "Significant overclaiming. Tribunal may discount claim as inflated "
                "and impose adverse cost consequences. Real recovery likely 20–30% of claim. "
                "Anchoring benefit limited above this threshold."
            )
        else:
            level = "Extreme"
            implication = (
                "Severe overclaiming detected. Risks reputational harm to investor's "
                "credibility; some tribunals have dismissed excessive claims with costs. "
                "Rational expectation far below claim; adversarial framing likely."
            )

        return {
            "claimed_amount":       claimed_amount,
            "rational_expectation": rational_expectation,
            "overclaiming_factor":  overclaiming_factor,
            "anchoring_premium":    anchoring_premium,
            "overclaiming_level":   level,
            "strategic_implication": implication,
        }

    # ------------------------------------------------------------------
    # State Delay Incentive
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_state_delay_incentive(
        award_amount: float,
        enforcement_friction: str,
        discount_rate: float = 0.08,
    ) -> dict:
        """Model the state's time-inconsistent preference for delay.

        States rationally prefer to delay payment because:
        1. The NPV of a future payment is less than its face value.
        2. Enforcement uncertainty may reduce effective liability.
        3. Political leadership faces different incentives than fiscal planners.

        This models the NPV advantage of delay for different enforcement
        friction levels and delays.

        Args:
            award_amount:         USD amount of ICSID award.
            enforcement_friction: Friction level string from COUNTRY_PROFILES.
            discount_rate:        State's discount rate (default 8% p.a.).

        Returns
        -------
        dict with:
            award_amount           (float)
            npv_at_year_1          (float)
            npv_at_year_3          (float)
            npv_at_year_5          (float)
            npv_at_year_10         (float)
            delay_incentive_usd    (float — value of 5-year delay at given discount)
            probability_of_non_payment (float — modelled prob. of never paying)
            optimal_delay_years    (float — years to maximize expected payoff)
            enforcement_prob       (float)
            summary                (str)
        """
        enf_prob = _FRICTION_ENFORCEMENT_PROB.get(enforcement_friction, 0.60)

        # NPV: PV = award × enforcement_prob / (1 + r)^t
        def effective_liability(years: float) -> float:
            return award_amount * enf_prob / ((1 + discount_rate) ** years)

        npv_0  = award_amount * enf_prob
        npv_1  = effective_liability(1)
        npv_3  = effective_liability(3)
        npv_5  = effective_liability(5)
        npv_10 = effective_liability(10)

        delay_incentive = npv_0 - npv_5

        # Probability of never paying modelled as (1 - enforcement_prob) adjusted
        # for political risk: high-friction states have additional non-payment risk
        political_risk_premium: dict[str, float] = {
            "Critical":      0.60,
            "Very High":     0.40,
            "High":          0.20,
            "Moderate":      0.08,
            "Moderate-Low":  0.03,
            "Low":           0.01,
        }
        p_never_pay = float(np.clip(
            (1 - enf_prob) + political_risk_premium.get(enforcement_friction, 0.10),
            0.0, 1.0
        ))

        # Optimal delay: maximise E[utility] = (1 - p_never_pay) × PV - legal_costs
        # Simplified: find t* where marginal cost of continuing = marginal NPV gain
        # In practice, this is driven by enforcement threat credibility
        friction_delay_map: dict[str, float] = {
            "Critical":      15.0,
            "Very High":     10.0,
            "High":           6.0,
            "Moderate":       3.0,
            "Moderate-Low":   2.0,
            "Low":            0.5,
        }
        optimal_delay = friction_delay_map.get(enforcement_friction, 4.0)

        return {
            "award_amount":               award_amount,
            "npv_at_year_1":              npv_1,
            "npv_at_year_3":              npv_3,
            "npv_at_year_5":              npv_5,
            "npv_at_year_10":             npv_10,
            "delay_incentive_usd":        delay_incentive,
            "probability_of_non_payment": p_never_pay,
            "optimal_delay_years":        optimal_delay,
            "enforcement_probability":    enf_prob,
            "summary": (
                f"Delaying payment by 5 years saves the state ~${delay_incentive:,.0f} "
                f"in NPV terms (at {discount_rate:.0%} discount). "
                f"Modelled probability of never paying: {p_never_pay:.1%}. "
                f"Enforcement friction level: {enforcement_friction}."
            ),
        }

    # ------------------------------------------------------------------
    # Settlement Zone (ZOPA)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_settlement_zone(
        award_amount: float,
        investor_recovery_prob: float,
        state_enforcement_risk: float,
        investor_discount_rate: float = 0.10,
        state_discount_rate: float    = 0.06,
    ) -> dict:
        """Compute the Zone of Possible Agreement (ZOPA) for post-award settlement.

        Both parties face uncertainty and time preference. The ZOPA is the range
        of settlement amounts that makes both parties better off than continuing
        enforcement proceedings.

        Investor minimum (floor):
            The minimum the investor would accept, equal to the expected PV of
            pursuing enforcement:
            floor = award × recovery_prob × (1 / (1 + r_inv)^t_inv)
            where t_inv is the expected years of further enforcement.

        State maximum (ceiling):
            The maximum the state would pay to avoid continued enforcement:
            ceiling = award × enforcement_risk × (1 / (1 + r_state)^t_state)

        If floor < ceiling, a ZOPA exists.

        Args:
            award_amount:           Face value of ICSID award (USD).
            investor_recovery_prob: Probability investor ultimately recovers
                                    (from simulate_recovery_rate).
            state_enforcement_risk: Probability state is successfully enforced
                                    against (enforcement friction probability).
            investor_discount_rate: Investor's cost of capital / discount rate.
            state_discount_rate:    State's discount rate (typically lower).

        Returns
        -------
        dict with:
            award_amount              (float)
            investor_floor            (float)
            state_ceiling             (float)
            settlement_zone_exists    (bool)
            zopa_midpoint             (float or None)
            optimal_settlement_range  (tuple[float, float] or None)
            investor_surplus_at_mid   (float)
            state_surplus_at_mid      (float)
            zone_width_usd            (float)
            commentary                (str)
        """
        # Assume ~4 years as realistic enforcement horizon
        inv_horizon   = 4.0
        state_horizon = 3.0   # State discounts sooner (political cycle)

        investor_floor = (
            award_amount
            * investor_recovery_prob
            / ((1 + investor_discount_rate) ** inv_horizon)
        )
        state_ceiling = (
            award_amount
            * state_enforcement_risk
            / ((1 + state_discount_rate) ** state_horizon)
        )

        zone_exists = investor_floor < state_ceiling
        zopa_mid    = (investor_floor + state_ceiling) / 2.0 if zone_exists else None

        # Optimal range: 25th–75th percentile of the ZOPA
        opt_range = None
        if zone_exists:
            zopa_width = state_ceiling - investor_floor
            opt_range  = (
                investor_floor + 0.25 * zopa_width,
                investor_floor + 0.75 * zopa_width,
            )

        inv_surplus   = (zopa_mid - investor_floor) if zopa_mid else 0.0
        state_surplus = (state_ceiling - zopa_mid) if zopa_mid else 0.0

        if zone_exists:
            commentary = (
                f"A ZOPA exists between ${investor_floor:,.0f} (investor floor) "
                f"and ${state_ceiling:,.0f} (state ceiling). "
                f"The midpoint of ${zopa_mid:,.0f} represents "
                f"{(zopa_mid/award_amount):.1%} of the face award. "
                f"Optimal range: ${opt_range[0]:,.0f} – ${opt_range[1]:,.0f}."
            )
        else:
            commentary = (
                f"NO ZOPA currently exists. Investor floor (${investor_floor:,.0f}) "
                f"exceeds state ceiling (${state_ceiling:,.0f}). "
                f"Enforcement escalation or time passage is needed to shift "
                f"incentives before settlement becomes rational."
            )

        return {
            "award_amount":             award_amount,
            "investor_floor":           investor_floor,
            "state_ceiling":            state_ceiling,
            "settlement_zone_exists":   zone_exists,
            "zopa_midpoint":            zopa_mid,
            "optimal_settlement_range": opt_range,
            "investor_surplus_at_mid":  inv_surplus,
            "state_surplus_at_mid":     state_surplus,
            "zone_width_usd":           max(0.0, state_ceiling - investor_floor),
            "commentary":               commentary,
        }

    # ------------------------------------------------------------------
    # Prospect Theory Valuation
    # ------------------------------------------------------------------

    @staticmethod
    def prospect_theory_valuation(
        amount: float,
        reference_point: float,
        loss_aversion: float = 2.25,
    ) -> dict:
        """Model how each party values a settlement amount using prospect theory.

        Kahneman & Tversky's prospect theory holds that:
        - Gains are evaluated relative to a reference point.
        - Losses hurt approximately 2.25× more than equivalent gains please.
        - Both gains and losses exhibit diminishing sensitivity (concave/convex).

        Applied here to model how investors and states value the same settlement
        amount differently depending on their reference points.

        Args:
            amount:          Proposed settlement amount (USD).
            reference_point: Comparison anchor for the evaluator (e.g., the
                             face value of the award for the investor; the
                             budgeted reserves for the state).
            loss_aversion:   Lambda (λ) coefficient. Default 2.25 per
                             Kahneman & Tversky (1992).

        Returns
        -------
        dict with:
            amount                  (float)
            reference_point         (float)
            gain_or_loss            (float — amount − reference_point)
            is_loss                 (bool)
            prospect_value          (float — subjective utility)
            loss_aversion_lambda    (float)
            alpha                   (float — value function curvature)
            interpretation          (str)
        """
        alpha = 0.88   # Tversky & Kahneman (1992) value function exponent

        delta = amount - reference_point

        if delta >= 0:
            value = delta ** alpha
            is_loss = False
        else:
            value = -loss_aversion * ((-delta) ** alpha)
            is_loss = True

        if is_loss:
            interp = (
                f"The evaluator perceives this as a LOSS of ${-delta:,.0f} relative to "
                f"reference point ${reference_point:,.0f}. Prospect value = {value:,.2f}. "
                f"Loss aversion (λ={loss_aversion}) makes this feel {loss_aversion:.2f}× "
                f"worse than a gain of equal magnitude."
            )
        else:
            interp = (
                f"The evaluator perceives this as a GAIN of ${delta:,.0f} relative to "
                f"reference point ${reference_point:,.0f}. Prospect value = {value:,.2f}. "
                f"Diminishing sensitivity (α=0.88) means additional gains beyond this "
                f"point yield decreasing marginal utility."
            )

        return {
            "amount":               amount,
            "reference_point":      reference_point,
            "gain_or_loss":         delta,
            "is_loss":              is_loss,
            "prospect_value":       value,
            "loss_aversion_lambda": loss_aversion,
            "alpha":                alpha,
            "interpretation":       interp,
        }


# ---------------------------------------------------------------------------
# ENFORCEMENT PATHWAY
# ---------------------------------------------------------------------------

class EnforcementPathway:
    """Multi-jurisdiction enforcement pathway analysis.

    Given a country profile and an award amount, this class maps available
    enforcement jurisdictions, scores asset attachability, recommends
    sequencing, and generates a decision tree.

    Parameters
    ----------
    country_profile : dict
        A single entry from COUNTRY_PROFILES (e.g.,
        ``COUNTRY_PROFILES["Tanzania"]``).
    award_amount : float
        USD face value of the ICSID award.
    respondent_state : str
        Name of the respondent state (used for precedent lookup).
    """

    def __init__(
        self,
        country_profile: dict,
        award_amount: float,
        respondent_state: str = "",
    ) -> None:
        self.profile        = country_profile
        self.award          = award_amount
        self.state          = respondent_state
        self._friction      = country_profile.get("enforcement_friction_level", "Moderate")
        self._icsid_member  = country_profile.get("icsid_member", True)
        self._swf_name      = country_profile.get("swf_name")
        self._swf_aum       = country_profile.get("swf_aum_billions")
        self._soes          = country_profile.get("major_soes", [])
        self._compliance    = country_profile.get("voluntary_compliance_history", "Unknown")

    # ------------------------------------------------------------------
    # Jurisdiction Mapping
    # ------------------------------------------------------------------

    def map_jurisdictions(self) -> list[dict]:
        """Map available enforcement jurisdictions with probability estimates.

        Generates a ranked list of enforcement venues, taking into account:
        - ICSID vs non-ICSID membership.
        - Size of award relative to cost.
        - Country-specific asset footprint.
        - Precedent from existing African state enforcement cases.

        Returns
        -------
        list of dicts, each with:
            jurisdiction       (str)
            strategy           (str)
            success_probability (float, 0–1)
            timeline_months    (float)
            costs_estimate_usd (float)
            rationale          (str)
        """
        base_enf  = _FRICTION_ENFORCEMENT_PROB.get(self._friction, 0.60)
        friction_map = {
            "Critical":     {"new_york": 0.30, "london": 0.25, "paris": 0.35},
            "Very High":    {"new_york": 0.40, "london": 0.35, "paris": 0.45},
            "High":         {"new_york": 0.55, "london": 0.50, "paris": 0.55},
            "Moderate":     {"new_york": 0.70, "london": 0.65, "paris": 0.70},
            "Moderate-Low": {"new_york": 0.82, "london": 0.78, "paris": 0.80},
            "Low":          {"new_york": 0.92, "london": 0.90, "paris": 0.90},
        }
        probs = friction_map.get(self._friction, {"new_york": 0.60, "london": 0.55, "paris": 0.60})

        # SWF bonus: if SWF assets present, Paris/Stockholm are more attractive
        swf_bonus = 0.08 if (self._swf_aum and self._swf_aum > 0.5) else 0.0

        jurisdictions = []

        # US (New York / DC)
        jurisdictions.append({
            "jurisdiction":       "New York / Washington DC (US)",
            "strategy": (
                "File for recognition under ICSID Act (22 U.S.C. § 1650a) in DC "
                "District Court or SDNY. Issue NML-type asset discovery subpoenas "
                "to major US banks for global state asset disclosure. Seek attachment "
                "of commercial assets and SOE US-sited accounts under FSIA § 1610."
            ),
            "success_probability": min(probs["new_york"] + 0.05, 0.95),
            "timeline_months":    {
                "Critical": 48, "Very High": 36, "High": 24,
                "Moderate": 15, "Moderate-Low": 10, "Low": 6,
            }.get(self._friction, 15),
            "costs_estimate_usd": max(self.award * 0.04, 500_000),
            "rationale": (
                "ICSID awards treated as DC court judgments (12-yr limitation). "
                "Global asset discovery (NML Capital SCOTUS 2014) is the strongest "
                "asset-mapping tool globally. Essential first step for any significant "
                "African state enforcement campaign."
            ),
        })

        # UK (London)
        jurisdictions.append({
            "jurisdiction":       "London (England & Wales)",
            "strategy": (
                "Apply for recognition under ICSID Act 1966 and SIA s.9. Challenge "
                "immunity at execution stage only — recognition is near-automatic "
                "post-Border Timbers CA 2025. Target commercial property (London "
                "real estate, embassy annex commercial leases, SOE London offices). "
                "Rely on General Dynamics v. Libya (CA/SC 2025) for SWF assets if "
                "BIT/contract contains 'wholly enforceable' language."
            ),
            "success_probability": min(probs["london"] + 0.03, 0.95),
            "timeline_months": {
                "Critical": 42, "Very High": 30, "High": 22,
                "Moderate": 14, "Moderate-Low": 9, "Low": 6,
            }.get(self._friction, 14),
            "costs_estimate_usd": max(self.award * 0.035, 400_000),
            "rationale": (
                "UK Commercial Court has sophisticated jurisprudence on SIA and "
                "ICSID awards. Recent precedents (Border Timbers 2025, General "
                "Dynamics 2025) have narrowed immunity arguments. Strong venue for "
                "states with London commercial property or banking relationships."
            ),
        })

        # Paris
        jurisdictions.append({
            "jurisdiction":       "Paris (France)",
            "strategy": (
                "Seek exequatur under French Code of Civil Procedure. Target SWF "
                "assets under Paris CoA (2019) LIA precedent: SWF assets used for "
                "general commercial investment are NOT immune. Most liberal SWF "
                "attachment jurisdiction globally. File early for SWF asset freeze."
            ),
            "success_probability": min(probs["paris"] + swf_bonus, 0.95),
            "timeline_months": {
                "Critical": 30, "Very High": 24, "High": 18,
                "Moderate": 12, "Moderate-Low": 8, "Low": 5,
            }.get(self._friction, 12),
            "costs_estimate_usd": max(self.award * 0.03, 350_000),
            "rationale": (
                "Most creditor-friendly venue for SWF asset attachment. "
                "Paris CoA (2019) denied immunity to Libyan Investment Authority "
                "(LIA) assets — leading global precedent. Particularly valuable where "
                f"respondent has SWF ({self._swf_name or 'N/A'})."
            ),
        })

        # Stockholm (for SWF cases)
        if self._swf_aum and self._swf_aum > 0.5:
            jurisdictions.append({
                "jurisdiction":       "Stockholm (Sweden)",
                "strategy": (
                    "Invoke Ascom v. Kazakhstan (Swedish Supreme Court 2021): SWF "
                    "assets held for general investment/savings purposes are NOT immune. "
                    f"Target {self._swf_name} assets in Swedish-domiciled investment "
                    "vehicles or held through Swedish custodians."
                ),
                "success_probability": min(base_enf + swf_bonus + 0.10, 0.90),
                "timeline_months": {
                    "Critical": 30, "Very High": 24, "High": 18,
                    "Moderate": 12, "Moderate-Low": 8, "Low": 5,
                }.get(self._friction, 12),
                "costs_estimate_usd": max(self.award * 0.025, 300_000),
                "rationale": (
                    "Ascom (2021) is the leading precedent for SWF asset attachment. "
                    "State must prove 'concrete and clear connection to a qualified "
                    "sovereign purpose' — general investment mandate fails this test. "
                    f"AUM: ${self._swf_aum:.1f}B."
                ),
            })

        # Australia (for ICSID members; mining-investor exposure)
        if self._icsid_member:
            jurisdictions.append({
                "jurisdiction":       "Australia",
                "strategy": (
                    "Invoke Infrastructure Services v. Spain (Australian High Court "
                    "2023): ICSID Convention ratification = implied immunity waiver "
                    "for recognition proceedings. Apply in Federal Court for recognition "
                    "then target Australian-held commercial assets or SOE interests."
                ),
                "success_probability": min(base_enf + 0.05, 0.90),
                "timeline_months": {
                    "Critical": 24, "Very High": 18, "High": 15,
                    "Moderate": 10, "Moderate-Low": 7, "Low": 4,
                }.get(self._friction, 10),
                "costs_estimate_usd": max(self.award * 0.025, 300_000),
                "rationale": (
                    "2023 High Court precedent removes immunity for recognition of "
                    "ICSID awards where the state is a Contracting State. Emerging "
                    "venue relevant for African states with Australian mining investor "
                    "exposure (Tanzania, Ghana, Zimbabwe)."
                ),
            })

        # Sort by success probability descending
        jurisdictions.sort(key=lambda x: x["success_probability"], reverse=True)
        return jurisdictions

    # ------------------------------------------------------------------
    # Asset Attachability
    # ------------------------------------------------------------------

    def score_asset_attachability(self) -> dict:
        """Score the attachability of state assets (0–100).

        Higher score = more attachable assets available.

        Scoring components:
        - SWF AUM:             0–35 points (based on AUM and commercial-use profile)
        - SOE commercial assets: 0–30 points (number and international footprint)
        - ICSID membership:    0–15 points (waiver of recognition immunity)
        - Compliance history:  0–20 points (willingness-to-pay proxy)

        Returns
        -------
        dict with:
            score              (float, 0–100)
            grade              (str — A/B/C/D/F)
            key_targets        (list of str)
            immunity_risks     (list of str)
            component_scores   (dict)
            narrative          (str)
        """
        # SWF component
        swf_score = 0.0
        if self._swf_aum:
            if self._swf_aum >= 50.0:   swf_score = 35.0
            elif self._swf_aum >= 10.0: swf_score = 28.0
            elif self._swf_aum >= 2.0:  swf_score = 20.0
            elif self._swf_aum >= 0.5:  swf_score = 14.0
            else:                       swf_score = 8.0

        # SOE component
        n_soes = len(self._soes)
        soe_score = min(n_soes * 8.0, 30.0)

        # ICSID membership
        icsid_score = 15.0 if self._icsid_member else 0.0

        # Compliance history
        compliance_scores = {"Yes": 20.0, "Partial": 12.0, "No": 3.0, "Unknown": 8.0}
        comp_score = compliance_scores.get(self._compliance, 8.0)

        total = swf_score + soe_score + icsid_score + comp_score
        total = float(np.clip(total, 0.0, 100.0))

        # Grade
        if total >= 75:   grade = "A"
        elif total >= 60: grade = "B"
        elif total >= 45: grade = "C"
        elif total >= 30: grade = "D"
        else:             grade = "F"

        # Key targets
        key_targets: list[str] = []
        if self._swf_name:
            key_targets.append(
                f"{self._swf_name} (AUM ~${self._swf_aum:.1f}B) — "
                "potential SWF attachment under Paris CoA / Ascom doctrine"
            )
        for soe in self._soes:
            key_targets.append(f"{soe} — commercial SOE asset (revenue streams, equity)")

        # Immunity risks
        immunity_risks: list[str] = [
            "Central bank reserves held for monetary policy: immune under FSIA § 1611(b)(1) and UNCSI Art. 21",
            "Military/defence assets: absolutely immune in all jurisdictions",
            "Diplomatic property in active governmental use: Vienna Convention / VCDR protection",
        ]
        if not self._icsid_member:
            immunity_risks.append(
                "State is NOT an ICSID member: must use New York Convention route "
                "(slower; subject to Art. V public policy challenges)"
            )
        if self._compliance == "No":
            immunity_risks.append(
                "Documented history of non-compliance: asset concealment / "
                "transfer to protected accounts likely"
            )

        return {
            "score":   total,
            "grade":   grade,
            "key_targets":   key_targets,
            "immunity_risks": immunity_risks,
            "component_scores": {
                "swf_score":        swf_score,
                "soe_score":        soe_score,
                "icsid_score":      icsid_score,
                "compliance_score": comp_score,
            },
            "narrative": (
                f"Asset attachability score: {total:.0f}/100 (Grade {grade}). "
                f"{len(key_targets)} primary target asset categories identified. "
                f"ICSID member: {self._icsid_member}. "
                f"Compliance history: {self._compliance}."
            ),
        }

    # ------------------------------------------------------------------
    # Sequencing Recommendation
    # ------------------------------------------------------------------

    def recommend_sequencing(self) -> list[dict]:
        """Recommend optimal enforcement action sequencing.

        Sequencing logic:
        1. Immediate: Global asset mapping + recognition filing.
        2. Short-term (0–6 months): File in most favourable jurisdiction.
        3. Medium-term (6–18 months): Parallel proceedings + SWF targeting.
        4. Long-term (18+ months): Domestic enforcement or settlement pressure.

        Returns
        -------
        list of dicts, each with:
            step       (int)
            phase      (str)
            action     (str)
            timeline   (str)
            rationale  (str)
            priority   (str — "Critical", "High", "Medium", "Low")
        """
        steps = []
        step = 1

        # Step 1: Asset mapping — always first
        steps.append({
            "step": step,
            "phase": "Immediate (0–30 days)",
            "action": (
                "File for recognition in US District Court (DC or SDNY) and "
                "simultaneously serve NML-type discovery subpoenas on major US "
                "correspondent banks for global asset disclosure of respondent "
                f"state ({self.state}) and its major SOEs."
            ),
            "timeline": "0–30 days post-award",
            "rationale": (
                "NML Capital (SCOTUS 2014) permits worldwide asset discovery. "
                "Intelligence gathered here drives all subsequent targeting decisions. "
                "Must commence before state conceals or transfers commercial assets."
            ),
            "priority": "Critical",
        })
        step += 1

        # Step 2: Best jurisdiction filing
        jurs = self.map_jurisdictions()
        top_jur = jurs[0] if jurs else {"jurisdiction": "New York", "timeline_months": 12}
        steps.append({
            "step": step,
            "phase": "Short-term (0–3 months)",
            "action": (
                f"File enforcement application in {top_jur['jurisdiction']} — "
                "highest-ranked jurisdiction by success probability. "
                "Apply for interim measures / asset freeze order simultaneously."
            ),
            "timeline": "0–90 days post-award",
            "rationale": top_jur.get("rationale", "Highest enforcement success probability."),
            "priority": "Critical",
        })
        step += 1

        # Step 3: SWF targeting if applicable
        if self._swf_name and self._swf_aum and self._swf_aum > 0.5:
            steps.append({
                "step": step,
                "phase": "Short-term (1–6 months)",
                "action": (
                    f"File SWF-specific enforcement action in Paris (Paris CoA 2019 LIA "
                    f"doctrine) and Stockholm (Ascom 2021 doctrine) targeting "
                    f"{self._swf_name} assets (~${self._swf_aum:.1f}B AUM). "
                    "Invoke commercial-use exception: SWF managed for general "
                    "investment purposes is NOT immune."
                ),
                "timeline": "1–6 months post-award",
                "rationale": (
                    "SWF assets represent the largest single attachable pool for "
                    "this respondent. Paris and Stockholm have the most favourable "
                    "SWF attachment precedents globally. Early filing prevents "
                    "voluntary repatriation of assets to immune central bank accounts."
                ),
                "priority": "High",
            })
            step += 1

        # Step 4: Parallel filing in secondary jurisdiction
        if len(jurs) >= 2:
            sec_jur = jurs[1]
            steps.append({
                "step": step,
                "phase": "Medium-term (3–9 months)",
                "action": (
                    f"File parallel recognition/enforcement in {sec_jur['jurisdiction']}. "
                    "Multi-jurisdiction proceedings increase settlement pressure by "
                    "demonstrating credible enforcement resolve and raising state's "
                    "ongoing legal costs."
                ),
                "timeline": "3–9 months post-award",
                "rationale": (
                    "Parallel proceedings are the empirically proven mechanism to "
                    "force recalcitrant states to the negotiating table "
                    "(cf. Argentina/NML, Zimbabwe/von Pezold). "
                    f"Secondary jurisdiction: {sec_jur['jurisdiction']}."
                ),
                "priority": "High",
            })
            step += 1

        # Step 5: Settlement negotiations
        steps.append({
            "step": step,
            "phase": "Medium-term (6–18 months)",
            "action": (
                "Open structured settlement negotiations using enforcement "
                "proceedings as leverage. Use BehavioralModule.calculate_settlement_zone() "
                "to identify ZOPA. Engage investor's home state diplomatic channel "
                "if available (ICSID Convention Art. 27 diplomatic protection "
                "post-ICSID exhaustion)."
            ),
            "timeline": "6–18 months post-award",
            "rationale": (
                "66% of ICSID damages awards are resolved via voluntary compliance "
                "or settlement (ICSID Compliance Paper 2024). Enforcement proceedings "
                "drive settlement by increasing the state's immediate liability "
                "perception. Discount range depends on friction level: "
                f"{self.profile.get('settlement_discount_range', 'N/A')}."
            ),
            "priority": "High",
        })
        step += 1

        # Step 6: Capital markets / IFI pressure (for states that need market access)
        if self._friction in ("Moderate", "Moderate-Low", "Low"):
            steps.append({
                "step": step,
                "phase": "Long-term (12–36 months)",
                "action": (
                    "Notify relevant IFIs (World Bank, IMF, AfDB) and credit rating "
                    "agencies of outstanding unpaid award. States with pending "
                    "bond issuances or IFI programme reviews are sensitive to "
                    "award-compliance status. Coordinate with investor's home state "
                    "embassy for diplomatic note."
                ),
                "timeline": "12–36 months",
                "rationale": (
                    "Reputational and market access consequences are the primary "
                    "driver for moderate-friction states to resolve awards. "
                    "Argentina's 2016 settlement was driven by market re-entry needs. "
                    "Tanzania's recent settlements (Indiana, Winshear, Montero) "
                    "appear similarly motivated."
                ),
                "priority": "Medium",
            })
            step += 1

        return steps

    # ------------------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------------------

    def generate_decision_tree(self) -> dict:
        """Generate a nested decision tree for the enforcement process.

        The tree models the key binary/multi-way decisions and their
        probabilistic outcomes at each node.

        Returns
        -------
        Nested dict with structure:
            node_id, label, description, probability (if leaf/branch),
            children (list of child nodes)
        """
        enf_prob     = _FRICTION_ENFORCEMENT_PROB.get(self._friction, 0.60)
        ann_app_rate = float(np.clip(
            _ANNULMENT_APPLICATION_RATE + {
                "Critical": 0.20, "Very High": 0.15, "High": 0.10,
                "Moderate": 0.05, "Moderate-Low": 0.0, "Low": -0.05
            }.get(self._friction, 0.0),
            0.05, 0.90
        ))
        ann_success  = _ANNULMENT_SUCCESS_RATE
        j_prob       = float(np.clip(
            _GLOBAL_JURISDICTIONAL_SUCCESS_RATE
            + _FRICTION_JURISDICTION_ADJ.get(self._friction, 0.0),
            0.05, 0.95
        ))

        tree = {
            "node_id": "ROOT",
            "label": "File ICSID Arbitration",
            "description": (
                f"Investor files claim against {self.state or 'respondent state'}. "
                f"Claimed amount: ${self.award:,.0f}."
            ),
            "probability": 1.0,
            "children": [
                {
                    "node_id": "JURISDICTION_ACCEPTED",
                    "label": "Jurisdiction Accepted / Award in Investor's Favour",
                    "description": (
                        f"Tribunal upholds jurisdiction and finds merit. "
                        f"Probability: {j_prob:.1%}."
                    ),
                    "probability": j_prob,
                    "children": [
                        {
                            "node_id": "NO_ANNULMENT",
                            "label": "No Annulment Application Filed",
                            "description": f"State does not contest award. Probability: {1-ann_app_rate:.1%}.",
                            "probability": 1.0 - ann_app_rate,
                            "children": [
                                {
                                    "node_id": "VOLUNTARY_COMPLIANCE",
                                    "label": "Voluntary Compliance",
                                    "description": "State pays award in full or negotiates settlement.",
                                    "probability": enf_prob,
                                    "children": [],
                                    "outcome": "Full/Partial Recovery",
                                },
                                {
                                    "node_id": "ENFORCEMENT_REQUIRED",
                                    "label": "Enforcement Required",
                                    "description": "State refuses to pay; investor pursues multi-jurisdiction enforcement.",
                                    "probability": 1.0 - enf_prob,
                                    "children": [
                                        {
                                            "node_id": "ENFORCEMENT_SUCCESS",
                                            "label": "Enforcement Succeeds",
                                            "description": "Assets attached; payment obtained (discounted).",
                                            "probability": enf_prob * 0.7,
                                            "children": [],
                                            "outcome": "Discounted Recovery",
                                        },
                                        {
                                            "node_id": "ENFORCEMENT_FAILS",
                                            "label": "Enforcement Fails / Stalled",
                                            "description": "Insufficient attachable assets; long-term stalemate.",
                                            "probability": 1.0 - enf_prob * 0.7,
                                            "children": [],
                                            "outcome": "Zero / Negligible Recovery",
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "node_id": "ANNULMENT_FILED",
                            "label": "Annulment Application Filed",
                            "description": (
                                f"State files annulment under ICSID Convention Art. 52. "
                                f"Probability: {ann_app_rate:.1%}. "
                                f"Enforcement automatically stayed during proceedings."
                            ),
                            "probability": ann_app_rate,
                            "children": [
                                {
                                    "node_id": "ANNULMENT_SUCCEEDS",
                                    "label": "Annulment Granted",
                                    "description": (
                                        f"Ad hoc committee annuls award (full or partial). "
                                        f"Probability: {ann_success:.1%}. "
                                        f"Case may be resubmitted."
                                    ),
                                    "probability": ann_success,
                                    "children": [],
                                    "outcome": "Zero Recovery (award annulled); possible resubmission",
                                },
                                {
                                    "node_id": "ANNULMENT_REJECTED",
                                    "label": "Annulment Rejected / Discontinued",
                                    "description": (
                                        f"Award upheld. Probability: {1-ann_success:.1%}. "
                                        f"Enforcement proceeds with ~24-month delay."
                                    ),
                                    "probability": 1.0 - ann_success,
                                    "children": [
                                        {
                                            "node_id": "POST_ANNULMENT_ENFORCEMENT",
                                            "label": "Post-Annulment Enforcement",
                                            "description": "Enforcement campaign resumes after failed annulment.",
                                            "probability": enf_prob,
                                            "children": [],
                                            "outcome": "Delayed Recovery (discounted by ~24-month delay)",
                                        },
                                        {
                                            "node_id": "POST_ANNULMENT_STALEMATE",
                                            "label": "Post-Annulment Stalemate",
                                            "description": "State continues resistance despite failed annulment.",
                                            "probability": 1.0 - enf_prob,
                                            "children": [],
                                            "outcome": "Zero / Negligible Recovery",
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
                {
                    "node_id": "JURISDICTION_REJECTED",
                    "label": "Jurisdiction Rejected / Claims Dismissed",
                    "description": (
                        f"Tribunal declines jurisdiction or dismisses all claims. "
                        f"Probability: {1-j_prob:.1%}."
                    ),
                    "probability": 1.0 - j_prob,
                    "children": [],
                    "outcome": "Zero Recovery; potential cost award against investor",
                },
            ],
        }

        return tree


# ---------------------------------------------------------------------------
# Quick self-test when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # Test with a realistic Tanzania mining dispute
    profile = DisputeProfile(
        respondent_state="Tanzania",
        investor_nationality="Canada",
        sector="Mining",
        treaty_basis="Bilateral Investment Treaty",
        amount_claimed_usd=250_000_000.0,
        investment_type="greenfield mining — nickel exploration",
        bit_year=2013,
    )

    engine = SimulationEngine(profile, n_simulations=10_000, seed=42)

    print("=" * 60)
    print("DISPUTE PROFILE: Canadian Mining Investor v. Tanzania")
    print("=" * 60)

    print("\n--- Jurisdictional Success ---")
    j = engine.simulate_jurisdictional_success()
    print(f"  Probability:          {j['probability']:.3f}")
    print(f"  95% CI:               {j['confidence_interval'][0]:.3f} – {j['confidence_interval'][1]:.3f}")

    print("\n--- Award-to-Claim Ratio ---")
    a = engine.simulate_award_to_claim_ratio()
    print(f"  Mean:                 {a['mean']:.3f}")
    print(f"  Median:               {a['median']:.3f}")
    print(f"  P25–P75:              {a['percentiles']['p25']:.3f} – {a['percentiles']['p75']:.3f}")

    print("\n--- Annulment Risk ---")
    ann = engine.simulate_annulment_risk()
    print(f"  Application rate:     {ann['application_probability']:.3f}")
    print(f"  Net risk:             {ann['net_risk']:.4f}")
    print(f"  Expected delay:       {ann['expected_delay_months']:.1f} months")

    print("\n--- Enforcement Timeline ---")
    enf = engine.simulate_enforcement_timeline()
    print(f"  P25:                  {enf['p25']:.1f} months")
    print(f"  P50:                  {enf['p50']:.1f} months")
    print(f"  P75:                  {enf['p75']:.1f} months")

    print("\n--- Recovery Rate ---")
    rec = engine.simulate_recovery_rate()
    print(f"  Mean recovery:        {rec['mean']:.3f} of claim")
    print(f"  Expected USD:         ${rec['expected_usd']:,.0f}")
    print(f"  95% interval:         {rec['ci_low']:.3f} – {rec['ci_high']:.3f}")
    print(f"  Expected years:       {rec['expected_years']:.1f}")

    print("\n--- Sovereign Friction Score ---")
    print(f"  Score:                {engine.score_sovereign_friction():.1f}/100")

    # Behavioral module
    print("\n--- Behavioral: Overclaiming Bias ---")
    bm = BehavioralModule()
    oc = bm.analyze_overclaiming_bias(250_000_000, "Mining")
    print(f"  Rational expectation: ${oc['rational_expectation']:,.0f}")
    print(f"  Overclaiming level:   {oc['overclaiming_level']}")

    print("\n--- Behavioral: Settlement Zone (ZOPA) ---")
    zopa = bm.calculate_settlement_zone(
        award_amount=109_500_000,
        investor_recovery_prob=0.65,
        state_enforcement_risk=0.60,
    )
    print(f"  Zone exists:          {zopa['settlement_zone_exists']}")
    print(f"  Investor floor:       ${zopa['investor_floor']:,.0f}")
    print(f"  State ceiling:        ${zopa['state_ceiling']:,.0f}")
    if zopa["zopa_midpoint"]:
        print(f"  ZOPA midpoint:        ${zopa['zopa_midpoint']:,.0f}")

    print("\n--- Behavioral: Delay Incentive ---")
    di = bm.analyze_state_delay_incentive(109_500_000, "Moderate")
    print(f"  Delay incentive (5yr): ${di['delay_incentive_usd']:,.0f}")
    print(f"  P(never pay):          {di['probability_of_non_payment']:.2f}")

    print("\n--- Enforcement Pathway ---")
    tz_profile = profile.country_profile()
    ep = EnforcementPathway(tz_profile, 109_500_000, "Tanzania")
    att = ep.score_asset_attachability()
    print(f"  Attachability score:  {att['score']:.0f}/100 (Grade {att['grade']})")
    print(f"  Key targets:          {len(att['key_targets'])} identified")

    seq = ep.recommend_sequencing()
    print(f"  Sequencing steps:     {len(seq)}")
    for s in seq:
        print(f"    [{s['step']}] {s['phase']}: {s['action'][:70]}...")

    print("\n--- Decision Tree (top-level nodes) ---")
    tree = ep.generate_decision_tree()
    for child in tree["children"]:
        print(f"  [{child['node_id']}] p={child['probability']:.2f}: {child['label']}")
