"""
ISDS Recovery Realism Engine — Memo Generator
==============================================
Generates advisory memo briefs from simulation results and dispute profiles.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Optional

from data_module import COUNTRY_PROFILES, ENFORCEMENT_JURISDICTIONS, CASES
from simulation_engine import (
    BehavioralModule,
    DisputeProfile,
    EnforcementPathway,
    SimulationEngine,
)


def _fmt_usd(amount: Optional[float], default: str = "N/A") -> str:
    """Format a float as a USD string."""
    if amount is None:
        return default
    if abs(amount) >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    return f"${amount:,.0f}"


def _pct(value: Optional[float], default: str = "N/A") -> str:
    """Format a float (0–1) as a percentage string."""
    if value is None:
        return default
    return f"{value:.1%}"


class MemoGenerator:
    """
    Generates structured advisory memo briefs from ISDS simulation output.

    Parameters
    ----------
    simulation_results : dict
        Full output from SimulationEngine.run_full_simulation().
    dispute_profile : DisputeProfile
        The input dispute profile used for the simulation.
    country_profile : dict
        The COUNTRY_PROFILES entry for the respondent state.
    enforcement_pathway : EnforcementPathway
        An EnforcementPathway instance initialised for this dispute.
    """

    def __init__(
        self,
        simulation_results: dict,
        dispute_profile: DisputeProfile,
        country_profile: dict,
        enforcement_pathway: EnforcementPathway,
    ) -> None:
        self.results = simulation_results
        self.dp = dispute_profile
        self.cp = country_profile
        self.ep = enforcement_pathway

        # Pre-extract common sub-dicts for readability
        self.summary = simulation_results.get("summary", {})
        self.jurisd = simulation_results.get("jurisdictional_success", {})
        self.award_r = simulation_results.get("award_to_claim", {})
        self.ann = simulation_results.get("annulment_risk", {})
        self.timeline = simulation_results.get("enforcement_timeline", {})
        self.recovery = simulation_results.get("recovery_rate", {})
        self.friction_score = simulation_results.get("sovereign_friction_score", 50.0)

        self._memo_date = datetime.utcnow().strftime("%d %B %Y")

    # ------------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------------

    def generate_executive_summary(self) -> str:
        """Generate a 3–4 paragraph executive summary of key findings."""

        j_prob = self.summary.get("jurisdictional_success_prob", 0.0)
        rec_mean = self.summary.get("expected_recovery_fraction", 0.0)
        rec_usd = self.summary.get("expected_recovery_usd", 0.0)
        years = self.summary.get("expected_years_to_recovery", 0.0)
        ann_risk = self.summary.get("annulment_net_risk", 0.0)
        friction = self.friction_score
        friction_level = self.cp.get("enforcement_friction_level", "Moderate")
        compliance = self.cp.get("voluntary_compliance_history", "Unknown")
        disc_lo, disc_hi = self.cp.get("settlement_discount_range", (0.10, 0.30))

        claimed = self.dp.amount_claimed_usd
        median_ratio = self.award_r.get("median", 0.38)
        expected_award = claimed * median_ratio

        para1 = (
            f"This advisory memorandum provides a quantitative assessment of the "
            f"recovery prospects for a {_fmt_usd(claimed)} investor-state claim against "
            f"{self.dp.respondent_state} arising from activities in the {self.dp.sector} "
            f"sector pursuant to a {self.dp.treaty_basis}. Our Monte Carlo simulation "
            f"({self.results.get('dispute_profile', {}).get('investment_type', 'investment dispute')}) "
            f"draws on historical ICSID/UNCTAD data calibrated to the specific governance, "
            f"enforcement, and behavioural characteristics of {self.dp.respondent_state}."
        )

        para2 = (
            f"The jurisdictional success probability—encompassing both upholding of "
            f"jurisdiction and a meritorious award—is estimated at {_pct(j_prob)}. "
            f"Conditional on success, the median award-to-claim ratio is {median_ratio:.0%}, "
            f"implying a median award of approximately {_fmt_usd(expected_award)}. "
            f"After applying the probability of enforcement success (friction level: "
            f"{friction_level}) and a modelled settlement discount of {disc_lo:.0%}–{disc_hi:.0%}, "
            f"the composite expected recovery rate is {_pct(rec_mean)} of the claimed amount, "
            f"or approximately {_fmt_usd(rec_usd)}. This recovery is expected to take "
            f"approximately {years:.1f} years from award to final satisfaction."
        )

        para3 = (
            f"The sovereign friction score for {self.dp.respondent_state} is "
            f"{friction:.0f}/100, reflecting its governance indicators, compliance history "
            f"({compliance}), and enforcement track record. The annulment net risk—the joint "
            f"probability that the state applies for and succeeds in annulling an adverse "
            f"award—is {_pct(ann_risk)}, which is below the global average. The primary "
            f"enforcement challenge is {friction_level.lower()} resistance to voluntary "
            f"compliance, necessitating multi-jurisdiction enforcement proceedings."
        )

        # SWF comment if applicable
        swf_name = self.cp.get("swf_name")
        swf_aum = self.cp.get("swf_aum_billions")
        if swf_name and swf_aum and swf_aum > 0.5:
            para4 = (
                f"A key strategic asset for enforcement is the respondent's sovereign wealth "
                f"fund, {swf_name} (AUM ~${swf_aum:.1f}B). Under the Paris Court of Appeal "
                f"(2019) LIA doctrine and the Swedish Supreme Court's Ascom (2021) precedent, "
                f"SWF assets managed for general commercial investment purposes are not immune "
                f"from attachment. Parallel enforcement filings in Paris and Stockholm targeting "
                f"these assets are recommended as a priority action alongside US discovery "
                f"proceedings."
            )
        else:
            att = self.ep.score_asset_attachability()
            targets = att.get("key_targets", [])
            target_str = (
                f"{len(targets)} SOE and commercial asset categories" if targets
                else "limited commercial assets"
            )
            para4 = (
                f"The primary enforcement targets are {target_str} of {self.dp.respondent_state}. "
                f"Asset attachability is rated {att.get('grade', 'C')} ({att.get('score', 0):.0f}/100). "
                f"The recommended enforcement sequencing begins with US discovery proceedings "
                f"under the NML Capital doctrine, followed by parallel recognition filings in "
                f"London and Paris to maximise settlement pressure."
            )

        return "\n\n".join([para1, para2, para3, para4])

    # ------------------------------------------------------------------
    # Full Memo
    # ------------------------------------------------------------------

    def generate_full_memo(self) -> str:
        """
        Generate a full structured advisory memo with all ten sections.

        Returns
        -------
        str — plain-text formatted memo suitable for display or download.
        """
        lines: list[str] = []

        def h1(title: str) -> None:
            lines.append("")
            lines.append("=" * 72)
            lines.append(title.upper())
            lines.append("=" * 72)

        def h2(title: str) -> None:
            lines.append("")
            lines.append(f"  {title}")
            lines.append("  " + "-" * (len(title) + 2))

        def para(text: str) -> None:
            lines.append(f"  {text}")

        def kv(key: str, value: str) -> None:
            lines.append(f"    {key:<42} {value}")

        # ── Header ───────────────────────────────────────────────────────
        lines.append("=" * 72)
        lines.append("PRIVILEGED AND CONFIDENTIAL — ATTORNEY–CLIENT COMMUNICATION")
        lines.append("ISDS RECOVERY REALISM ENGINE — ADVISORY MEMORANDUM")
        lines.append("=" * 72)
        lines.append(f"  Date:     {self._memo_date}")
        lines.append(f"  Re:       {self.dp.investor_nationality} Investor v. {self.dp.respondent_state}")
        lines.append(f"  Subject:  {_fmt_usd(self.dp.amount_claimed_usd)} ISDS Claim — "
                     f"Recovery Assessment ({self.dp.sector})")
        lines.append(f"  Prepared: ISDS Recovery Realism Engine v1.0 (Monte Carlo)")

        # ── Section I: Executive Summary ─────────────────────────────────
        h1("I. EXECUTIVE SUMMARY")
        exec_sum = self.generate_executive_summary()
        for p in exec_sum.split("\n\n"):
            para(p)
            lines.append("")

        # ── Section II: Dispute Profile ──────────────────────────────────
        h1("II. DISPUTE PROFILE")
        h2("2.1 Input Parameters")
        kv("Respondent State:", self.dp.respondent_state)
        kv("Investor Nationality:", self.dp.investor_nationality)
        kv("Sector:", self.dp.sector)
        kv("Treaty Basis:", self.dp.treaty_basis)
        kv("Amount Claimed (USD):", _fmt_usd(self.dp.amount_claimed_usd))
        kv("Investment Type:", self.dp.investment_type)
        bit_yr = self.dp.bit_year
        kv("BIT Year:", str(bit_yr) if bit_yr else "N/A")
        kv("ICSID Member:", str(self.cp.get("icsid_member", "Unknown")))
        kv("WGI Rule of Law (percentile):", f"{self.cp.get('wgi_rule_of_law', 'N/A')}")
        kv("WGI Corruption Control:", f"{self.cp.get('wgi_corruption', 'N/A')}")
        kv("WGI Govt. Effectiveness:", f"{self.cp.get('wgi_govt_effectiveness', 'N/A')}")
        kv("WJP Score:", str(self.cp.get("wjp_score", "N/A")))
        kv("Enforcement Friction Level:", self.cp.get("enforcement_friction_level", "N/A"))
        kv("Voluntary Compliance History:", self.cp.get("voluntary_compliance_history", "Unknown"))

        # ── Section III: Jurisdictional Analysis ─────────────────────────
        h1("III. JURISDICTIONAL ANALYSIS")
        j_prob = self.jurisd.get("probability", 0.0)
        ci = self.jurisd.get("confidence_interval", (0.0, 1.0))
        adjs = self.jurisd.get("adjustments", {})

        h2("3.1 Probability Assessment")
        kv("Jurisdictional Success Probability:", _pct(j_prob))
        kv("95% Confidence Interval:", f"{_pct(ci[0])} – {_pct(ci[1])}")
        kv("Base Rate (Global ICSID):", _pct(self.jurisd.get("base_rate_used")))

        h2("3.2 Adjustment Factors")
        kv("Friction Level Adjustment:", f"{adjs.get('friction_adj', 0)*100:+.1f}pp ({adjs.get('friction_level', '')})")
        kv("Sector Adjustment:", f"{adjs.get('sector_adj', 0)*100:+.1f}pp ({self.dp.sector})")
        kv("Treaty Basis Adjustment:", f"{adjs.get('treaty_adj', 0)*100:+.1f}pp ({self.dp.treaty_basis})")

        h2("3.3 Country Historical Blend")
        from data_module import calculate_historical_rates
        hist = calculate_historical_rates(self.dp.respondent_state)
        kv("Historical Investor Win Rate:", _pct(hist.get("investor_win_rate")))
        kv("Historical Cases (resolved):", str(hist.get("resolved_cases", 0)))
        kv("Historical Annulment Success Rate:", _pct(hist.get("annulment_success_rate")))

        # ── Section IV: Award Probability Assessment ──────────────────────
        h1("IV. AWARD PROBABILITY ASSESSMENT")
        median_ratio = self.award_r.get("median", 0.38)
        mean_ratio = self.award_r.get("mean", 0.38)
        pcts = self.award_r.get("percentiles", {})

        h2("4.1 Award-to-Claim Distribution")
        kv("Mean Award-to-Claim Ratio:", f"{mean_ratio:.1%}")
        kv("Median Award-to-Claim Ratio:", f"{median_ratio:.1%}")
        kv("P25 Award-to-Claim Ratio:", f"{pcts.get('p25', 0):.1%}")
        kv("P75 Award-to-Claim Ratio:", f"{pcts.get('p75', 0):.1%}")
        kv("P5 (downside) Ratio:", f"{pcts.get('p5', 0):.1%}")
        kv("P95 (upside) Ratio:", f"{pcts.get('p95', 0):.1%}")

        h2("4.2 Expected Award in USD")
        kv("Median Expected Award:", _fmt_usd(self.dp.amount_claimed_usd * median_ratio))
        kv("P25 Expected Award:", _fmt_usd(self.dp.amount_claimed_usd * pcts.get("p25", 0)))
        kv("P75 Expected Award:", _fmt_usd(self.dp.amount_claimed_usd * pcts.get("p75", 0)))

        para("")
        para(
            "Note: Award-to-claim ratios are sampled from the ICSID historical bracket "
            "distribution (2025 Statistics) with sector-specific scale adjustments applied. "
            "The distribution is inherently broad; a significant proportion (~21%) of "
            "investor-win awards fall below 10% of claimed amounts."
        )

        # ── Section V: Annulment Risk ─────────────────────────────────────
        h1("V. ANNULMENT RISK")
        h2("5.1 Key Metrics")
        kv("Annulment Application Probability:", _pct(self.ann.get("application_probability")))
        kv("Annulment Success Probability (given application):", _pct(self.ann.get("success_probability")))
        kv("Net Annulment Risk (P(applied) × P(success)):", _pct(self.ann.get("net_risk")))
        kv("Expected Annulment Delay (if applied):", f"{self.ann.get('expected_delay_months', 0):.0f} months")
        kv("P25 Delay:", f"{self.ann.get('p25_delay', 0):.0f} months")
        kv("P75 Delay:", f"{self.ann.get('p75_delay', 0):.0f} months")

        h2("5.2 Commentary")
        state = self.dp.respondent_state
        if state == "DRC":
            ann_note = (
                "The DRC is the only African state to have successfully annulled an ICSID award "
                "(Patrick Mitchell v. DRC, 2006). The annulment success probability is elevated "
                "to 12% accordingly."
            )
        elif state in ("Zimbabwe", "Egypt"):
            ann_note = (
                f"{state} has previously sought but failed in annulment proceedings. "
                "The annulment success probability is marginally below the global average "
                "at 4%, reflecting documented annulment committee scepticism."
            )
        else:
            ann_note = (
                "Annulment success rates globally are approximately 5% of all applied-for "
                "cases (BIICL 2021). States file annulments primarily as a delay tactic; "
                "the automatic stay of enforcement during proceedings is the principal benefit."
            )
        para(ann_note)

        # ── Section VI: Enforcement Landscape ────────────────────────────
        h1("VI. ENFORCEMENT LANDSCAPE")
        h2("6.1 Enforcement Timeline")
        kv("P25 (Optimistic):", self.timeline.get("scenario_labels", {}).get("optimistic", "N/A"))
        kv("P50 (Base Case):", self.timeline.get("scenario_labels", {}).get("base", "N/A"))
        kv("P75 (Pessimistic):", self.timeline.get("scenario_labels", {}).get("pessimistic", "N/A"))
        kv("Expected Years to Recovery:", f"{self.timeline.get('expected_years', 0):.1f} years")

        h2("6.2 Jurisdiction Mapping")
        jurisdictions = self.ep.map_jurisdictions()
        for i, jur in enumerate(jurisdictions, 1):
            kv(
                f"  [{i}] {jur['jurisdiction']}",
                f"P(success)={jur['success_probability']:.0%}, "
                f"Timeline={jur['timeline_months']}m, "
                f"Cost≈{_fmt_usd(jur['costs_estimate_usd'])}"
            )

        h2("6.3 Asset Attachability")
        att = self.ep.score_asset_attachability()
        kv("Asset Attachability Score:", f"{att['score']:.0f}/100 (Grade {att['grade']})")
        kv("SWF Score:", f"{att['component_scores']['swf_score']:.0f}/35")
        kv("SOE Score:", f"{att['component_scores']['soe_score']:.0f}/30")
        kv("ICSID Membership Score:", f"{att['component_scores']['icsid_score']:.0f}/15")
        kv("Compliance History Score:", f"{att['component_scores']['compliance_score']:.0f}/20")

        lines.append("")
        lines.append("    Key Target Assets:")
        for target in att.get("key_targets", []):
            lines.append(f"      • {target}")

        lines.append("")
        lines.append("    Immunity Risks:")
        for risk in att.get("immunity_risks", []):
            lines.append(f"      ⚠ {risk}")

        # ── Section VII: Behavioral Dynamics ─────────────────────────────
        h1("VII. BEHAVIORAL DYNAMICS")
        bm = BehavioralModule()

        h2("7.1 Overclaiming Bias Analysis")
        oc = bm.analyze_overclaiming_bias(
            self.dp.amount_claimed_usd,
            self.dp.sector,
        )
        kv("Claimed Amount:", _fmt_usd(oc["claimed_amount"]))
        kv("Rational Expectation:", _fmt_usd(oc["rational_expectation"]))
        kv("Overclaiming Factor:", f"{oc['overclaiming_factor']:.2f}×")
        kv("Overclaiming Level:", oc["overclaiming_level"])
        kv("Anchoring Premium:", _fmt_usd(oc["anchoring_premium"]))
        para("")
        para(oc["strategic_implication"])

        h2("7.2 State Delay Incentive (at Median Award)")
        median_award = self.dp.amount_claimed_usd * self.award_r.get("median", 0.38)
        friction_level = self.cp.get("enforcement_friction_level", "Moderate")
        di = bm.analyze_state_delay_incentive(median_award, friction_level)
        kv("Award Amount (Median):", _fmt_usd(di["award_amount"]))
        kv("NPV at Year 1:", _fmt_usd(di["npv_at_year_1"]))
        kv("NPV at Year 3:", _fmt_usd(di["npv_at_year_3"]))
        kv("NPV at Year 5:", _fmt_usd(di["npv_at_year_5"]))
        kv("NPV at Year 10:", _fmt_usd(di["npv_at_year_10"]))
        kv("5-Year Delay Incentive (USD):", _fmt_usd(di["delay_incentive_usd"]))
        kv("P(Never Pay):", _pct(di["probability_of_non_payment"]))
        kv("Optimal Delay Years:", f"{di['optimal_delay_years']:.0f} years")
        para("")
        para(di["summary"])

        # ── Section VIII: Settlement Analysis ────────────────────────────
        h1("VIII. SETTLEMENT ANALYSIS")
        j_prob = self.summary.get("jurisdictional_success_prob", 0.57)
        enf_risk = self.cp.get("enforcement_friction_level", "Moderate")
        enf_prob = {
            "Critical": 0.15, "Very High": 0.25, "High": 0.45,
            "Moderate": 0.65, "Moderate-Low": 0.80, "Low": 0.92
        }.get(enf_risk, 0.60)

        zopa = bm.calculate_settlement_zone(
            award_amount=median_award,
            investor_recovery_prob=j_prob * enf_prob,
            state_enforcement_risk=enf_prob,
        )

        h2("8.1 ZOPA Analysis")
        kv("Award Amount (Median):", _fmt_usd(zopa["award_amount"]))
        kv("Investor Floor (minimum acceptable):", _fmt_usd(zopa["investor_floor"]))
        kv("State Ceiling (maximum rational payment):", _fmt_usd(zopa["state_ceiling"]))
        kv("Settlement Zone Exists:", str(zopa["settlement_zone_exists"]))
        if zopa["zopa_midpoint"]:
            kv("ZOPA Midpoint:", _fmt_usd(zopa["zopa_midpoint"]))
            opt = zopa.get("optimal_settlement_range")
            if opt:
                kv("Optimal Settlement Range:", f"{_fmt_usd(opt[0])} – {_fmt_usd(opt[1])}")
        kv("Zone Width:", _fmt_usd(zopa["zone_width_usd"]))
        para("")
        para(zopa["commentary"])

        h2("8.2 Prospect Theory Dynamics")
        pt_inv = bm.prospect_theory_valuation(
            amount=zopa.get("zopa_midpoint") or median_award * 0.5,
            reference_point=median_award,
            loss_aversion=2.25,
        )
        para(f"INVESTOR perspective (reference = award of {_fmt_usd(median_award)}):")
        para(pt_inv["interpretation"])
        lines.append("")

        disc_lo, disc_hi = self.cp.get("settlement_discount_range", (0.10, 0.30))
        state_reference = median_award * (disc_lo + disc_hi) / 2
        pt_state = bm.prospect_theory_valuation(
            amount=zopa.get("zopa_midpoint") or median_award * 0.5,
            reference_point=state_reference,
            loss_aversion=2.25,
        )
        para(f"STATE perspective (reference = budgeted reserve of {_fmt_usd(state_reference)}):")
        para(pt_state["interpretation"])

        # ── Section IX: Recommended Strategy ─────────────────────────────
        h1("IX. RECOMMENDED STRATEGY")
        seq = self.ep.recommend_sequencing()
        for step in seq:
            priority_marker = {"Critical": "!!", "High": "! ", "Medium": "  ", "Low": "  "}.get(
                step.get("priority", "Medium"), "  "
            )
            lines.append(f"  [{priority_marker}] STEP {step['step']}: {step['phase']}")
            lines.append(f"       Priority: {step.get('priority', 'Medium')}")
            lines.append(f"       Action:   {step['action']}")
            lines.append(f"       Rationale:{step['rationale']}")
            lines.append("")

        # ── Section X: Risk Factors ───────────────────────────────────────
        h1("X. RISK FACTORS")
        h2("10.1 Enforcement Risks")
        immunity_risks = att.get("immunity_risks", [])
        for risk in immunity_risks:
            lines.append(f"    • {risk}")

        h2("10.2 Award Risk")
        lines.append(f"    • Overclaiming level: {oc['overclaiming_level']} — {oc['strategic_implication']}")
        lines.append(f"    • Annulment net risk: {_pct(self.ann.get('net_risk', 0.0))}")
        lines.append(f"    • Sector track record: {self.dp.sector} cases have mixed outcomes in Africa")

        h2("10.3 Political and Behavioural Risks")
        lines.append(f"    • Compliance history: {self.cp.get('voluntary_compliance_history', 'Unknown')}")
        lines.append(f"    • State delay incentive: {_fmt_usd(di.get('delay_incentive_usd'))} over 5 years")
        lines.append(f"    • Political risk — P(never pay): {_pct(di.get('probability_of_non_payment'))}")

        h2("10.4 Simulation Limitations")
        para(
            "This analysis is generated by a Monte Carlo simulation calibrated on historical "
            "ICSID/UNCTAD data. It does not constitute legal advice. Actual outcomes depend on "
            "specific factual and legal arguments, tribunal composition, and political developments "
            "beyond the scope of statistical modelling. Historical base rates for African ISDS cases "
            "are drawn from a relatively small database (n<50 resolved cases); confidence intervals "
            "are correspondingly wide."
        )

        # ── Footer ────────────────────────────────────────────────────────
        lines.append("")
        lines.append("=" * 72)
        lines.append("DATA SOURCES")
        lines.append("=" * 72)
        sources = [
            "ICSID Caseload Statistics 2025: https://icsid.worldbank.org/sites/default/files/publications/2025-1%20ENG%20-%20The%20ICSID%20Caseload%20Statistics%20(Issue%202025-1).pdf",
            "BIICL Empirical Study on Annulment (2021): https://www.biicl.org/documents/10899_annulment-in-icsid-arbitration190821.pdf",
            "Public Citizen \"The Scramble for Africa Continues\" (Dec 2024): https://www.citizen.org/article/the-scramble-for-africa-continues-impacts-of-investor-state-dispute-settlement-on-african-countries/",
            "TNI ISDS in Numbers (2019): https://www.tni.org/files/publication-downloads/isds_africa_web.pdf",
            "World Bank WGI DataBank 2023: https://databank.worldbank.org/embed/WGI-Table/id/ceea4d8b",
            "WJP Rule of Law Index 2023: https://worldjusticeproject.org/rule-of-law-index/downloads/WJPIndex2023.pdf",
        ]
        for src in sources:
            lines.append(f"  • {src}")

        lines.append("")
        lines.append("=" * 72)
        lines.append(f"Generated: {self._memo_date} | ISDS Recovery Realism Engine v1.0")
        lines.append("=" * 72)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def generate_csv_export(self, simulation_results: Optional[dict] = None) -> str:
        """
        Generate a CSV string of key simulation outputs.

        Parameters
        ----------
        simulation_results : dict, optional
            Full simulation results dict. Uses self.results if None.

        Returns
        -------
        str — CSV-formatted string.
        """
        if simulation_results is None:
            simulation_results = self.results

        summary = simulation_results.get("summary", {})
        jurisd = simulation_results.get("jurisdictional_success", {})
        award_r = simulation_results.get("award_to_claim", {})
        ann = simulation_results.get("annulment_risk", {})
        timeline = simulation_results.get("enforcement_timeline", {})
        recovery = simulation_results.get("recovery_rate", {})
        dp_dict = simulation_results.get("dispute_profile", {})
        pcts = recovery.get("percentiles", {})
        award_pcts = award_r.get("percentiles", {})

        rows = [
            # Dispute inputs
            ("DISPUTE PROFILE", ""),
            ("Respondent State", dp_dict.get("respondent_state", "")),
            ("Investor Nationality", dp_dict.get("investor_nationality", "")),
            ("Sector", dp_dict.get("sector", "")),
            ("Treaty Basis", dp_dict.get("treaty_basis", "")),
            ("Amount Claimed (USD)", dp_dict.get("amount_claimed_usd", "")),
            ("Investment Type", dp_dict.get("investment_type", "")),
            ("BIT Year", dp_dict.get("bit_year", "")),
            ("", ""),
            # Jurisdictional
            ("JURISDICTIONAL SUCCESS", ""),
            ("Probability", jurisd.get("probability", "")),
            ("95% CI Low", jurisd.get("confidence_interval", [None, None])[0]),
            ("95% CI High", jurisd.get("confidence_interval", [None, None])[1]),
            ("Base Rate Used", jurisd.get("base_rate_used", "")),
            ("", ""),
            # Award
            ("AWARD TO CLAIM RATIO", ""),
            ("Mean Ratio", award_r.get("mean", "")),
            ("Median Ratio", award_r.get("median", "")),
            ("Std Dev", award_r.get("std_dev", "")),
            ("P5 Ratio", award_pcts.get("p5", "")),
            ("P25 Ratio", award_pcts.get("p25", "")),
            ("P50 Ratio", award_pcts.get("p50", "")),
            ("P75 Ratio", award_pcts.get("p75", "")),
            ("P95 Ratio", award_pcts.get("p95", "")),
            ("", ""),
            # Annulment
            ("ANNULMENT RISK", ""),
            ("Application Probability", ann.get("application_probability", "")),
            ("Success Probability", ann.get("success_probability", "")),
            ("Net Risk", ann.get("net_risk", "")),
            ("Expected Delay (months)", ann.get("expected_delay_months", "")),
            ("P25 Delay (months)", ann.get("p25_delay", "")),
            ("P75 Delay (months)", ann.get("p75_delay", "")),
            ("", ""),
            # Timeline
            ("ENFORCEMENT TIMELINE", ""),
            ("P25 (months)", timeline.get("p25", "")),
            ("P50 (months)", timeline.get("p50", "")),
            ("P75 (months)", timeline.get("p75", "")),
            ("Mean (months)", timeline.get("mean", "")),
            ("Expected Years", timeline.get("expected_years", "")),
            ("", ""),
            # Recovery
            ("RECOVERY RATE (fraction of claim)", ""),
            ("Mean Recovery", recovery.get("mean", "")),
            ("Median Recovery", recovery.get("median", "")),
            ("CI Low (5th pct)", recovery.get("ci_low", "")),
            ("CI High (95th pct)", recovery.get("ci_high", "")),
            ("P5", pcts.get("p5", "")),
            ("P10", pcts.get("p10", "")),
            ("P25", pcts.get("p25", "")),
            ("P50", pcts.get("p50", "")),
            ("P75", pcts.get("p75", "")),
            ("P90", pcts.get("p90", "")),
            ("P95", pcts.get("p95", "")),
            ("Expected USD Recovery", recovery.get("expected_usd", "")),
            ("Expected Years to Recovery", recovery.get("expected_years", "")),
            ("", ""),
            # Summary
            ("SUMMARY", ""),
            ("Expected Recovery Fraction", summary.get("expected_recovery_fraction", "")),
            ("Expected Recovery USD", summary.get("expected_recovery_usd", "")),
            ("Expected Years to Recovery", summary.get("expected_years_to_recovery", "")),
            ("Jurisdictional Success Prob", summary.get("jurisdictional_success_prob", "")),
            ("Median Award-to-Claim", summary.get("median_award_to_claim", "")),
            ("Annulment Net Risk", summary.get("annulment_net_risk", "")),
            ("Enforcement Prob", summary.get("enforcement_prob", "")),
            ("Sovereign Friction Score", summary.get("sovereign_friction_score", "")),
        ]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "Value"])
        for row in rows:
            writer.writerow(row)

        return output.getvalue()
