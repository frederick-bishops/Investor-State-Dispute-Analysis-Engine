"""
ISDS Recovery Realism Engine — Data Module
==========================================
Structured case database and reference data for African investor-state
dispute settlement cases. All data sourced from ICSID, UNCTAD Investment
Policy Hub, italaw.com, Public Citizen (Dec 2024), TNI (2019), BIICL (2021),
and World Bank WGI / WJP Rule of Law Index (2023).

Sources:
- ICSID Africa Special Focus Report (2017): https://icsid.worldbank.org/sites/default/files/publications/Caseload%20Statistics/en/Special%20Issues/ICSID%20Web%20Stats%20Africa%20(English)%20June%202017.pdf
- Public Citizen "The Scramble for Africa Continues" (Dec 2024): https://www.citizen.org/article/the-scramble-for-africa-continues-impacts-of-investor-state-dispute-settlement-on-african-countries/
- TNI ISDS in Numbers (2019): https://www.tni.org/files/publication-downloads/isds_africa_web.pdf
- BIICL Empirical Study on Annulment (2021): https://www.biicl.org/documents/10899_annulment-in-icsid-arbitration190821.pdf
- ICSID Caseload Statistics 2025: https://icsid.worldbank.org/sites/default/files/publications/2025-1%20ENG%20-%20The%20ICSID%20Caseload%20Statistics%20(Issue%202025-1).pdf
- World Bank WGI DataBank 2023: https://databank.worldbank.org/embed/WGI-Table/id/ceea4d8b
- WJP Rule of Law Index 2023: https://worldjusticeproject.org/rule-of-law-index/downloads/WJPIndex2023.pdf
- G.O. Sodipo & Co (2024): https://gos-law.com/50-years-of-icsid-arbitration-and-african-states-1974-2024/
"""

from __future__ import annotations
from typing import Optional, Union
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. CASE DATABASE
# ---------------------------------------------------------------------------
# Outcome values: "Investor Win", "State Win", "Settled", "Pending",
#                 "Discontinued", "Annulled"
# annulment_outcome values: "Annulment granted", "Annulment rejected",
#                           "Partial annulment", "Pending", None
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    # ── GHANA ──────────────────────────────────────────────────────────────
    {
        "case_name": "Vacuum Salt Products Ltd. v. Republic of Ghana",
        "respondent_state": "Ghana",
        "investor_nationality": "United Kingdom",
        "sector": "Manufacturing",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 1992,
        "year_decided": 1994,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed; no award to enforce",
        "notes": (
            "ICSID ARB/92/1. Tribunal declined jurisdiction (no investment under "
            "the ICSID Convention). First case testing ICSID jurisdiction via "
            "investment contract in Ghana."
        ),
    },
    {
        "case_name": "Telekom Malaysia Berhad v. Republic of Ghana",
        "respondent_state": "Ghana",
        "investor_nationality": "Malaysia",
        "sector": "Telecommunications",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2003,
        "year_decided": 2004,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — terms not publicly disclosed",
        "notes": (
            "Malaysia–Ghana BIT. Telecom dispute settled in 2004. "
            "Settlement terms remain confidential."
        ),
    },
    {
        "case_name": "Gustav F W Hamester GmbH & Co KG v. Republic of Ghana",
        "respondent_state": "Ghana",
        "investor_nationality": "Germany",
        "sector": "Manufacturing",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 142_700_000.0,   # ~€100M converted
        "amount_awarded_usd": None,
        "year_filed": 2007,
        "year_decided": 2010,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed; all claims dismissed",
        "notes": (
            "ICSID ARB/07/24. Germany–Ghana BIT (1995). German investor claimed "
            "Ghana Cocoa Board (GCB) expropriated cocoa processing JV in Takoradi. "
            "Tribunal dismissed all claims, finding GCB acts not attributable to the "
            "state. Leading case on attribution and state-owned enterprises."
        ),
    },
    {
        "case_name": "Everyway v. Republic of Ghana",
        "respondent_state": "Ghana",
        "investor_nationality": "China",
        "sector": "Infrastructure",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2021,
        "year_decided": 2023,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed",
        "notes": (
            "Infrastructure/traffic management dispute. Tribunal decided in favour "
            "of Ghana. Details remain limited in public record."
        ),
    },
    {
        "case_name": "BGHL (Blue Gold Holdings Ltd.) v. Republic of Ghana",
        "respondent_state": "Ghana",
        "investor_nationality": "United Kingdom",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2025,
        "year_decided": None,
        "outcome": "Pending",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Pending — proceedings ongoing",
        "notes": (
            "Ghana–UK BIT. Blue Gold Holdings filed claim after Ghana's Minerals "
            "Commission issued notice of termination of mining leases for the Bogoso "
            "Prestea Mine in 2024 and an interim management committee assumed control."
        ),
    },

    # ── NIGERIA ────────────────────────────────────────────────────────────
    {
        "case_name": "Process & Industrial Developments Ltd. v. Federal Republic of Nigeria",
        "respondent_state": "Nigeria",
        "investor_nationality": "Ireland",
        "sector": "Oil & Gas",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": 1_500_000_000.0,
        "amount_awarded_usd": 6_600_000_000.0,
        "year_filed": 2012,
        "year_decided": 2017,
        "outcome": "Annulled",
        "annulment_attempted": True,
        "annulment_outcome": "Annulment granted",
        "enforcement_status": (
            "Award set aside by UK High Court (Oct 2023): P&ID found to have "
            "procured contract by bribery and to have given perjured evidence. "
            "One of the most significant fraud-based annulments in arbitral history."
        ),
        "notes": (
            "Ad hoc arbitration under English law, London seat. Gas Supply & "
            "Processing Agreement (GSPA). 2017 award of $6.6B grew to >$11B with "
            "7% post-judgment interest by 2023. UK High Court set aside on fraud "
            "grounds Oct 2023."
        ),
    },
    {
        "case_name": "Shell Petroleum N.V. v. Federal Republic of Nigeria (I)",
        "respondent_state": "Nigeria",
        "investor_nationality": "Netherlands",
        "sector": "Oil & Gas",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2004,
        "year_decided": 2006,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — terms not publicly disclosed",
        "notes": (
            "ICSID. Netherlands–Nigeria BIT. Related to Ogoni oil spill disputes. "
            "Settled 2006."
        ),
    },
    {
        "case_name": "Shell Petroleum N.V. v. Federal Republic of Nigeria (II)",
        "respondent_state": "Nigeria",
        "investor_nationality": "Netherlands",
        "sector": "Oil & Gas",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2021,
        "year_decided": 2022,
        "outcome": "Discontinued",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Discontinued — no award issued",
        "notes": (
            "ICSID ARB/21/7. Filed Feb 10, 2021 by Shell Petroleum N.V. and Shell "
            "Petroleum Development Company of Nigeria under Netherlands–Nigeria BIT "
            "over community compensation order for oil pollution. Discontinued "
            "Oct 13, 2022, reportedly linked to Shell's exit from Nigerian onshore "
            "operations."
        ),
    },
    {
        "case_name": "Esso Exploration & Production Nigeria Ltd. / Shell v. NNPC",
        "respondent_state": "Nigeria",
        "investor_nationality": "United States",
        "sector": "Oil & Gas",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2009,
        "year_decided": 2011,
        "outcome": "Investor Win",
        "annulment_attempted": True,
        "annulment_outcome": "Partial annulment",
        "enforcement_status": (
            "Partially set aside by Nigerian courts; US Second Circuit ruled 2022 "
            "on partial enforcement."
        ),
        "notes": (
            "NNPC arbitration (contract-based). ExxonMobil/Shell joint venture claim "
            "against Nigerian National Petroleum Corporation. Award for Esso; "
            "Nigerian courts partially set aside; US Second Circuit ruled on partial "
            "enforcement in 2022."
        ),
    },

    # ── CÔTE D'IVOIRE ──────────────────────────────────────────────────────
    {
        "case_name": "Compagnie Française pour le Développement des Fibres Textiles (CFDT) v. Republic of Côte d'Ivoire",
        "respondent_state": "Côte d'Ivoire",
        "investor_nationality": "France",
        "sector": "Agriculture",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 1997,
        "year_decided": 2000,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — terms not publicly disclosed",
        "notes": (
            "ICSID ARB/97/8. Agriculture/textiles sector investment contract dispute. "
            "Settled circa 2000."
        ),
    },
    {
        "case_name": "Société Resort Company Invest Abidjan v. Republic of Côte d'Ivoire",
        "respondent_state": "Côte d'Ivoire",
        "investor_nationality": "France",
        "sector": "Hospitality",
        "treaty_basis": "Investment Law",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2016,
        "year_decided": 2019,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed on merits",
        "notes": (
            "ICSID ARB/16/11. 2012 Ivorian Investment Code. Resort demolition dispute. "
            "Jurisdiction upheld but claims dismissed. Tribunal recommended Côte "
            "d'Ivoire reform its arbitration consent language — Côte d'Ivoire enacted "
            "new Investment Code in 2018 removing direct ICSID consent, rerouting "
            "to OHADA/CCJA arbitration."
        ),
    },

    # ── TANZANIA ───────────────────────────────────────────────────────────
    {
        "case_name": "Biwater Gauff (Tanzania) Ltd. v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "United Kingdom",
        "sector": "Infrastructure",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 20_000_000.0,
        "amount_awarded_usd": 0.0,
        "year_filed": 2005,
        "year_decided": 2008,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "No award to enforce. Liability found but zero damages: investor's "
            "own fiscal mismanagement pre-dated BIT breaches."
        ),
        "notes": (
            "ICSID ARB/05/22. UK–Tanzania BIT. Water infrastructure concession. "
            "Anglo-German consortium. Tribunal found Tanzania breached BIT "
            "(expropriation, FET, full protection) but awarded NO damages due to "
            "investor's pre-existing mismanagement. Landmark causation precedent."
        ),
    },
    {
        "case_name": "Standard Chartered Bank v. United Republic of Tanzania (I)",
        "respondent_state": "Tanzania",
        "investor_nationality": "United Kingdom",
        "sector": "Energy",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 148_400_000.0,
        "year_filed": 2010,
        "year_decided": 2012,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "Award issued; Tanzania's High Court issued injunction against "
            "enforcement — potential ICSID Convention breach. Partially resolved "
            "through post-award negotiation."
        ),
        "notes": (
            "ICSID ARB/10/12. Implementation Agreement (power sector finance). "
            "Award of ~$148.4M for TANESCO arrears unpaid to SCB as security agent. "
            "Tanzania's domestic court issued injunction against operationalising "
            "ICSID decision — direct tension with Art. 54 ICSID Convention."
        ),
    },
    {
        "case_name": "Standard Chartered Bank (Hong Kong) Ltd. v. United Republic of Tanzania (II)",
        "respondent_state": "Tanzania",
        "investor_nationality": "United Kingdom",
        "sector": "Energy",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2015,
        "year_decided": 2019,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Award issued Oct 11, 2019; enforcement status not fully public",
        "notes": (
            "ICSID ARB/15/41. Companion case to SCB/TANESCO dispute. Compensation "
            "for breach of Implementation Agreement. Award issued Oct 11, 2019."
        ),
    },
    {
        "case_name": "EcoEnergy (EcoDevelopment) v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "Sweden",
        "sector": "Agriculture",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2017,
        "year_decided": 2022,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Award issued Apr 13, 2022; enforcement negotiations ongoing",
        "notes": (
            "ICSID ARB/17/33. Sweden–Tanzania BIT (1999). 20,000-hectare sugarcane "
            "and ethanol bioenergy project cancelled by government. Award for "
            "investor Apr 13, 2022."
        ),
    },
    {
        "case_name": "Winshear Gold Corp v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "Canada",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 250_000_000.0,
        "amount_awarded_usd": 30_000_000.0,
        "year_filed": 2020,
        "year_decided": 2022,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — C$30M paid; approximately 82% of implied award",
        "notes": (
            "Canada–Tanzania BIT. SMP Gold Project. Tanzania paid C$30M settlement. "
            "~82% recovery on settlement vs. award. Demonstrates Tanzania's pragmatic "
            "post-award settlement posture."
        ),
    },
    {
        "case_name": "Indiana Resources (Nachingwea UK Ltd.) v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "United Kingdom",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 109_500_000.0,
        "year_filed": 2020,
        "year_decided": 2023,
        "outcome": "Settled",
        "annulment_attempted": True,
        "annulment_outcome": "Pending",
        "enforcement_status": (
            "Settled — Tanzania agreed to pay $90M (82.5% of award) in July 2024 "
            "after annulment proceedings began. Paid in full by early 2025."
        ),
        "notes": (
            "ICSID ARB/20/38. UK–Tanzania BIT. Tanzania unlawfully expropriated "
            "Ntaka Hill nickel project Jan 10, 2018 by cancelling retention licences "
            "under 2017 Mining Act. ICSID awarded $109.5M (Jul 2023). Settled for "
            "$90M in July 2024 after annulment proceedings initiated."
        ),
    },
    {
        "case_name": "Montero Mining & Exploration Ltd. v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "Canada",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 27_000_000.0,
        "year_filed": 2021,
        "year_decided": 2024,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — $27M paid in full (Mar 2025). Wigu Hill rare earth project.",
        "notes": (
            "Canada–Tanzania BIT. Wigu Hill rare earth retention licence cancellation. "
            "Settled Nov 2024; $27M paid Mar 2025. Tanzania's third rapid "
            "post-award settlement."
        ),
    },
    {
        "case_name": "Orca Energy Group Inc. v. United Republic of Tanzania",
        "respondent_state": "Tanzania",
        "investor_nationality": "Canada",
        "sector": "Oil & Gas",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 1_200_000_000.0,
        "amount_awarded_usd": None,
        "year_filed": 2024,
        "year_decided": None,
        "outcome": "Pending",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Pending — proceedings ongoing",
        "notes": (
            "Mauritius–Tanzania BIT plus PSA/GA. Claims re failure to extend "
            "development licence. $1.2 billion claimed. Filed 2024."
        ),
    },

    # ── KENYA ──────────────────────────────────────────────────────────────
    {
        "case_name": "World Duty Free Company Ltd. v. Republic of Kenya",
        "respondent_state": "Kenya",
        "investor_nationality": "United Kingdom",
        "sector": "Retail",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": 500_000_000.0,
        "amount_awarded_usd": None,
        "year_filed": 2000,
        "year_decided": 2006,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "N/A — Claims dismissed entirely because investment procured by bribery "
            "($2M to President Moi); contract void ab initio."
        ),
        "notes": (
            "ICSID ARB/00/7. 1989 Investment Agreement. Duty-free retail concession "
            "at Nairobi and Mombasa airports. Landmark case establishing that "
            "investments procured through bribery of a head of state receive no "
            "treaty protection. Tribunal declared contract void ab initio."
        ),
    },
    {
        "case_name": "Cortec Mining Kenya Ltd., Cortec (Pty) Ltd. and Stirling Capital Ltd. v. Republic of Kenya",
        "respondent_state": "Kenya",
        "investor_nationality": "United Kingdom",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": -3_226_429.21,   # Negative: costs awarded AGAINST investor
        "year_filed": 2015,
        "year_decided": 2018,
        "outcome": "State Win",
        "annulment_attempted": True,
        "annulment_outcome": "Annulment rejected",
        "enforcement_status": (
            "Kenya awarded $3.226M + $322K costs against Cortec. Annulment rejected "
            "Mar 2021. Enforcement of cost award by Kenya against Cortec."
        ),
        "notes": (
            "ICSID ARB/15/29. UK–Kenya BIT. Mrima Hill niobium/rare earths mining "
            "licence. Tribunal found Special Mining Licence 351 void for failure to "
            "conduct mandatory Environmental Impact Assessment (EIA). Legality "
            "requirement implied even absent explicit BIT language. Rare case of "
            "significant costs awarded to respondent state."
        ),
    },

    # ── DRC ────────────────────────────────────────────────────────────────
    {
        "case_name": "American Manufacturing & Trading Inc. (AMT) v. Republic of Zaire (DRC)",
        "respondent_state": "DRC",
        "investor_nationality": "United States",
        "sector": "Manufacturing",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 21_600_000.0,
        "amount_awarded_usd": 9_000_000.0,
        "year_filed": 1993,
        "year_decided": 1997,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "Award issued Feb 21, 1997. DRC largely non-compliant. Limited "
            "enforcement due to absence of identifiable commercial assets."
        ),
        "notes": (
            "ICSID ARB/93/1. USA–Zaire BIT (1984). Battery manufacturing facility "
            "(SINZA) looted by Zairian armed forces. First major ICSID award against "
            "an African state. Claimed ~$21.6M; awarded $9M + 7.5% interest. "
            "Award reflects 'realistic' valuation in precarious investment climate."
        ),
    },
    {
        "case_name": "Banro American Resources Inc. v. Democratic Republic of Congo",
        "respondent_state": "DRC",
        "investor_nationality": "United States",
        "sector": "Mining",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 1998,
        "year_decided": 2000,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — Jurisdiction declined",
        "notes": (
            "ICSID ARB/98/7. Mining convention/investment contract. Gold mining in "
            "Kivu/Maniema. Jurisdiction declined Sept 1, 2000 — Canadian parent Banro "
            "not an ICSID member at time; US subsidiary lacked valid consent. "
            "Illustrates the importance of structuring investment nationality."
        ),
    },
    {
        "case_name": "Patrick Mitchell v. Democratic Republic of Congo",
        "respondent_state": "DRC",
        "investor_nationality": "United States",
        "sector": "Legal Services",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 1999,
        "year_decided": 2006,
        "outcome": "Annulled",
        "annulment_attempted": True,
        "annulment_outcome": "Annulment granted",
        "enforcement_status": (
            "Original award (Feb 9, 2004) for Mitchell; ANNULLED Nov 1, 2006. "
            "DRC became first African state to successfully annul an ICSID award."
        ),
        "notes": (
            "ICSID ARB/99/7. USA–DRC BIT. US attorney's law practice in DRC had "
            "premises sealed by military court orders. Merits award for Mitchell "
            "(2004) annulled (2006) on ground that tribunal manifestly exceeded its "
            "powers by misidentifying 'investment' under the ICSID Convention — "
            "a law practice was found not to constitute an 'investment'. Leading "
            "annulment precedent; DRC is only African state to successfully annul."
        ),
    },

    # ── GUINEA ─────────────────────────────────────────────────────────────
    {
        "case_name": "Atlantic Triton Company Ltd. v. People's Revolutionary Republic of Guinea",
        "respondent_state": "Guinea",
        "investor_nationality": "France",
        "sector": "Maritime",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 1984,
        "year_decided": 1986,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Award for investor; enforcement details not public",
        "notes": "ICSID ARB/84/1. Maritime/shipping services investment contract dispute.",
    },
    {
        "case_name": "BSG Resources (Guinea) Limited v. Republic of Guinea",
        "respondent_state": "Guinea",
        "investor_nationality": "United Kingdom",
        "sector": "Mining",
        "treaty_basis": "Investment Law",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2014,
        "year_decided": 2024,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "N/A — Claims declared inadmissible (2024). BSGR ordered to pay 80% "
            "of Guinea's legal costs including success fees."
        ),
        "notes": (
            "ICSID ARB/14/22. Guinean Investment Law (2011 Investment Code). Simandou "
            "and Zogota iron ore concessions (est. 2bn tonnes). Guinea revoked in 2014 "
            "after anti-corruption review. Tribunal found by 'reasonable certainty' "
            "that licences procured through corrupt payments (~$9.42M) to intermediary "
            "connected to former President Condé's wife. Claims inadmissible. "
            "BSGR controlled by Beny Steinmetz."
        ),
    },
    {
        "case_name": "Axis International Holding DMCC v. Republic of Guinea",
        "respondent_state": "Guinea",
        "investor_nationality": "United Arab Emirates",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 28_900_000_000.0,
        "amount_awarded_usd": None,
        "year_filed": 2026,
        "year_decided": None,
        "outcome": "Pending",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Pending — proceedings ongoing (Jan 2026 filing)",
        "notes": (
            "Guinea–UAE BIT (2011) + Guinea Investment Code (1995). Boffa bauxite "
            "mining licence terminated May 2025. $28.9 billion claimed — largest ICSID "
            "claim ever against a West African state. Filed Jan 2026."
        ),
    },
    {
        "case_name": "Getma International and others v. Republic of Guinea",
        "respondent_state": "Guinea",
        "investor_nationality": "France",
        "sector": "Infrastructure",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 508_221.0,
        "year_filed": 2011,
        "year_decided": 2016,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Minimal award; enforcement reportedly not pursued vigorously",
        "notes": (
            "ICSID ARB/11/29. Investment Contract + Guinean Investment Code. Conakry "
            "Port concession. Guinea violated agreement; tribunal awarded only actual "
            "costs ($508K) despite hundreds of millions claimed. Illustrates "
            "pyrrhic investor wins in port/infrastructure sector."
        ),
    },

    # ── ZIMBABWE ───────────────────────────────────────────────────────────
    {
        "case_name": "Bernardus Henricus Funnekotter and others v. Republic of Zimbabwe",
        "respondent_state": "Zimbabwe",
        "investor_nationality": "Netherlands",
        "sector": "Agriculture",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2005,
        "year_decided": 2009,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": (
            "Zimbabwe refused voluntary payment. Dutch investors sought US attachment "
            "via SDNY (confirmed 2011). Zimbabwe refused to appear. Insufficient "
            "unprotected commercial assets identified. Non-compliant as of 2025."
        ),
        "notes": (
            "ICSID ARB/05/6. Netherlands–Zimbabwe BIT. Agricultural land expropriated "
            "under Zimbabwe's land reform programme. ~€450K–930K per farm plus "
            "10% compound interest. Zimbabwe non-compliant."
        ),
    },
    {
        "case_name": "Bernhard von Pezold and others v. Republic of Zimbabwe",
        "respondent_state": "Zimbabwe",
        "investor_nationality": "Germany",
        "sector": "Agriculture",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 195_000_000.0,
        "year_filed": 2010,
        "year_decided": 2015,
        "outcome": "Investor Win",
        "annulment_attempted": True,
        "annulment_outcome": "Annulment rejected",
        "enforcement_status": (
            "Non-compliant. Annulment dismissed Nov 21, 2018. Multi-jurisdiction "
            "enforcement: US DC Circuit upheld (2024); Malaysian High Court recognized "
            "(2023); UK litigation ongoing. Zimbabwe pledged compliance post-annulment "
            "then defaulted on pledge. Awards substantially unpaid as of 2025."
        ),
        "notes": (
            "ICSID ARB/10/15. Germany–Zimbabwe BIT (1995) + Switzerland–Zimbabwe BIT. "
            "Farm expropriation under Land Reform. First ICSID case to explicitly find "
            "discriminatory expropriation based on race (white European landowners). "
            "Tribunal ordered restitution OR $195M+ cash. Zimbabwe's annulment "
            "application (Oct 2015) rejected Nov 2018. Total both cases ~$230M."
        ),
    },
    {
        "case_name": "Border Timbers Ltd. and others v. Republic of Zimbabwe",
        "respondent_state": "Zimbabwe",
        "investor_nationality": "Switzerland",
        "sector": "Agriculture",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 35_000_000.0,
        "year_filed": 2010,
        "year_decided": 2015,
        "outcome": "Investor Win",
        "annulment_attempted": True,
        "annulment_outcome": "Annulment rejected",
        "enforcement_status": (
            "Non-compliant. Annulment dismissed Nov 21, 2018. Complex UK High Court "
            "and Court of Appeal immunity litigation 2020–2025. UK Court of Appeal "
            "(2025) rejected Zimbabwe's immunity arguments for recognition. Execution "
            "enforcement ongoing."
        ),
        "notes": (
            "ICSID ARB/10/25. Switzerland–Zimbabwe BIT. Forestry/timber operations; "
            "farm expropriation. Companion case to von Pezold. Separate award (~$35M); "
            "total both cases ~$230M. UK SIA litigation established principle: "
            "sovereign immunity available only at execution stage, not recognition."
        ),
    },

    # ── EGYPT ──────────────────────────────────────────────────────────────
    {
        "case_name": "Unión Fenosa Gas S.A. v. Arab Republic of Egypt",
        "respondent_state": "Egypt",
        "investor_nationality": "Spain",
        "sector": "Oil & Gas",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 2_013_000_000.0,
        "year_filed": 2014,
        "year_decided": 2018,
        "outcome": "Investor Win",
        "annulment_attempted": True,
        "annulment_outcome": "Pending",
        "enforcement_status": (
            "Egypt pursuing ICSID annulment; enforcement stayed pending annulment. "
            "Largest-ever ICSID award against an African state. Complex non-monetary "
            "settlement (LNG plant resumption + corporate restructuring) reportedly "
            "partially resolved; full cash payment not confirmed."
        ),
        "notes": (
            "ICSID ARB/14/4. Spain–Egypt BIT. LNG plant and gas supply agreement. "
            "Award Aug 31, 2018: $2.013 billion. Largest award ever against any "
            "African state. Egypt pursuing annulment. Settlement involving resumption "
            "of LNG plant production and corporate restructuring reported."
        ),
    },
    {
        "case_name": "Veolia Propreté v. Arab Republic of Egypt",
        "respondent_state": "Egypt",
        "investor_nationality": "France",
        "sector": "Infrastructure",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 110_000_000.0,
        "amount_awarded_usd": None,
        "year_filed": 2012,
        "year_decided": 2018,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed; FET claim dismissed",
        "notes": (
            "France–Egypt BIT. Waste services and transport concession. Egypt raised "
            "minimum wage post-Arab Spring revolution; Veolia claimed FET violation. "
            "Tribunal dismissed; legitimate regulatory measure. Landmark case on "
            "states' right to regulate."
        ),
    },

    # ── LIBYA ──────────────────────────────────────────────────────────────
    {
        "case_name": "Libyan American Oil Company (LIAMCO) v. Libyan Arab Republic",
        "respondent_state": "Libya",
        "investor_nationality": "United States",
        "sector": "Oil & Gas",
        "treaty_basis": "Investment Contract",
        "amount_claimed_usd": None,
        "amount_awarded_usd": 66_000_000.0,
        "year_filed": 1974,
        "year_decided": 1977,
        "outcome": "Investor Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Award for investor; partial enforcement obtained via ICC; pre-ICSID era case",
        "notes": (
            "ICC arbitration (1977). Oil concession nationalization. ~$66M awarded. "
            "Pre-dates modern BIT era. Landmark case in oil concession nationalization "
            "jurisprudence."
        ),
    },

    # ── SENEGAL ────────────────────────────────────────────────────────────
    {
        "case_name": "Millicom International Operations B.V. and Sentel GSM SA v. Republic of Senegal",
        "respondent_state": "Senegal",
        "investor_nationality": "Netherlands",
        "sector": "Telecommunications",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2008,
        "year_decided": 2012,
        "outcome": "Settled",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "Settled — award embodying settlement (Nov 27, 2012); non-pecuniary relief",
        "notes": (
            "ICSID ARB/08/20. Netherlands–Senegal BIT. Dutch telecom Millicom claimed "
            "expropriation after Senegal revoked 20-year GSM concession granted to "
            "subsidiary Sentel. Settled Nov 2012 with non-pecuniary relief "
            "(licence effectively restored/compensated)."
        ),
    },

    # ── GABON ──────────────────────────────────────────────────────────────
    {
        "case_name": "Navodaya Investment Ltd. v. Gabonese Republic",
        "respondent_state": "Gabon",
        "investor_nationality": "United Arab Emirates",
        "sector": "Mining",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": None,
        "amount_awarded_usd": None,
        "year_filed": 2018,
        "year_decided": 2022,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed",
        "notes": (
            "UAE–Gabon BIT. Okondja manganese mining project dispute. Tribunal "
            "decided in favour of Gabon in 2022."
        ),
    },

    # ── MOZAMBIQUE ─────────────────────────────────────────────────────────
    {
        "case_name": "CMC Muratori & Cementisti CMC di Ravenna v. Republic of Mozambique",
        "respondent_state": "Mozambique",
        "investor_nationality": "Italy",
        "sector": "Infrastructure",
        "treaty_basis": "Bilateral Investment Treaty",
        "amount_claimed_usd": 8_200_000.0,
        "amount_awarded_usd": None,
        "year_filed": 2017,
        "year_decided": 2019,
        "outcome": "State Win",
        "annulment_attempted": False,
        "annulment_outcome": None,
        "enforcement_status": "N/A — State prevailed; tribunal found no binding settlement existed",
        "notes": (
            "ICSID ARB/17/23. Italy–Mozambique BIT. Highway reconstruction contract. "
            ">€8.2M claimed. Tribunal found no binding settlement existed and "
            "dismissed all claims (Oct 2019)."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. COUNTRY PROFILES
# ---------------------------------------------------------------------------
# WGI values: 0–100 percentile ranks (World Bank WGI DataBank, 2023 data)
# WJP scores: 0–1 scale (WJP Rule of Law Index 2023)
# enforcement_friction_level: Critical / Very High / High / Moderate /
#                              Moderate-Low / Low
# voluntary_compliance_history: Yes / Partial / No / Unknown
# settlement_discount_range: (min_fraction, max_fraction) — e.g. (0.0, 0.15)
#   means discount of 0%–15% off face value of award
# avg_settlement_years: expected years from award to final settlement
# ---------------------------------------------------------------------------

COUNTRY_PROFILES: dict[str, dict] = {
    "Zimbabwe": {
        "wgi_rule_of_law": 13.0,
        "wgi_corruption": 14.0,
        "wgi_govt_effectiveness": 12.0,
        "wjp_score": 0.40,
        "enforcement_friction_level": "Very High",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["ZESA Holdings (electricity)", "Air Zimbabwe", "NetOne (telecom)", "ZIMRA"],
        "voluntary_compliance_history": "No",
        "settlement_discount_range": (0.40, 0.70),
        "avg_settlement_years": 12.0,
    },
    "Libya": {
        "wgi_rule_of_law": 7.0,
        "wgi_corruption": 8.0,
        "wgi_govt_effectiveness": 10.0,
        "wjp_score": None,          # Not measured by WJP
        "enforcement_friction_level": "Very High",
        "icsid_member": True,
        "swf_name": "Libyan Investment Authority (LIA)",
        "swf_aum_billions": 68.35,  # Africa's largest SWF
        "major_soes": ["National Oil Corporation (NOC)", "Libyan Investment Authority"],
        "voluntary_compliance_history": "No",
        "settlement_discount_range": (0.40, 0.70),
        "avg_settlement_years": 15.0,
    },
    "DRC": {
        "wgi_rule_of_law": 6.0,
        "wgi_corruption": 6.0,
        "wgi_govt_effectiveness": 8.0,
        "wjp_score": 0.34,
        "enforcement_friction_level": "Critical",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Gécamines (mining)", "SNEL (electricity)", "SNCC (railways)"],
        "voluntary_compliance_history": "No",
        "settlement_discount_range": (0.50, 0.80),
        "avg_settlement_years": 15.0,
    },
    "Sudan": {
        "wgi_rule_of_law": 5.0,
        "wgi_corruption": 5.0,
        "wgi_govt_effectiveness": 5.0,
        "wjp_score": 0.36,
        "enforcement_friction_level": "Critical",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Sudan National Petroleum Corporation (Sudapet)", "Sudan Telecom (Sudatel)"],
        "voluntary_compliance_history": "Unknown",
        "settlement_discount_range": (0.50, 0.80),
        "avg_settlement_years": 15.0,
    },
    "Egypt": {
        "wgi_rule_of_law": 30.0,
        "wgi_corruption": 29.0,
        "wgi_govt_effectiveness": 32.0,
        "wjp_score": 0.35,
        "enforcement_friction_level": "High",
        "icsid_member": True,
        "swf_name": "The Sovereign Fund of Egypt (TSFE)",
        "swf_aum_billions": 13.5,
        "major_soes": ["Egyptian General Petroleum Corporation (EGPC)", "Telecom Egypt", "Egypt Air"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.15, 0.35),
        "avg_settlement_years": 5.0,
    },
    "Nigeria": {
        "wgi_rule_of_law": 14.0,
        "wgi_corruption": 14.0,
        "wgi_govt_effectiveness": 14.0,
        "wjp_score": 0.41,
        "enforcement_friction_level": "High",
        "icsid_member": True,
        "swf_name": "Nigerian Sovereign Investment Authority (NSIA)",
        "swf_aum_billions": 2.5,
        "major_soes": [
            "Nigerian National Petroleum Corporation (NNPC)",
            "Nigerian Communications Commission (NCC)",
            "Bank of Industry (BOI)",
        ],
        "voluntary_compliance_history": "Unknown",
        "settlement_discount_range": (0.20, 0.45),
        "avg_settlement_years": 8.0,
    },
    "Ethiopia": {
        "wgi_rule_of_law": 17.0,
        "wgi_corruption": 18.0,
        "wgi_govt_effectiveness": 16.0,
        "wjp_score": 0.38,
        "enforcement_friction_level": "High",
        "icsid_member": False,
        "swf_name": "Ethiopia Investment Holdings (EIH)",
        "swf_aum_billions": 150.0,   # Includes 30 SOEs; primarily domestic
        "major_soes": ["Ethiopian Airlines", "Ethiopian Electric Power (EEP)", "Ethio Telecom"],
        "voluntary_compliance_history": "Unknown",
        "settlement_discount_range": (0.20, 0.45),
        "avg_settlement_years": 8.0,
    },
    "Mozambique": {
        "wgi_rule_of_law": 24.0,
        "wgi_corruption": 22.0,
        "wgi_govt_effectiveness": 20.0,
        "wjp_score": 0.38,
        "enforcement_friction_level": "High",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Electricidade de Moçambique (EDM)", "Mozambique Telecommunications (TDM)"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.15, 0.35),
        "avg_settlement_years": 5.0,
    },
    "Algeria": {
        "wgi_rule_of_law": 26.4,
        "wgi_corruption": 30.2,
        "wgi_govt_effectiveness": 27.4,
        "wjp_score": 0.49,
        "enforcement_friction_level": "High",
        "icsid_member": False,       # Algeria is not an ICSID member
        "swf_name": "Revenue Regulation Fund (Fonds de Régulation des Recettes, FRR)",
        "swf_aum_billions": 50.0,    # Variable with oil revenues
        "major_soes": ["Sonatrach (oil & gas)", "Sonelgaz (electricity)", "Algeria Telecom"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.20, 0.40),
        "avg_settlement_years": 7.0,
    },
    "Cameroon": {
        "wgi_rule_of_law": 20.0,
        "wgi_corruption": 15.0,
        "wgi_govt_effectiveness": 19.0,
        "wjp_score": 0.35,
        "enforcement_friction_level": "High",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Société Nationale de Raffinage (SONARA)", "AES Sonel (electricity)", "Camtel (telecom)"],
        "voluntary_compliance_history": "Unknown",
        "settlement_discount_range": (0.20, 0.45),
        "avg_settlement_years": 8.0,
    },
    "Tanzania": {
        "wgi_rule_of_law": 35.0,
        "wgi_corruption": 33.0,
        "wgi_govt_effectiveness": 36.0,
        "wjp_score": 0.47,
        "enforcement_friction_level": "Moderate",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": [
            "Tanzania Electric Supply Company (TANESCO)",
            "Tanzania Petroleum Development Corporation (TPDC)",
            "Air Tanzania",
        ],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.10, 0.25),
        "avg_settlement_years": 3.5,
    },
    "Kenya": {
        "wgi_rule_of_law": 38.0,
        "wgi_corruption": 34.0,
        "wgi_govt_effectiveness": 38.0,
        "wjp_score": 0.46,
        "enforcement_friction_level": "Moderate",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Kenya Power (KPLC)", "Kenya Airways", "Kenya Pipeline Company (KPC)"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.10, 0.25),
        "avg_settlement_years": 4.0,
    },
    "Morocco": {
        "wgi_rule_of_law": 52.0,
        "wgi_corruption": 45.0,
        "wgi_govt_effectiveness": 50.0,
        "wjp_score": 0.48,
        "enforcement_friction_level": "Moderate",
        "icsid_member": True,
        "swf_name": "Ithmar Capital",
        "swf_aum_billions": 2.0,
        "major_soes": ["OCP (phosphates)", "Royal Air Maroc", "Maroc Telecom", "ONE (electricity)"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.10, 0.25),
        "avg_settlement_years": 4.0,
    },
    "Ghana": {
        "wgi_rule_of_law": 55.0,
        "wgi_corruption": 50.0,
        "wgi_govt_effectiveness": 48.0,
        "wjp_score": 0.55,
        "enforcement_friction_level": "Moderate-Low",
        "icsid_member": True,
        "swf_name": "Ghana Petroleum Funds (Heritage + Stabilization Funds)",
        "swf_aum_billions": 1.0,
        "major_soes": ["Ghana Cocoa Board (GCB)", "Ghana National Petroleum Corporation (GNPC)", "ECG (electricity)"],
        "voluntary_compliance_history": "Yes",
        "settlement_discount_range": (0.0, 0.15),
        "avg_settlement_years": 2.0,
    },
    "Senegal": {
        "wgi_rule_of_law": 48.0,
        "wgi_corruption": 43.0,
        "wgi_govt_effectiveness": 45.0,
        "wjp_score": 0.55,
        "enforcement_friction_level": "Moderate",
        "icsid_member": True,
        "swf_name": "FONSIS (Fonds Souverain d'Investissements Stratégiques)",
        "swf_aum_billions": 0.3,
        "major_soes": ["Senelec (electricity)", "La Poste du Sénégal", "Port Autonome de Dakar"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.10, 0.25),
        "avg_settlement_years": 3.5,
    },
    "South Africa": {
        "wgi_rule_of_law": 55.0,
        "wgi_corruption": 48.0,
        "wgi_govt_effectiveness": 52.0,
        "wjp_score": 0.57,
        "enforcement_friction_level": "Moderate-Low",
        "icsid_member": False,      # South Africa withdrew from ICSID
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Eskom (electricity)", "Transnet (rail/ports)", "SAA (airline)", "PIC"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.05, 0.20),
        "avg_settlement_years": 3.0,
    },
    "Namibia": {
        "wgi_rule_of_law": 60.0,
        "wgi_corruption": 58.0,
        "wgi_govt_effectiveness": 55.0,
        "wjp_score": 0.61,
        "enforcement_friction_level": "Low",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["NamPower (electricity)", "Telecom Namibia", "Air Namibia (closed 2021)"],
        "voluntary_compliance_history": "Yes",
        "settlement_discount_range": (0.0, 0.10),
        "avg_settlement_years": 1.5,
    },
    "Rwanda": {
        "wgi_rule_of_law": 58.0,
        "wgi_corruption": 62.0,
        "wgi_govt_effectiveness": 60.0,
        "wjp_score": 0.63,
        "enforcement_friction_level": "Low",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Rwanda Energy Group (REG)", "RwandAir", "MTN Rwanda (partial state)"],
        "voluntary_compliance_history": "Yes",
        "settlement_discount_range": (0.0, 0.10),
        "avg_settlement_years": 1.5,
    },
    "Botswana": {
        "wgi_rule_of_law": 66.0,
        "wgi_corruption": 65.0,
        "wgi_govt_effectiveness": 63.0,
        "wjp_score": 0.59,
        "enforcement_friction_level": "Low",
        "icsid_member": True,
        "swf_name": "Pula Fund",
        "swf_aum_billions": 5.0,
        "major_soes": ["Botswana Power Corporation (BPC)", "Debswana Diamond Company", "Air Botswana"],
        "voluntary_compliance_history": "Yes",
        "settlement_discount_range": (0.0, 0.10),
        "avg_settlement_years": 1.5,
    },
    "Guinea": {
        "wgi_rule_of_law": 14.0,
        "wgi_corruption": 13.0,
        "wgi_govt_effectiveness": 16.0,
        "wjp_score": 0.36,
        "enforcement_friction_level": "Very High",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": ["Compagnie des Bauxites de Guinée (CBG)", "Électricité de Guinée (EDG)"],
        "voluntary_compliance_history": "Unknown",
        "settlement_discount_range": (0.30, 0.60),
        "avg_settlement_years": 10.0,
    },
    "Gabon": {
        "wgi_rule_of_law": 32.0,
        "wgi_corruption": 28.0,
        "wgi_govt_effectiveness": 30.0,
        "wjp_score": 0.42,
        "enforcement_friction_level": "High",
        "icsid_member": True,
        "swf_name": "FGIS (Fonds Gabonais d'Investissements Stratégiques)",
        "swf_aum_billions": 0.5,
        "major_soes": ["Gabon Oil Company (GOC)", "SEEG (water/electricity)", "Air Gabon"],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.15, 0.35),
        "avg_settlement_years": 5.0,
    },
    "Côte d'Ivoire": {
        "wgi_rule_of_law": 35.0,
        "wgi_corruption": 30.0,
        "wgi_govt_effectiveness": 38.0,
        "wjp_score": 0.45,
        "enforcement_friction_level": "Moderate",
        "icsid_member": True,
        "swf_name": None,
        "swf_aum_billions": None,
        "major_soes": [
            "Société Ivoirienne de Raffinage (SIR)",
            "Compagnie Ivoirienne d'Électricité (CIE)",
            "Air Côte d'Ivoire",
        ],
        "voluntary_compliance_history": "Partial",
        "settlement_discount_range": (0.10, 0.30),
        "avg_settlement_years": 4.0,
    },
}


# ---------------------------------------------------------------------------
# 3. SECTOR STATISTICS
# ---------------------------------------------------------------------------
# Source: Kluwer Arbitration Blog (2012) analysis of ICSID cases involving
# African states. Shares expressed as percentages.
# https://legalblogs.wolterskluwer.com/arbitration-blog/africas-track-record-in-icsid-proceedings/

SECTOR_STATS: dict[str, float] = {
    "Manufacturing":      15.0,
    "Mining":             14.0,
    "Agriculture":        12.0,
    "Oil & Gas":          11.0,
    "Infrastructure":     11.0,
    "Hospitality":        10.0,
    "Construction":        8.0,
    "Energy":              7.0,
    "General Commercial":  7.0,
    "Telecommunications":  5.0,
    "Retail":              2.0,
    "Maritime":            2.0,
    "Legal Services":      1.0,
    "Other":               5.0,
}


# ---------------------------------------------------------------------------
# 4. TREATY BASIS STATISTICS
# ---------------------------------------------------------------------------
# Source: ICSID Africa Special Focus Report (2017), 135 cases.
# https://icsid.worldbank.org/sites/default/files/publications/Caseload%20Statistics/
#   en/Special%20Issues/ICSID%20Web%20Stats%20Africa%20(English)%20June%202017.pdf

TREATY_BASIS_STATS: dict[str, float] = {
    "Bilateral Investment Treaty":  45.0,
    "Investment Contract":          39.0,
    "Investment Law":               16.0,
}


# ---------------------------------------------------------------------------
# 5. AWARD-TO-CLAIM DISTRIBUTION
# ---------------------------------------------------------------------------
# Source: ICSID Caseload Statistics 2025 (global ICSID data; applied to
# African cases). Each tuple: (bracket_label, percentage_of_awards)
# https://icsid.worldbank.org/sites/default/files/publications/
#   2025-1%20ENG%20-%20The%20ICSID%20Caseload%20Statistics%20(Issue%202025-1).pdf

AWARD_TO_CLAIM_DISTRIBUTION: list[tuple[str, float]] = [
    ("<10% of amount claimed",       21.0),
    ("10–25% of amount claimed",     19.0),
    ("26–50% of amount claimed",     26.0),
    ("51–75% of amount claimed",     13.0),
    ("76–100% of amount claimed",    21.0),
]


# ---------------------------------------------------------------------------
# 6. ENFORCEMENT JURISDICTIONS
# ---------------------------------------------------------------------------
# Sources:
# - WilmerHale: https://www.wilmerhale.com/-/media/files/shared_content/editorial/
#     publications/documents/enforcing-arbitral-awards-in-sub-saharan-africa-part-2-lexispslarbitration.pdf
# - Milbank (Oct 2024): https://www.milbank.com/a/web/xgkMhd8J4vi5oXYZpRZsay/
#     milbank-insights_enforcing-arbitral-awards-against-sovereigns-in-the-united-states_october-2024.pdf
# - Slaughter and May (2025): https://www.slaughterandmay.com/insights/new-insights/
#     enforcement-of-awards-against-states-exceptions-to-state-immunity/
# - Oxford Academic (2025): https://academic.oup.com/arbitration/article/41/3/607/8078736

ENFORCEMENT_JURISDICTIONS: dict[str, dict] = {
    "New York": {
        "legal_framework": (
            "Foreign Sovereign Immunities Act (FSIA, 28 U.S.C. §§ 1602–1611); "
            "ICSID Act (22 U.S.C. § 1650a); Federal Arbitration Act (FAA). "
            "ICSID awards treated as final judgments of DC/SDNY courts; "
            "12-year limitations period."
        ),
        "immunity_standard": (
            "Restrictive immunity. Commercial activity exception (§ 1605(a)(2)); "
            "Arbitration exception (§ 1605(a)(6)); Waiver exception (§ 1605(a)(1)). "
            "Execution requires showing property 'used for commercial activity in "
            "the United States' (§ 1610(a))."
        ),
        "key_advantage": (
            "Global asset discovery subpoenas (NML Capital v. Argentina, SCOTUS 2014): "
            "US courts can compel disclosure of ALL foreign state assets worldwide. "
            "FSIA arbitration exception (§ 1605(a)(6)) removes immunity from "
            "jurisdiction for ICSID/NY Convention enforcement. 12-year limitation "
            "for ICSID awards (vs. 3-year FAA limit)."
        ),
        "african_case_precedents": [
            "LETCO v. Liberia (SDNY, 1986) — enforcement permitted but assets not found",
            "Funnekotter v. Zimbabwe (SDNY, 2011/2013) — confirmed; Zimbabwe non-compliant",
            "Miminco v. Congo (DC, 2015) — recognition granted",
            "von Pezold v. Zimbabwe (DC Circuit, 2024) — enforcement upheld; immunity rejected",
            "Amaplat Mauritius v. Zimbabwe (DC Circuit, 2025) — FAA deadline risk",
        ],
    },
    "London": {
        "legal_framework": (
            "State Immunity Act 1978 (SIA); Arbitration (International Investment "
            "Disputes) Act 1966 (ICSID Act). Section 9 SIA: written agreement to "
            "arbitrate removes immunity from jurisdiction. Section 13(3): written "
            "consent or commercial use required for execution."
        ),
        "immunity_standard": (
            "Restrictive immunity. Section 9 SIA (arbitration exception) generally "
            "effective post-Border Timbers (UK Court of Appeal 2025). "
            "Execution immunity under § 13 requires commercial use test or "
            "written consent; 'wholly enforceable' in ICC Rules = execution waiver "
            "per General Dynamics v. Libya (CA 2025, SC 2025)."
        ),
        "key_advantage": (
            "Clear written consent doctrine post-General Dynamics v. Libya. "
            "Issue estoppel from foreign courts (Hulley v. Russia, CA 2025) prevents "
            "states re-arguing consent. Strong commercial courts with arbitration "
            "expertise. SWF assets denied immunity if commercial investment purpose "
            "(2019 Paris approach persuasive)."
        ),
        "african_case_precedents": [
            "Border Timbers / ISL v. Zimbabwe — extended immunity litigation 2020–2025; recognition upheld",
            "von Pezold v. Zimbabwe — recognition and execution proceedings",
            "General Dynamics v. Libya (CA 2025, SC 2025) — 'wholly enforceable' = execution waiver",
            "Nigeria v. P&ID — UK High Court set aside $6.6B award for fraud (Oct 2023)",
        ],
    },
    "Paris": {
        "legal_framework": (
            "French Code of Civil Procedure; French customary international law on "
            "immunity. New York Convention for non-ICSID awards. Liberal enforcement "
            "tradition with narrow public policy exception."
        ),
        "immunity_standard": (
            "Liberal restrictive immunity. Paris Court of Appeal (2019) denied "
            "immunity to Libyan Investment Authority (LIA) assets on grounds they "
            "were used for commercial investment, not sovereign purposes. "
            "French courts interpret immunity most narrowly of major jurisdictions "
            "for SWF/commercial assets."
        ),
        "key_advantage": (
            "Most liberal sovereign wealth fund attachment jurisdiction globally. "
            "Paris CoA (2019) is leading precedent for denying immunity to "
            "commercial-use SWF assets. Largest cluster of international arbitration "
            "seats (ICC, ICDR); strong arbitration judiciary."
        ),
        "african_case_precedents": [
            "Libya SWF (LIA) enforcement (Paris CoA 2019) — immunity denied to LIA assets used for commercial investment",
            "Multiple North African state enforcement proceedings in Paris Commercial Court",
        ],
    },
    "The Hague": {
        "legal_framework": (
            "Dutch Code of Civil Procedure; Netherlands ICSID Implementation Act; "
            "New York Convention. Restrictive immunity doctrine; Dutch Supreme Court "
            "upheld $50B Yukos award (2021)."
        ),
        "immunity_standard": (
            "Restrictive immunity. SWF assets managed by central bank may be immune "
            "under 2020 Dutch Supreme Court ruling (contrast with Sweden's Ascom test). "
            "Commercial use exception applies broadly to trading and revenue-generating "
            "state assets."
        ),
        "key_advantage": (
            "Large body of investment arbitration jurisprudence; Yukos paradigm. "
            "Strong bilateral treaty network makes Netherlands a natural enforcement "
            "seat for European-nationality investors. Generally pro-creditor on "
            "recognition."
        ),
        "african_case_precedents": [
            "Multiple ECT and BIT-based enforcement actions against North African states",
            "Shell v. Nigeria disputes — Netherlands jurisdiction invoked under Netherlands–Nigeria BIT",
        ],
    },
    "Stockholm": {
        "legal_framework": (
            "Swedish Execution Code; Swedish Immunity Act; New York Convention. "
            "SCC Rules provide for Stockholm as default seat for many African BITs."
        ),
        "immunity_standard": (
            "Most creditor-friendly for SWF assets globally. Ascom v. Kazakhstan "
            "(Swedish Supreme Court, 2021): SWF assets are NOT immune from attachment "
            "unless state demonstrates 'a concrete and clear connection to a qualified "
            "purpose of sovereign character'. General investment/savings mandate is "
            "insufficient to establish immunity."
        ),
        "key_advantage": (
            "Landmark Ascom (2021) precedent for SWF attachment: applicable to all "
            "African SWFs managed for general investment purposes (LIA, NSIA, TSFE, "
            "Pula Fund etc.). SCC arbitration centre widely used for BIT disputes. "
            "Strong rule-of-law environment."
        ),
        "african_case_precedents": [
            "Ascom v. Kazakhstan (2021) — SWF attachment precedent applicable to African SWFs",
            "Various SCC-seated arbitrations with African state respondents",
        ],
    },
    "Australia": {
        "legal_framework": (
            "Foreign States Immunities Act 1985 (Cth); Australian ICSID Act. "
            "High Court of Australia (Infrastructure Services Luxembourg v. Kingdom "
            "of Spain, 2023): ICSID Convention ratification constitutes implied "
            "waiver of immunity from recognition proceedings."
        ),
        "immunity_standard": (
            "Restrictive immunity. High Court 2023 ruling extends automatic waiver "
            "from ICSID ratification to recognition (not just enforcement). "
            "Commercial use test for execution. Emerging jurisdiction for "
            "enforcement given Asia-Pacific trade links."
        ),
        "key_advantage": (
            "2023 High Court precedent that ICSID Convention ratification = "
            "immunity waiver for recognition — directly applicable to 41 African "
            "ICSID member states. Strong rule-of-law environment. Growing relevance "
            "for African states with Australian mining investor exposure "
            "(Winshear, Indiana Resources)."
        ),
        "african_case_precedents": [
            "Infrastructure Services Luxembourg v. Kingdom of Spain (High Court 2023) — ICSID ratification = recognition immunity waiver; applicable template for African states",
            "Emerging venue for enforcement against Tanzania (Canadian mining cases with Australian-listed investors)",
        ],
    },
}


# ---------------------------------------------------------------------------
# 7. BIT NETWORK
# ---------------------------------------------------------------------------
# Key BITs relevant to the cases in the database.
# Format: {(country_a, country_b): {"year": int, "in_force": bool}}

BIT_NETWORK: dict[tuple[str, str], dict] = {
    ("Germany", "Ghana"):         {"year": 1995, "in_force": True},
    ("Netherlands", "Nigeria"):   {"year": 1992, "in_force": True},
    ("Netherlands", "Zimbabwe"):  {"year": 1996, "in_force": True},
    ("Germany", "Zimbabwe"):      {"year": 1995, "in_force": True},
    ("Switzerland", "Zimbabwe"):  {"year": 1996, "in_force": True},
    ("Netherlands", "Tanzania"):  {"year": 2001, "in_force": False},  # Terminated Aug 2018
    ("UK", "Tanzania"):           {"year": 1994, "in_force": True},
    ("Canada", "Tanzania"):       {"year": 2013, "in_force": True},
    ("Sweden", "Tanzania"):       {"year": 1999, "in_force": True},
    ("Mauritius", "Tanzania"):    {"year": 2009, "in_force": True},
    ("UK", "Kenya"):              {"year": 1999, "in_force": True},
    ("USA", "Zaire/DRC"):         {"year": 1984, "in_force": True},
    ("Netherlands", "Senegal"):   {"year": 1979, "in_force": True},
    ("UAE", "Guinea"):            {"year": 2011, "in_force": True},
    ("UAE", "Gabon"):             {"year": 2012, "in_force": True},
    ("Spain", "Egypt"):           {"year": 1994, "in_force": True},
    ("France", "Egypt"):          {"year": 1974, "in_force": True},
    ("Italy", "Mozambique"):      {"year": 1998, "in_force": True},
    ("UK", "Mozambique"):         {"year": 2004, "in_force": True},
    ("Ghana", "UK"):              {"year": 1989, "in_force": True},
    ("Malaysia", "Ghana"):        {"year": 1996, "in_force": True},
}


# ---------------------------------------------------------------------------
# 8. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_cases_by_country(country: str) -> list[dict]:
    """Return all cases where respondent_state matches the given country name.

    Args:
        country: Name of the African respondent state (e.g., "Tanzania").

    Returns:
        List of case dicts for that country.
    """
    return [c for c in CASES if c["respondent_state"].lower() == country.lower()]


def get_cases_by_sector(sector: str) -> list[dict]:
    """Return all cases in the given sector.

    Args:
        sector: Sector string (e.g., "Mining", "Oil & Gas").

    Returns:
        List of matching case dicts.
    """
    return [c for c in CASES if c["sector"].lower() == sector.lower()]


def get_cases_by_outcome(outcome: str) -> list[dict]:
    """Return all cases with the given outcome.

    Args:
        outcome: One of "Investor Win", "State Win", "Settled", "Pending",
                 "Discontinued", "Annulled".

    Returns:
        List of matching case dicts.
    """
    return [c for c in CASES if c["outcome"].lower() == outcome.lower()]


def calculate_historical_rates(country: Optional[str] = None) -> dict:
    """Compute historical outcome and financial ratios from the case database.

    Considers only decided / resolved cases (excludes Pending).

    Args:
        country: Optional respondent state filter. If None, calculates across
                 all cases in the database.

    Returns:
        dict with keys:
            investor_win_rate        (float, 0–1)
            state_win_rate           (float, 0–1)
            settlement_rate          (float, 0–1)
            discontinued_rate        (float, 0–1)
            annulment_rate           (float, 0–1)
            avg_award_to_claim_ratio (float or None — mean where both values known)
            annulment_success_rate   (float, 0–1 — share of annulment attempts that succeeded)
            total_cases              (int)
            resolved_cases           (int)
    """
    pool = get_cases_by_country(country) if country else CASES

    resolved = [c for c in pool if c["outcome"] != "Pending"]
    total = len(pool)
    n = len(resolved)

    if n == 0:
        return {
            "investor_win_rate": None,
            "state_win_rate": None,
            "settlement_rate": None,
            "discontinued_rate": None,
            "annulment_rate": None,
            "avg_award_to_claim_ratio": None,
            "annulment_success_rate": None,
            "total_cases": total,
            "resolved_cases": 0,
        }

    outcome_counts: dict[str, int] = defaultdict(int)
    for c in resolved:
        outcome_counts[c["outcome"]] += 1

    investor_wins = outcome_counts["Investor Win"]
    state_wins    = outcome_counts["State Win"]
    settled       = outcome_counts["Settled"]
    discontinued  = outcome_counts["Discontinued"]
    annulled      = outcome_counts["Annulled"]

    # Award-to-claim ratio: computed only where both amounts are known & positive
    ratios = []
    for c in resolved:
        claimed = c.get("amount_claimed_usd")
        awarded = c.get("amount_awarded_usd")
        if (
            claimed is not None
            and awarded is not None
            and claimed > 0
            and awarded > 0
        ):
            ratios.append(awarded / claimed)
    avg_ratio = (sum(ratios) / len(ratios)) if ratios else None

    # Annulment success rate
    attempted = [c for c in pool if c.get("annulment_attempted")]
    succeeded = [
        c for c in attempted
        if c.get("annulment_outcome") in ("Annulment granted", "Partial annulment")
    ]
    ann_success = (len(succeeded) / len(attempted)) if attempted else None

    return {
        "investor_win_rate":        investor_wins / n,
        "state_win_rate":           state_wins / n,
        "settlement_rate":          settled / n,
        "discontinued_rate":        discontinued / n,
        "annulment_rate":           annulled / n,
        "avg_award_to_claim_ratio": avg_ratio,
        "annulment_success_rate":   ann_success,
        "total_cases":              total,
        "resolved_cases":           n,
    }


# ---------------------------------------------------------------------------
# Quick self-test when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Total cases in database: {len(CASES)}")
    print(f"Country profiles loaded: {len(COUNTRY_PROFILES)}")

    print("\n--- Global historical rates ---")
    global_rates = calculate_historical_rates()
    for k, v in global_rates.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Tanzania cases ---")
    tz = get_cases_by_country("Tanzania")
    for c in tz:
        print(f"  [{c['year_filed']}] {c['case_name']} → {c['outcome']}")

    print("\n--- Mining sector cases ---")
    mining = get_cases_by_sector("Mining")
    for c in mining:
        print(f"  {c['respondent_state']}: {c['case_name']} → {c['outcome']}")

    print("\n--- Zimbabwe profile ---")
    zw = COUNTRY_PROFILES["Zimbabwe"]
    for k, v in zw.items():
        print(f"  {k}: {v}")
