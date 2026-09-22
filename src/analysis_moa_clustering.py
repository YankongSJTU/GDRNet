# -*- coding: utf-8 -*-
"""
Drug Mechanism of Action (MOA) Annotations
============================================
Provides MOA category mapping for GDSC drugs.
Used for color-coding drug heatmaps (Figure 4E revision).

Output: results/tables/drug_moa_annotations.csv

NOTE: This file contains a manually curated MOA mapping for the 229 GDSC drugs.
Drugs not in this mapping will be labeled as 'Other'.
"""

import io
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results/tables"
EXT = ROOT / "data/external"

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

# ─── MOA Mapping (curated for GDSC drugs) ──────────────────────────────────
# Categories based on standard pharmacological classification.
# Sources: GDSC drug annotations, PubChem, DrugBank.

MOA_MAP = {
    # Topoisomerase Inhibitors
    "Topotecan": "Topoisomerase Inhibitor",
    "Irinotecan": "Topoisomerase Inhibitor",
    "Etoposide": "Topoisomerase Inhibitor",
    "Doxorubicin": "Topoisomerase Inhibitor",
    "Daunorubicin": "Topoisomerase Inhibitor",
    "Epirubicin": "Topoisomerase Inhibitor",
    "Idarubicin": "Topoisomerase Inhibitor",
    "Mitoxantrone": "Topoisomerase Inhibitor",
    "Amsacrine": "Topoisomerase Inhibitor",
    "Teniposide": "Topoisomerase Inhibitor",

    # Platinum Agents (DNA cross-linking)
    "Cisplatin": "Platinum Agent",
    "Carboplatin": "Platinum Agent",
    "Oxaliplatin": "Platinum Agent",
    "Satraplatin": "Platinum Agent",

    # Kinase Inhibitors (broad)
    "Erlotinib": "Kinase Inhibitor (EGFR)",
    "Gefitinib": "Kinase Inhibitor (EGFR)",
    "Afatinib": "Kinase Inhibitor (EGFR)",
    "Lapatinib": "Kinase Inhibitor (EGFR/HER2)",
    "Sunitinib": "Kinase Inhibitor (multi-target)",
    "Sorafenib": "Kinase Inhibitor (multi-target)",
    "Pazopanib": "Kinase Inhibitor (multi-target)",
    "Axitinib": "Kinase Inhibitor (VEGFR)",
    "Imatinib": "Kinase Inhibitor (BCR-ABL/c-KIT)",
    "Dasatinib": "Kinase Inhibitor (BCR-ABL/SRC)",
    "Nilotinib": "Kinase Inhibitor (BCR-ABL)",
    "Bosutinib": "Kinase Inhibitor (BCR-ABL/SRC)",
    "Ponatinib": "Kinase Inhibitor (BCR-ABL)",
    "Crizotinib": "Kinase Inhibitor (ALK/MET)",
    "Ceritinib": "Kinase Inhibitor (ALK)",
    "Alectinib": "Kinase Inhibitor (ALK)",
    "Lorlatinib": "Kinase Inhibitor (ALK)",
    "Vemurafenib": "Kinase Inhibitor (BRAF)",
    "Dabrafenib": "Kinase Inhibitor (BRAF)",
    "Trametinib": "Kinase Inhibitor (MEK)",
    "Cobimetinib": "Kinase Inhibitor (MEK)",
    "Binimetinib": "Kinase Inhibitor (MEK)",
    "Selumetinib": "Kinase Inhibitor (MEK)",
    "PD-0325901": "Kinase Inhibitor (MEK)",
    "Refametinib": "Kinase Inhibitor (MEK)",
    "AZD6244": "Kinase Inhibitor (MEK)",
    "GSK1120212": "Kinase Inhibitor (MEK)",

    # HDAC Inhibitors
    "Vorinostat": "HDAC Inhibitor",
    "Panobinostat": "HDAC Inhibitor",
    "Romidepsin": "HDAC Inhibitor",
    "Belinostat": "HDAC Inhibitor",
    "Entinostat": "HDAC Inhibitor",
    "Quisinostat": "HDAC Inhibitor",
    "SB939": "HDAC Inhibitor",
    "CUDC-101": "HDAC Inhibitor",
    "Ricolinostat": "HDAC Inhibitor",

    # DNA Alkylating Agents
    "Temozolomide": "DNA Alkylating Agent",
    "Cyclophosphamide": "DNA Alkylating Agent (Prodrug)",
    "Ifosfamide": "DNA Alkylating Agent (Prodrug)",
    "Melphalan": "DNA Alkylating Agent",
    "Chlorambucil": "DNA Alkylating Agent",
    "Busulfan": "DNA Alkylating Agent",
    "Carmustine": "DNA Alkylating Agent",
    "Lomustine": "DNA Alkylating Agent",
    "Streptozocin": "DNA Alkylating Agent",
    "Dacarbazine": "DNA Alkylating Agent (Prodrug)",
    "Procarbazine hydrochloride": "DNA Alkylating Agent (Prodrug)",
    "Altretamine": "DNA Alkylating Agent (Prodrug)",
    "Thiotepa": "DNA Alkylating Agent",
    "Mitomycin C": "DNA Alkylating Agent",
    "Bendamustine": "DNA Alkylating Agent",
    "Treosulfan": "DNA Alkylating Agent",

    # Antimetabolites
    "5-Fluorouracil": "Antimetabolite (pyrimidine)",
    "Capecitabine": "Antimetabolite (pyrimidine, Prodrug)",
    "Gemcitabine": "Antimetabolite (pyrimidine)",
    "Cytarabine": "Antimetabolite (pyrimidine)",
    "Azacitidine": "Antimetabolite (pyrimidine)",
    "Decitabine": "Antimetabolite (pyrimidine)",
    "6-Mercaptopurine": "Antimetabolite (purine)",
    "6-Thioguanine": "Antimetabolite (purine)",
    "Methotrexate": "Antimetabolite (folate)",
    "Pemetrexed": "Antimetabolite (folate)",
    "Raltitrexed": "Antimetabolite (folate)",
    "Cladribine": "Antimetabolite (purine)",
    "Clofarabine": "Antimetabolite (purine)",
    "Fludarabine": "Antimetabolite (purine)",
    "Nelarabine": "Antimetabolite (purine)",

    # Tubulin Inhibitors
    "Paclitaxel": "Tubulin Inhibitor (stabilizer)",
    "Docetaxel": "Tubulin Inhibitor (stabilizer)",
    "Vinblastine": "Tubulin Inhibitor (destabilizer)",
    "Vincristine": "Tubulin Inhibitor (destabilizer)",
    "Vinorelbine": "Tubulin Inhibitor (destabilizer)",
    "Eribulin": "Tubulin Inhibitor (destabilizer)",
    "Cabazitaxel": "Tubulin Inhibitor (stabilizer)",

    # Proteasome Inhibitors
    "Bortezomib": "Proteasome Inhibitor",
    "Carfilzomib": "Proteasome Inhibitor",
    "Ixazomib": "Proteasome Inhibitor",

    # BCL-2 / BCL-xL Inhibitors
    "Navitoclax": "BCL-2/BCL-xL Inhibitor",
    "Venetoclax": "BCL-2 Inhibitor",
    "ABT-263": "BCL-2/BCL-xL Inhibitor",
    "A-1331852": "BCL-xL Inhibitor",

    # CDK Inhibitors
    "Palbociclib": "CDK Inhibitor (CDK4/6)",
    "Ribociclib": "CDK Inhibitor (CDK4/6)",
    "Abemaciclib": "CDK Inhibitor (CDK4/6)",
    "Dinaciclib": "CDK Inhibitor (CDK1/2/5/9)",
    "SNS-032": "CDK Inhibitor (CDK2/7/9)",
    "AZD5438": "CDK Inhibitor (CDK1/2/9)",
    "CGP-60474": "CDK Inhibitor (CDK1/2/4/6)",
    "AT7519": "CDK Inhibitor (CDK1/2/4/6/9)",
    "RO-3306": "CDK Inhibitor (CDK1)",
    "PD-0332991": "CDK Inhibitor (CDK4/6)",

    # PI3K/AKT/mTOR Inhibitors
    "Everolimus": "mTOR Inhibitor",
    "Temsirolimus": "mTOR Inhibitor",
    "Ridaforolimus": "mTOR Inhibitor",
    "AZD8055": "mTOR Inhibitor",
    "BEZ235": "PI3K/mTOR Inhibitor",
    "BKM120": "PI3K Inhibitor",
    "GDC-0941": "PI3K Inhibitor",
    "BYL719": "PI3K Inhibitor (alpha)",
    "GDC-0980": "PI3K/mTOR Inhibitor",
    "MK-2206": "AKT Inhibitor",
    "AZD5363": "AKT Inhibitor",
    "Perifosine": "AKT Inhibitor",
    "GSK690693": "AKT Inhibitor",
    "MK-8669": "mTOR Inhibitor",

    # PARP Inhibitors
    "Olaparib": "PARP Inhibitor",
    "Rucaparib": "PARP Inhibitor",
    "Niraparib": "PARP Inhibitor",
    "Talazoparib": "PARP Inhibitor",
    "Veliparib": "PARP Inhibitor",
    "AG-014699": "PARP Inhibitor",
    "BMN-673": "PARP Inhibitor",

    # HSP90 Inhibitors
    "17-AAG": "HSP90 Inhibitor",
    "Ganetespib": "HSP90 Inhibitor",
    "AUY922": "HSP90 Inhibitor",
    "NVP-AUY922": "HSP90 Inhibitor",

    # WEE1/CHK1 Inhibitors
    "MK-1775": "WEE1 Inhibitor",
    "AZD7762": "CHK1 Inhibitor",
    "Prexasertib": "CHK1 Inhibitor",
    "SCH900776": "CHK1 Inhibitor",
    "CCT245737": "CHK1 Inhibitor",

    # ATR/ATM Inhibitors
    "VE-821": "ATR Inhibitor",
    "AZD6738": "ATR Inhibitor",
    "M6620": "ATR Inhibitor",
    "KU-55933": "ATM Inhibitor",
    "AZD0156": "ATR Inhibitor",
    "Ceralasertib": "ATR Inhibitor",
    "Camonsertib": "ATR Inhibitor",

    # Aurora Kinase Inhibitors
    "Barasertib": "Aurora Kinase Inhibitor",
    "Tozasertib": "Aurora Kinase Inhibitor",
    "Alisertib": "Aurora Kinase Inhibitor",
    "ZM447439": "Aurora Kinase Inhibitor",
    "Reversine": "Aurora Kinase Inhibitor",

    # JAK/STAT Inhibitors
    "Ruxolitinib": "JAK Inhibitor",
    "Fedratinib": "JAK Inhibitor",
    "Cyt387": "JAK Inhibitor",
    "AZD1480": "JAK Inhibitor",
    "TG101348": "JAK Inhibitor",
    "LS104": "STAT3 Inhibitor",
    "Stattic": "STAT3 Inhibitor",
    "S3I-201": "STAT3 Inhibitor",

    # BET/BRD Inhibitors
    "JQ1": "BET Inhibitor",
    "I-BET-762": "BET Inhibitor",
    "OTX015": "BET Inhibitor",
    "CPI-0610": "BET Inhibitor",
    "PFI-1": "BET Inhibitor",

    # PKC Inhibitors
    "Enzastaurin": "PKC Inhibitor",
    "AEB071": "PKC Inhibitor",
    "Gö6983": "PKC Inhibitor",
    "Sotrastaurin": "PKC Inhibitor",

    # PLK1 Inhibitors
    "BI-2536": "PLK1 Inhibitor",
    "Volasertib": "PLK1 Inhibitor",
    "GSK461364": "PLK1 Inhibitor",
    "BI-6727": "PLK1 Inhibitor",

    # CDK9/Transcription Inhibitors
    "Flavopiridol": "CDK9 Inhibitor",
    "Dinaciclib": "CDK9 Inhibitor",
    "SNS-032": "CDK9 Inhibitor",
    "AZD-4573": "CDK9 Inhibitor",
    "LDC000067": "CDK9 Inhibitor",
    "BAY-1143572": "CDK9 Inhibitor",
    "NVP-2": "CDK9 Inhibitor",
    "AT-7519": "CDK9 Inhibitor",

    # Others / Targeted agents
    "Gefitinib": "Kinase Inhibitor (EGFR)",
    "Lapatinib": "Kinase Inhibitor (EGFR/HER2)",
    "AZD-9291": "Kinase Inhibitor (EGFR T790M)",
    "Osimertinib": "Kinase Inhibitor (EGFR T790M)",
    "AZD3759": "Kinase Inhibitor (EGFR)",
    "Pelitinib": "Kinase Inhibitor (EGFR)",
    "Neratinib": "Kinase Inhibitor (HER2)",
    "Tucatinib": "Kinase Inhibitor (HER2)",
    "Tucatinib (ARQ 761)": "Kinase Inhibitor (HER2)",
    "AZD-8931": "Kinase Inhibitor (EGFR)",
    "GSK-2879552": "GSK-3 Inhibitor",
    "CHIR-99021": "GSK-3 Inhibitor",
    "SB-216763": "GSK-3 Inhibitor",
    "BI-78D3": "GSK-3 Inhibitor",
    "TWS-119": "GSK-3 Inhibitor",
}

# Color palette for MOA categories (Okabe-Ito style)
MOA_COLORS = {
    "Topoisomerase Inhibitor": "#E69F00",
    "Platinum Agent": "#56B4E9",
    "Kinase Inhibitor": "#009E73",
    "HDAC Inhibitor": "#F0E442",
    "DNA Alkylating Agent": "#0072B2",
    "DNA Alkylating Agent (Prodrug)": "#0072B2",
    "Antimetabolite": "#D55E00",
    "Tubulin Inhibitor": "#CC79A7",
    "Tubulin Inhibitor (stabilizer)": "#CC79A7",
    "Tubulin Inhibitor (destabilizer)": "#CC79A7",
    "Proteasome Inhibitor": "#666666",
    "BCL-2/BCL-xL Inhibitor": "#8C564B",
    "BCL-2 Inhibitor": "#8C564B",
    "BCL-xL Inhibitor": "#8C564B",
    "CDK Inhibitor": "#E377C2",
    "PARP Inhibitor": "#7F7F7F",
    "HSP90 Inhibitor": "#BCBD22",
    "WEE1 Inhibitor": "#17BECF",
    "CHK1 Inhibitor": "#17BECF",
    "ATR Inhibitor": "#AEC7E8",
    "Aurora Kinase Inhibitor": "#FFB982",
    "JAK Inhibitor": "#98DF8A",
    "STAT3 Inhibitor": "#98DF8A",
    "BET Inhibitor": "#FF9896",
    "PKC Inhibitor": "#C5B0D5",
    "PLK1 Inhibitor": "#C7C7C7",
    "CDK9 Inhibitor": "#DBDB8D",
    "PI3K/mTOR Inhibitor": "#9EDAE5",
    "PI3K Inhibitor": "#9EDAE5",
    "mTOR Inhibitor": "#9EDAE5",
    "AKT Inhibitor": "#9EDAE5",
    "Other": "#BBBBBB",
}


def get_moa_category(drug_name):
    """Look up MOA category for a drug name."""
    if drug_name in MOA_MAP:
        return MOA_MAP[drug_name]
    # Fuzzy match for common aliases
    for key, val in MOA_MAP.items():
        if key.lower() == drug_name.lower():
            return val
    return "Other"


def get_moa_color(category):
    """Get color for an MOA category."""
    return MOA_COLORS.get(category, MOA_COLORS["Other"])


def build_moa_table(drug_smiles_file=None):
    """Build complete MOA annotation table for all GDSC drugs."""
    if drug_smiles_file is None:
        drug_smiles_file = str(ROOT / "data/external/gdsc_drug_smiles.csv")

    drugs = pd.read_csv(drug_smiles_file)
    drugs["moa_category"] = drugs["drug_name"].apply(get_moa_category)
    drugs["moa_color"] = drugs["moa_category"].apply(get_moa_color)

    # Summary
    print("  MOA Category Distribution:", flush=True)
    for cat, grp in drugs.groupby("moa_category"):
        print(f"    {cat:45s}  {len(grp):3d} drugs", flush=True)

    return drugs


def main():
    print("=" * 60, flush=True)
    print("  Drug MOA Annotation", flush=True)
    print("=" * 60, flush=True)

    moa_df = build_moa_table()
    out_path = TABLES / "drug_moa_annotations.csv"
    moa_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
