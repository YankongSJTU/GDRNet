# -*- coding: utf-8 -*-
"""
Resolve SMILES for PDAC (GSE194249) organoid drugs.

- 59 epigenetic drugs with Selleckchem catalog numbers → extract base name → PubChem lookup
- 5 chemotherapy drugs with abbreviations → manual mapping → GDSC SMILES
- Drugs already in GDSC → use existing SMILES

Output: data/external/pdac_drug_smiles.csv (drug_name, smiles, catalog, source)
"""
import pandas as pd
import numpy as np
import re, time, sys
from pathlib import Path

ROOT = Path("/export/home/kongyan/project/Organoid")
EXT_DIR = ROOT / "data/external"
RAW_DIR = ROOT / "data/raw/geo"


def extract_base_name(product_name: str) -> str:
    """Extract the base drug name from product name like 'Vorinostat (SAHA, MK0683)'."""
    # Remove content in parentheses and trailing salts
    name = re.sub(r'\s*\([^)]*\)', '', product_name).strip()
    # Remove common salt suffixes
    name = re.sub(r'\s+(HCl|2HCl|HBr|Na|free base|L-\(+\)-Tartaric acid)\s*$', '', name).strip()
    # Fix leading special chars like (+)-JQ1 → JQ1
    name = re.sub(r'^[+\-()]+', '', name).strip()
    name = name.strip()
    return name


def main():
    print("=" * 60)
    print("  Resolving SMILES for PDAC drugs (GSE194249)")
    print("=" * 60)

    # Load AUC data to get drug IDs
    auc = pd.read_excel(RAW_DIR / "GSE194249/GSE194249_drug_AUC_MOESM11.xlsx", header=1)
    auc_ids = auc.iloc[:, 0].tolist()  # 64 drug IDs (59 S* + 5 chemo)
    print(f"  Total AUC drug IDs: {len(auc_ids)}")

    # Load drug list for mapping S* → product name
    drug_list = pd.read_excel(RAW_DIR / "GSE194249/GSE194249_drug_list_MOESM10.xlsx", header=1)
    drug_list.columns = ['Number', 'Catalog', 'Product_Name', 'Concentration',
                          'Formula', 'Target', 'Pathway', 'Information']
    catalog_to_name = dict(zip(drug_list['Catalog'], drug_list['Product_Name']))

    # Load GDSC SMILES
    gdsc = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    gdsc_map = dict(zip(gdsc['drug_name'], gdsc['smiles']))
    print(f"  GDSC drugs: {len(gdsc_map)}")

    # Manual mapping for chemo drugs
    chemo_map = {
        '5-FU': ('5-Fluorouracil', None),
        'GEM':  ('Gemcitabine', None),
        'IRI':  ('Irinotecan', None),
        'OXA':  ('Oxaliplatin', None),
        'PTX':  ('Paclitaxel', None),
    }

    # Build drug name mapping
    drug_info = []
    for did in auc_ids:
        # Chemo drugs
        if did in chemo_map:
            base_name, _ = chemo_map[did]
            drug_info.append({
                'catalog': did,
                'product_name': did,
                'drug_name': base_name,
                'source': 'chemo_manual',
            })
            continue

        # Selleckchem catalog drugs
        product_name = catalog_to_name.get(did, '')
        if not product_name:
            print(f"  [WARN] No product name for {did}")
            drug_info.append({
                'catalog': did,
                'product_name': '',
                'drug_name': did,
                'source': 'unknown',
            })
            continue

        base_name = extract_base_name(product_name)
        drug_info.append({
            'catalog': did,
            'product_name': product_name,
            'drug_name': base_name,
            'source': 'selleckchem',
        })

    df = pd.DataFrame(drug_info)
    print(f"\n  Resolved base names for {len(df)} drugs")

    # Get SMILES
    smiles_results = {}
    need_pubchem = []

    # First pass: check GDSC
    for _, row in df.iterrows():
        name = row['drug_name']
        if name in gdsc_map:
            smiles_results[name] = (gdsc_map[name], 'gdsc')
        else:
            need_pubchem.append(row)

    print(f"  Found in GDSC: {len(smiles_results)}")
    print(f"  Need PubChem lookup: {len(need_pubchem)}")

    # Second pass: PubChem lookup
    import requests
    for row in need_pubchem:
        name = row['drug_name']
        name = row['drug_name']
        if name in smiles_results:
            continue
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                props = data["PropertyTable"]["Properties"][0]
                # PubChem may return "SMILES" or "IsomericSMILES"
                smi = props.get("IsomericSMILES") or props.get("SMILES", "")
                if smi:
                    smiles_results[name] = (smi, 'pubchem')
                    print(f"    ✓ {name}: SMILES found via PubChem")
                else:
                    print(f"    ✗ {name}: no SMILES in response keys={list(props.keys())}")
            else:
                # Try fallback: if name contains space, try first word
                fallback = name.split()[0]
                if fallback != name:
                    try:
                        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(fallback)}/property/IsomericSMILES,SMILES/JSON"
                        resp2 = requests.get(url2, timeout=15)
                        if resp2.status_code == 200:
                            data2 = resp2.json()
                            props2 = data2["PropertyTable"]["Properties"][0]
                            smi = props2.get("IsomericSMILES") or props2.get("SMILES", "")
                            if smi:
                                smiles_results[name] = (smi, 'pubchem')
                                print(f"    ✓ {name}: SMILES found via PubChem (fallback: {fallback})")
                                continue
                    except Exception:
                        pass
                print(f"    ✗ {name}: PubChem returned {resp.status_code}")
        except Exception as e:
            print(f"    ✗ {name}: {e}")
        time.sleep(0.3)

    # Build final output
    output_rows = []
    for _, row in df.iterrows():
        name = row['drug_name']
        if name in smiles_results:
            smi, source = smiles_results[name]
            output_rows.append({
                'drug_name': name,
                'smiles': smi,
                'catalog': row['catalog'],
                'product_name': row['product_name'],
                'source': source,
            })
        else:
            output_rows.append({
                'drug_name': name,
                'smiles': '',
                'catalog': row['catalog'],
                'product_name': row['product_name'],
                'source': 'not_found',
            })

    out_df = pd.DataFrame(output_rows)
    found = (out_df['smiles'] != '').sum()
    missing = out_df[out_df['smiles'] == '']

    print(f"\n  SMILES resolved: {found}/{len(out_df)}")
    if len(missing) > 0:
        print(f"  Missing SMILES for {len(missing)} drugs:")
        for _, r in missing.iterrows():
            print(f"    - {r['drug_name']} ({r['catalog']})")

    # Validate SMILES with RDKit
    try:
        from rdkit import Chem
        valid = 0
        for _, r in out_df.iterrows():
            if r['smiles']:
                mol = Chem.MolFromSmiles(r['smiles'])
                if mol is not None:
                    valid += 1
                else:
                    print(f"  [WARN] Invalid SMILES: {r['drug_name']} = {r['smiles']}")
        print(f"\n  RDKit validated: {valid}/{found} SMILES are valid")
    except ImportError:
        print("  [WARN] RDKit not available for validation")

    # Save
    out_path = EXT_DIR / "pdac_drug_smiles.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
