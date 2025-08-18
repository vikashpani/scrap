import os

def update_service_date_in_folder(folder_path: str, new_service_date: str, output_folder: str):
    """
    Update service dates in EDI files inside a folder.
    
    - Institutional claims (837I) → update DTP*434
    - Professional claims (837P) → update DTP*472
    
    Args:
        folder_path (str): Path to the folder containing EDI files.
        new_service_date (str): New date of service (YYYYMMDD).
        output_folder (str): Path to save updated files.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".edi", ".txt")):
            continue  # Skip non-EDI files

        input_file = os.path.join(folder_path, filename)
        output_file = os.path.join(output_folder, filename)

        with open(input_file, "r") as f:
            edi_content = f.read()

        segments = edi_content.split("~")
        updated_segments = []
        claim_type = None

        for seg in segments:
            if seg.startswith("ST*837*"):
                if "005010X223A2" in edi_content:  
                    claim_type = "Institutional"
                elif "005010X222A1" in edi_content:  
                    claim_type = "Professional"

            if claim_type == "Institutional" and seg.startswith("DTP*434*"):
                parts = seg.split("*")
                if len(parts) >= 3:
                    parts[-1] = new_service_date
                    seg = "*".join(parts)

            elif claim_type == "Professional" and seg.startswith("DTP*472*"):
                parts = seg.split("*")
                if len(parts) >= 3:
                    parts[-1] = new_service_date
                    seg = "*".join(parts)

            updated_segments.append(seg)

        updated_edi = "~".join(updated_segments)

        with open(output_file, "w") as f:
            f.write(updated_edi)

        print(f"✅ Updated {filename} and saved to {output_file}")


# Example usage:
# folder with edi files = "input_edi"
# updated files will go to "output_edi"
# new service date = "20250818"

update_service_date_in_folder(
    folder_path="input_edi",
    new_service_date="20250818",
    output_folder="output_edi"
)













# app.py
import streamlit as st
import pandas as pd
import re
import os
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from typing import Dict, List, Tuple

st.set_page_config(page_title="837I ↔ 820 Comparison (Strict match)", layout="wide")

# -------------------
# Utilities
# -------------------
def norm_str(x):
    return (x or "").strip()

def sanitize_sheet_name(name: str) -> str:
    if not name:
        return "UNKNOWN"
    s = re.sub(r'[\[\]\*\?/\\:]', "_", name)
    return s[:31]

def to_decimal(x):
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def equal_amounts(a, b):
    aa = to_decimal(a)
    bb = to_decimal(b)
    if aa is None or bb is None:
        return False
    return aa == bb

def name_match(first1, last1, first2, last2):
    return norm_str(first1).upper() == norm_str(first2).upper() and norm_str(last1).upper() == norm_str(last2).upper()

# -------------------
# ADX Code Definitions
# -------------------
ADX_CODES = {
    "52": "Credit for Overpayment",
    "53": "Remittance for Previous Underpayment",
    "80": "Overpayment",
    "81": "Credit as Agreed",
    "86": "Duplicate Payment",
    "BJ": "Insurance Charge",
    "H1": "Information Forthcoming",
    "H6": "Partial Payment Remitted",
    "RU": "Interest",
    "WO": "Overpayment Recovery",
    "WW": "Overpayment Credit",
}

# -------------------
# Parsers
# -------------------

def parse_837i(content: str) -> Dict[str, List[dict]]:
    providers = defaultdict(list)
    current_provider = None
    current_first = ""
    current_last = ""

    for raw in content.split("~"):
        seg = raw.strip()
        if not seg:
            continue
        parts = seg.split("*")
        if len(parts) < 1:
            continue

        if parts[0] == "NM1" and len(parts) >= 4 and parts[1] == "41":
            current_provider = norm_str(parts[3])
        elif parts[0] == "NM1" and len(parts) >= 5 and parts[1] == "IL":
            current_last = norm_str(parts[3])
            current_first = norm_str(parts[4])
        elif parts[0] == "CLM" and len(parts) >= 3:
            clm01 = parts[1]
            amt = norm_str(parts[2])
            claim_type = ""
            claim_id = ""
            member_id = ""

            if clm01.startswith("C-"):
                claim_type = "C"
                m = re.match(r"^C-([^\-\s]+)", clm01)
                if m:
                    claim_id = m.group(1)
            elif clm01.startswith("P-"):
                claim_type = "P"
                m = re.match(r"^P-([^\-\s]+)", clm01)
                if m:
                    member_id = m.group(1)
                    if len(member_id) > 1 and re.match(r".*[A-Z]$", member_id) and re.search(r"\d", member_id[:-1]):
                        member_id = member_id[:-1]

            if current_provider is None:
                current_provider = "UNKNOWN_PROVIDER"

            record = {
                "Provider": current_provider,
                "FirstName": current_first,
                "LastName": current_last,
                "Type": claim_type,
                "ClaimID": claim_id,
                "MemberID": member_id,
                "Amount": amt
            }
            providers[current_provider].append(record)

    return providers

def parse_820(content: str) -> Dict[str, List[dict]]:
    records = []
    payees = set()
    current_qe_first = ""
    current_qe_last = ""
    current_qe_memberid = ""
    last_record = None

    for raw in content.split("~"):
        seg = raw.strip()
        if not seg:
            continue
        parts = seg.split("*")
        if len(parts) < 1:
            continue

        if parts[0] == "N1" and len(parts) >= 3 and parts[1] == "PE":
            payee_name = norm_str(parts[2])
            if payee_name:
                payees.add(payee_name)
        if parts[0] == "NM1" and len(parts) >= 4 and parts[1] == "PE":
            payee_name = norm_str(parts[3])
            if payee_name:
                payees.add(payee_name)

        if parts[0] == "NM1" and len(parts) >= 3 and parts[1] == "QE":
            if len(parts) >= 5:
                current_qe_last = norm_str(parts[3])
                current_qe_first = norm_str(parts[4])
            else:
                name_field = norm_str(parts[3]) if len(parts) >= 4 else ""
                if name_field:
                    tokens = name_field.split()
                    if len(tokens) >= 2:
                        current_qe_last = tokens[0]
                        current_qe_first = " ".join(tokens[1:])
                    else:
                        current_qe_last = name_field
                        current_qe_first = ""
            current_qe_memberid = norm_str(parts[-1]) if len(parts) >= 1 else ""

        if parts[0] == "RMR" and len(parts) >= 3 and parts[1] == "IK":
            claim_id = norm_str(parts[2])
            amount = ""
            for elem in parts[3:]:
                if re.match(r'^[\+\-]?\d+(?:\.\d+)?$', elem.strip()):
                    amount = elem.strip()
                    break

            rec = {
                "ClaimID": claim_id,
                "Amount": amount,
                "MemberID": current_qe_memberid,
                "FirstName": current_qe_first,
                "LastName": current_qe_last,
                "ADX_Code": "",
                "ADX_Description": ""
            }
            records.append(rec)
            last_record = rec

        if parts[0] == "ADX" and last_record is not None:
            reason_code = norm_str(parts[-1])
            last_record["ADX_Code"] = reason_code
            last_record["ADX_Description"] = ADX_CODES.get(reason_code, "")

    return {"payees": payees, "records": records}

# -------------------
# Comparison logic
# -------------------

def compare_providers_vs_820(providers_data: Dict[str, List[dict]], parsed_820_files: List[Tuple[str, dict]]):
    results = defaultdict(list)
    per_file_indexes = []

    for fname, parsed in parsed_820_files:
        claim_index = defaultdict(list)
        member_index = defaultdict(list)
        for r in parsed["records"]:
            cid = norm_str(r.get("ClaimID",""))
            mid = norm_str(r.get("MemberID",""))
            if cid:
                claim_index[cid].append(r)
            if mid:
                member_index[mid].append(r)
        per_file_indexes.append((fname, parsed["payees"], claim_index, member_index))

    for provider, claims in providers_data.items():
        for c in claims:
            p_first = c.get("FirstName","")
            p_last = c.get("LastName","")
            p_type = c.get("Type","")
            p_claimid = norm_str(c.get("ClaimID",""))
            p_memberid = norm_str(c.get("MemberID",""))
            p_amount = norm_str(c.get("Amount",""))

            matched = False
            matched_file = ""
            matched_rec = None
            status = "Member not found in 820"

            for fname, payees, claim_index, member_index in per_file_indexes:
                if p_type == "C" and p_claimid:
                    candidates = claim_index.get(p_claimid, [])
                    for cand in candidates:
                        if name_match(p_first, p_last, cand.get("FirstName",""), cand.get("LastName","")) and equal_amounts(p_amount, cand.get("Amount","")):
                            matched = True
                            matched_file = fname
                            matched_rec = cand
                            break
                    if matched:
                        status = "Match found"
                        break

                if p_type == "P" and p_memberid:
                    candidates = member_index.get(p_memberid, [])
                    for cand in candidates:
                        if name_match(p_first, p_last, cand.get("FirstName",""), cand.get("LastName","")) and equal_amounts(p_amount, cand.get("Amount","")):
                            matched = True
                            matched_file = fname
                            matched_rec = cand
                            break
                    if matched:
                        status = "Match found"
                        break

            row = {
                "Provider": provider,
                "FirstName": p_first,
                "LastName": p_last,
                "Type": p_type,
                "ClaimID": p_claimid,
                "MemberID": p_memberid,
                "Amount_837I": p_amount,
                "Matched_820_File": matched_file if matched else "",
                "Amount_820": matched_rec.get("Amount","") if matched_rec else "",
                "820_FirstName": matched_rec.get("FirstName","") if matched_rec else "",
                "820_LastName": matched_rec.get("LastName","") if matched_rec else "",
                "ADX_Code": matched_rec.get("ADX_Code","") if matched_rec else "",
                "ADX_Description": matched_rec.get("ADX_Description","") if matched_rec else "",
                "Status": status
            }
            results[provider].append(row)

    return results

# -------------------
# Excel writer
# -------------------

def save_excel(provider_results: Dict[str, List[dict]], basename: str, output_dir="output_reports"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_comparison.xlsx")

    provider_summary = []
    users_rows = []

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for provider, rows in provider_results.items():
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.drop_duplicates(subset=["FirstName","LastName","Type","ClaimID","MemberID","Amount_837I"])
            sheet_name = sanitize_sheet_name(provider)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            total_claims = df.shape[0]
            unique_users = sorted(set((r["LastName"], r["FirstName"]) for r in rows))
            provider_summary.append({"Provider": provider, "TotalClaims": total_claims, "UniqueUsersCount": len(unique_users)})

            for ln, fn in unique_users:
                cnt = df[(df["LastName"] == ln) & (df["FirstName"] == fn)].shape[0]
                users_rows.append({"Provider": provider, "UserLastName": ln, "UserFirstName": fn, "UserClaimCount": cnt})

        report_df = pd.DataFrame(provider_summary) if provider_summary else pd.DataFrame(columns=["Provider","TotalClaims","UniqueUsersCount"])
        users_df = pd.DataFrame(users_rows) if users_rows else pd.DataFrame(columns=["Provider","UserLastName","UserFirstName","UserClaimCount"])

        report_df.to_excel(writer, sheet_name="Report", index=False, startrow=0)
        startrow = len(report_df) + 3 if not report_df.empty else 3
        if not users_df.empty:
            ws = writer.sheets["Report"]
            ws.write(startrow - 1, 0, "Unique Users per Provider")
            users_df.to_excel(writer, sheet_name="Report", index=False, startrow=startrow)

    return out_path

# -------------------
# Streamlit UI
# -------------------

st.title("837I ↔ 820 Comparison (strict id/member + name + amount)")

with st.sidebar:
    st.header("Upload files")
    files_837 = st.file_uploader("Upload 837I files (multiple)", type=["txt","edi"], accept_multiple_files=True)
    files_820 = st.file_uploader("Upload 820 files (multiple)", type=["txt","edi"], accept_multiple_files=True)
    run = st.button("Run Comparison")

if run:
    if not files_837 or not files_820:
        st.error("Please upload at least one 837I file and one 820 file.")
    else:
        parsed_820 = []
        st.info("Parsing 820 files...")
        for f in files_820:
            txt = f.read().decode("utf-8", errors="ignore")
            parsed = parse_820(txt)
            parsed_820.append((f.name, parsed))
            st.write(f"- {f.name}: RMR rows = {len(parsed['records'])}; payees detected = {', '.join(sorted(parsed['payees'])) if parsed['payees'] else '(none)'}")

        saved_files = []
        st.info("Processing 837I files and comparing...")
        for f in files_837:
            txt = f.read().decode("utf-8", errors="ignore")
            providers = parse_837i(txt)

            with st.expander(f"837I preview: {f.name} (Providers & claims)"):
                for prov, rows in providers.items():
                    st.write(f"Provider: {prov} — claims: {len(rows)}")
                    if rows:
                        st.dataframe(pd.DataFrame(rows).head(30))

            results = compare_providers_vs_820(providers, parsed_820)
            outpath = save_excel(results, f.name, output_dir="output_reports")
            saved_files.append(outpath)

            provider_summary = []
            users_rows = []
            for prov, rows in results.items():
                df = pd.DataFrame(rows).drop_duplicates(subset=["FirstName","LastName","Type","ClaimID","MemberID","Amount_837I"]) if rows else pd.DataFrame()
                total_claims = df.shape[0]
                unique_users = sorted(set((r["LastName"], r["FirstName"]) for r in rows))
                provider_summary.append({"Provider": prov, "TotalClaims": total_claims, "UniqueUsersCount": len(unique_users)})
                for ln, fn in unique_users:
                    cnt = df[(df["LastName"] == ln) & (df["FirstName"] == fn)].shape[0]
                    users_rows.append({"Provider": prov, "UserLastName": ln, "UserFirstName": fn, "UserClaimCount": cnt})

            st.subheader(f"Report for {f.name}")
            if provider_summary:
                report_df = pd.DataFrame(provider_summary)
                st.dataframe(report_df, use_container_width=True)
            else:
                st.write("No providers/claims found in this 837I file.")

            if users_rows:
                st.write("Unique users per provider:")
                users_df = pd.DataFrame(users_rows)
                st.dataframe(users_df, use_container_width=True)

            with st.expander(f"Detailed comparisons for {f.name}"):
                for prov, rows in results.items():
                    st.write(f"Provider: {prov}")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        outdir = os.path.abspath("output_reports")
        st.success(f"Reports saved to folder: {outdir}")
        st.write("Saved files:")
        for p in saved_files:
            st.write(f"• {os.path.basename(p)}")














# app.py
import streamlit as st
import pandas as pd
import re
import os
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from typing import Dict, List, Tuple

st.set_page_config(page_title="837I ↔ 820 Comparison (Strict match)", layout="wide")

# -------------------
# Utilities
# -------------------
def norm_str(x):
    return (x or "").strip()

def sanitize_sheet_name(name: str) -> str:
    if not name:
        return "UNKNOWN"
    s = re.sub(r'[\[\]\*\?\/\\:]', "_", name)
    return s[:31]

def to_decimal(x):
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def equal_amounts(a, b):
    aa = to_decimal(a)
    bb = to_decimal(b)
    if aa is None or bb is None:
        return False
    return aa == bb

def name_match(first1, last1, first2, last2):
    return norm_str(first1).upper() == norm_str(first2).upper() and norm_str(last1).upper() == norm_str(last2).upper()

# -------------------
# Parsers
# -------------------

def parse_837i(content: str) -> Dict[str, List[dict]]:
    """
    Parse 837I content and return providers dict:
      provider_name -> list of claims dicts:
        {
          "Provider", "FirstName", "LastName", "Type"("C"|"P"), "ClaimID", "MemberID", "Amount"
        }
    """
    providers = defaultdict(list)
    current_provider = None
    current_first = ""
    current_last = ""

    for raw in content.split("~"):
        seg = raw.strip()
        if not seg:
            continue
        parts = seg.split("*")
        if len(parts) < 1:
            continue

        # Provider: NM1*41*...
        if parts[0] == "NM1" and len(parts) >= 4 and parts[1] == "41":
            current_provider = norm_str(parts[3])

        # Subscriber: NM1*IL*1*LAST*FIRST...
        elif parts[0] == "NM1" and len(parts) >= 5 and parts[1] == "IL":
            current_last = norm_str(parts[3])
            current_first = norm_str(parts[4])

        # Claim: CLM*<token>*<amount>...
        elif parts[0] == "CLM" and len(parts) >= 3:
            clm01 = parts[1]
            amt = norm_str(parts[2])
            claim_type = ""
            claim_id = ""
            member_id = ""

            # Patterns in your examples: "C-<claimid>-..." or "P-<memberid>-..."
            if clm01.startswith("C-"):
                claim_type = "C"
                m = re.match(r"^C-([^-\s]+)", clm01)
                if m:
                    claim_id = m.group(1)
            elif clm01.startswith("P-"):
                claim_type = "P"
                m = re.match(r"^P-([^-\s]+)", clm01)
                if m:
                    member_id = m.group(1)
                    # optional normalization: drop trailing single uppercase letter (if present and desirable)
                    if len(member_id) > 1 and re.match(r".*[A-Z]$", member_id) and re.search(r"\d", member_id[:-1]):
                        # only strip if it ends with a letter and there is a digit earlier (safe heuristic)
                        member_id = member_id[:-1]

            if current_provider is None:
                current_provider = "UNKNOWN_PROVIDER"

            record = {
                "Provider": current_provider,
                "FirstName": current_first,
                "LastName": current_last,
                "Type": claim_type,
                "ClaimID": claim_id,
                "MemberID": member_id,
                "Amount": amt
            }
            providers[current_provider].append(record)

    return providers

def parse_820(content: str) -> Dict[str, List[dict]]:
    """
    Parse 820 content to extract RMR*IK rows and associate them with nearest prior NM1*QE (if any).
    Returns:
      {
        "payees": set(payee names),
        "records": [ {"ClaimID":..., "Amount":..., "MemberID":..., "FirstName":..., "LastName":...}, ... ]
      }
    """
    records = []
    payees = set()
    current_qe_first = ""
    current_qe_last = ""
    current_qe_memberid = ""

    for raw in content.split("~"):
        seg = raw.strip()
        if not seg:
            continue
        parts = seg.split("*")
        if len(parts) < 1:
            continue

        # N1*PE or NM1*PE can contain payee name (optional)
        if parts[0] == "N1" and len(parts) >= 3 and parts[1] == "PE":
            payee_name = norm_str(parts[2])
            if payee_name:
                payees.add(payee_name)
        if parts[0] == "NM1" and len(parts) >= 4 and parts[1] == "PE":
            # NM1*PE*2*PAYEENAME...
            payee_name = norm_str(parts[3])
            if payee_name:
                payees.add(payee_name)

        # NM1*QE sets context for subsequent RMR rows; last element is often member id
        if parts[0] == "NM1" and len(parts) >= 3 and parts[1] == "QE":
            # Typical: NM1*QE*1*LAST*FIRST*...*N*<memberid>
            # We'll try to set last and first with available fields
            if len(parts) >= 5:
                current_qe_last = norm_str(parts[3])
                current_qe_first = norm_str(parts[4])
            else:
                # fallback: try to split the single name field into last/first
                name_field = norm_str(parts[3]) if len(parts) >= 4 else ""
                if name_field:
                    # try splitting by space: last is first token, rest first name
                    tokens = name_field.split()
                    if len(tokens) >= 2:
                        current_qe_last = tokens[0]
                        current_qe_first = " ".join(tokens[1:])
                    else:
                        current_qe_last = name_field
                        current_qe_first = ""
            # The member id is often last element
            current_qe_memberid = norm_str(parts[-1]) if len(parts) >= 1 else ""

        # RMR*IK core row
        if parts[0] == "RMR" and len(parts) >= 3 and parts[1] == "IK":
            claim_id = norm_str(parts[2])  # may be blank for P-premium submissions
            amount = ""
            # find first numeric-like element after p[2]
            for elem in parts[3:]:
                if re.match(r'^[\+\-]?\d+(?:\.\d+)?$', elem.strip()):
                    amount = elem.strip()
                    break

            rec = {
                "ClaimID": claim_id,
                "Amount": amount,
                "MemberID": current_qe_memberid,
                "FirstName": current_qe_first,
                "LastName": current_qe_last
            }
            records.append(rec)

    return {"payees": payees, "records": records}

# -------------------
# Comparison logic (strict: id/member AND names AND amount)
# -------------------

def compare_providers_vs_820(providers_data: Dict[str, List[dict]], parsed_820_files: List[Tuple[str, dict]]):
    """
    providers_data: provider -> list of claim dicts (from parse_837i)
    parsed_820_files: list of (filename, parsed_dict) where parsed_dict = parse_820(...)
    Returns: provider -> list of result rows (dictionaries)
    """
    results = defaultdict(list)

    # Build per-file indexes (do not merge files; we need filename where match found)
    per_file_indexes = []
    for fname, parsed in parsed_820_files:
        claim_index = defaultdict(list)
        member_index = defaultdict(list)
        # Also store name-indexed records for strict checking if needed
        for r in parsed["records"]:
            cid = norm_str(r.get("ClaimID",""))
            mid = norm_str(r.get("MemberID",""))
            amt = norm_str(r.get("Amount",""))
            # index by claim id (if present)
            if cid:
                claim_index[cid].append(r)
            # index by member id (if present)
            if mid:
                member_index[mid].append(r)
        per_file_indexes.append((fname, parsed["payees"], claim_index, member_index))

    # Iterate every provider claim and attempt find first match across 820 files
    for provider, claims in providers_data.items():
        for c in claims:
            p_first = c.get("FirstName","")
            p_last = c.get("LastName","")
            p_type = c.get("Type","")
            p_claimid = norm_str(c.get("ClaimID",""))
            p_memberid = norm_str(c.get("MemberID",""))
            p_amount = norm_str(c.get("Amount",""))

            matched = False
            matched_file = ""
            matched_rec = None
            status = "Member not found in 820"

            for fname, payees, claim_index, member_index in per_file_indexes:
                # For C-type: match ClaimID presence AND names AND amount
                if p_type == "C" and p_claimid:
                    candidates = claim_index.get(p_claimid, [])
                    for cand in candidates:
                        # require names and amount to match
                        if name_match(p_first, p_last, cand.get("FirstName",""), cand.get("LastName","")) and equal_amounts(p_amount, cand.get("Amount","")):
                            matched = True
                            matched_file = fname
                            matched_rec = cand
                            break
                    if matched:
                        status = "Match found"
                        break

                # For P-type: match MemberID and names and amount
                if p_type == "P" and p_memberid:
                    candidates = member_index.get(p_memberid, [])
                    for cand in candidates:
                        if name_match(p_first, p_last, cand.get("FirstName",""), cand.get("LastName","")) and equal_amounts(p_amount, cand.get("Amount","")):
                            matched = True
                            matched_file = fname
                            matched_rec = cand
                            break
                    if matched:
                        status = "Match found"
                        break

                # If neither candidate or types missing, continue to next 820 file

            # Build output row
            row = {
                "Provider": provider,
                "FirstName": p_first,
                "LastName": p_last,
                "Type": p_type,
                "ClaimID": p_claimid,
                "MemberID": p_memberid,
                "Amount_837I": p_amount,
                "Matched_820_File": matched_file if matched else "",
                "Amount_820": matched_rec.get("Amount","") if matched_rec else "",
                "820_FirstName": matched_rec.get("FirstName","") if matched_rec else "",
                "820_LastName": matched_rec.get("LastName","") if matched_rec else "",
                "Status": status
            }
            results[provider].append(row)

    return results

# -------------------
# Excel writer (saves to output_reports/)
# -------------------

def save_excel(provider_results: Dict[str, List[dict]], basename: str, output_dir="output_reports"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_comparison.xlsx")

    provider_summary = []
    users_rows = []

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for provider, rows in provider_results.items():
            df = pd.DataFrame(rows)
            # Dedupe subscriber duplicate rows by key columns
            if not df.empty:
                df = df.drop_duplicates(subset=["FirstName","LastName","Type","ClaimID","MemberID","Amount_837I"])
            sheet_name = sanitize_sheet_name(provider)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            total_claims = df.shape[0]
            unique_users = sorted(set((r["LastName"], r["FirstName"]) for r in rows))
            provider_summary.append({"Provider": provider, "TotalClaims": total_claims, "UniqueUsersCount": len(unique_users)})

            for ln, fn in unique_users:
                cnt = df[(df["LastName"] == ln) & (df["FirstName"] == fn)].shape[0]
                users_rows.append({"Provider": provider, "UserLastName": ln, "UserFirstName": fn, "UserClaimCount": cnt})

        # Write Report sheet
        report_df = pd.DataFrame(provider_summary) if provider_summary else pd.DataFrame(columns=["Provider","TotalClaims","UniqueUsersCount"])
        users_df = pd.DataFrame(users_rows) if users_rows else pd.DataFrame(columns=["Provider","UserLastName","UserFirstName","UserClaimCount"])

        report_df.to_excel(writer, sheet_name="Report", index=False, startrow=0)
        # Write users table below
        startrow = len(report_df) + 3 if not report_df.empty else 3
        if not users_df.empty:
            ws = writer.sheets["Report"]
            ws.write(startrow - 1, 0, "Unique Users per Provider")
            users_df.to_excel(writer, sheet_name="Report", index=False, startrow=startrow)

    return out_path

# -------------------
# Streamlit UI
# -------------------

st.title("837I ↔ 820 Comparison (strict id/member + name + amount)")

with st.sidebar:
    st.header("Upload files")
    files_837 = st.file_uploader("Upload 837I files (multiple)", type=["txt","edi"], accept_multiple_files=True)
    files_820 = st.file_uploader("Upload 820 files (multiple)", type=["txt","edi"], accept_multiple_files=True)
    run = st.button("Run Comparison")

if run:
    if not files_837 or not files_820:
        st.error("Please upload at least one 837I file and one 820 file.")
    else:
        # Read & parse all 820 files (preserve their order)
        parsed_820 = []
        st.info("Parsing 820 files...")
        for f in files_820:
            txt = f.read().decode("utf-8", errors="ignore")
            parsed = parse_820(txt)
            parsed_820.append((f.name, parsed))
            st.write(f"- {f.name}: RMR rows = {len(parsed['records'])}; payees detected = {', '.join(sorted(parsed['payees'])) if parsed['payees'] else '(none)'}")

        saved_files = []
        st.info("Processing 837I files and comparing...")
        for f in files_837:
            txt = f.read().decode("utf-8", errors="ignore")
            providers = parse_837i(txt)

            # preview
            with st.expander(f"837I preview: {f.name} (Providers & claims)"):
                for prov, rows in providers.items():
                    st.write(f"Provider: {prov} — claims: {len(rows)}")
                    if rows:
                        st.dataframe(pd.DataFrame(rows).head(30))

            # compare
            results = compare_providers_vs_820(providers, parsed_820)

            # Save excel
            outpath = save_excel(results, f.name, output_dir="output_reports")
            saved_files.append(outpath)

            # Show Report sheet in UI
            # Recreate Report and Users tables same as save_excel does
            provider_summary = []
            users_rows = []
            for prov, rows in results.items():
                df = pd.DataFrame(rows).drop_duplicates(subset=["FirstName","LastName","Type","ClaimID","MemberID","Amount_837I"]) if rows else pd.DataFrame()
                total_claims = df.shape[0]
                unique_users = sorted(set((r["LastName"], r["FirstName"]) for r in rows))
                provider_summary.append({"Provider": prov, "TotalClaims": total_claims, "UniqueUsersCount": len(unique_users)})
                for ln, fn in unique_users:
                    cnt = df[(df["LastName"] == ln) & (df["FirstName"] == fn)].shape[0]
                    users_rows.append({"Provider": prov, "UserLastName": ln, "UserFirstName": fn, "UserClaimCount": cnt})

            st.subheader(f"Report for {f.name}")
            if provider_summary:
                report_df = pd.DataFrame(provider_summary)
                st.dataframe(report_df, use_container_width=True)
            else:
                st.write("No providers/claims found in this 837I file.")

            if users_rows:
                st.write("Unique users per provider:")
                users_df = pd.DataFrame(users_rows)
                st.dataframe(users_df, use_container_width=True)

            # Optionally show detailed comparisons
            with st.expander(f"Detailed comparisons for {f.name}"):
                for prov, rows in results.items():
                    st.write(f"Provider: {prov}")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Final message
        outdir = os.path.abspath("output_reports")
        st.success(f"Reports saved to folder: {outdir}")
        st.write("Saved files:")
        for p in saved_files:
            st.write(f"• {os.path.basename(p)}")











import os
import pandas as pd

def create_excel(provider_results, filename):
    # Ensure output folder exists
    output_folder = "output_reports"
    os.makedirs(output_folder, exist_ok=True)

    # Build output file path
    file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_results.xlsx")

    total_claims = 0
    summary_rows = []

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for provider, claims in provider_results.items():
            df = pd.DataFrame(claims)
            df.to_excel(writer, sheet_name=provider[:31], index=False)
            total_claims += len(claims)

            unique_users = len(set([f"{c['firstname']} {c['lastname']}" for c in claims]))
            summary_rows.append({
                "Provider": provider,
                "Total Claims": len(claims),
                "Unique Users": unique_users
            })

        # Summary sheet
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Report", index=False)

    return file_path













import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO
from collections import defaultdict

# ---------- Parsing Functions ----------

def parse_837i(file_content):
    """Parse 837I EDI content into structured data grouped by provider."""
    providers_data = defaultdict(list)
    current_provider = None
    current_subscriber = None

    lines = file_content.split("~")
    for line in lines:
        segments = line.strip().split("*")

        if segments[0] == "NM1" and segments[1] == "41":
            current_provider = segments[2] if len(segments) > 2 else "UNKNOWN_PROVIDER"

        if segments[0] == "NM1" and segments[1] == "IL":
            current_subscriber = {
                "firstname": segments[4] if len(segments) > 4 else "",
                "lastname": segments[3] if len(segments) > 3 else "",
                "claim_id": "",
                "member_id": "",
                "claim_amount": "",
                "type": ""
            }

        if segments[0] == "CLM":
            claim_type = segments[1].split("-")[0][-1]  # 'C' or 'P'
            if claim_type.upper() == "C":
                claim_id = segments[1].split("-")[1]
                amount = segments[2]
                current_subscriber["claim_id"] = claim_id
                current_subscriber["type"] = "C"
                current_subscriber["claim_amount"] = amount
            elif claim_type.upper() == "P":
                member_id = segments[1].split("-")[1]
                amount = segments[2]
                current_subscriber["member_id"] = member_id
                current_subscriber["type"] = "P"
                current_subscriber["claim_amount"] = amount
            providers_data[current_provider].append(current_subscriber)

    return providers_data


def parse_820(file_content):
    """Parse 820 EDI content into structured data."""
    records = []
    lines = file_content.split("~")
    current_payee = None
    current_member_id = None
    current_claim_id = None
    current_amount = None

    for line in lines:
        segments = line.strip().split("*")

        if segments[0] == "NM1" and segments[1] == "QE":
            current_payee = segments[3] + " " + segments[4] if len(segments) > 4 else segments[3]
            current_member_id = segments[-1]

        if segments[0] == "RMR" and segments[1] == "IK":
            current_claim_id = segments[2]

        if segments[0] == "AMT" and segments[1] == "R8":
            current_amount = segments[2]
            records.append({
                "payee": current_payee.strip(),
                "member_id": current_member_id.strip(),
                "claim_id": current_claim_id.strip() if current_claim_id else "",
                "amount": current_amount.strip()
            })
            current_claim_id = None
            current_amount = None

    return records

# ---------- Comparison Logic ----------

def compare_837i_820(providers_data, all_820_data):
    """Compare 837I providers data with all 820 data."""
    result_per_provider = defaultdict(list)

    for provider, claims in providers_data.items():
        for claim in claims:
            match_found = False
            found_in_file = None

            for fname, records in all_820_data.items():
                for rec in records:
                    if claim["type"] == "C" and claim["claim_id"] and rec["claim_id"] == claim["claim_id"]:
                        match_found = True
                        found_in_file = fname
                        break
                    elif claim["type"] == "P" and claim["member_id"] and rec["member_id"] == claim["member_id"]:
                        match_found = True
                        found_in_file = fname
                        break
                if match_found:
                    break

            claim_result = claim.copy()
            claim_result["found_in"] = found_in_file if match_found else "Member not found in 820"
            result_per_provider[provider].append(claim_result)

    return result_per_provider

# ---------- Excel Export ----------

def create_excel(provider_results, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        total_claims = 0
        summary_rows = []
        
        for provider, claims in provider_results.items():
            df = pd.DataFrame(claims)
            df.to_excel(writer, sheet_name=provider[:31], index=False)
            total_claims += len(claims)
            unique_users = len(set([f"{c['firstname']} {c['lastname']}" for c in claims]))
            summary_rows.append({
                "Provider": provider,
                "Total Claims": len(claims),
                "Unique Users": unique_users
            })
        
        # Summary sheet
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Report", index=False)

    return output.getvalue()

# ---------- Streamlit App ----------

st.title("837I vs 820 Claims Comparison")

st.sidebar.header("Upload EDI Files")
uploaded_837i_files = st.sidebar.file_uploader("Upload 837I files", type=["txt"], accept_multiple_files=True)
uploaded_820_files = st.sidebar.file_uploader("Upload 820 files", type=["txt"], accept_multiple_files=True)

if uploaded_837i_files and uploaded_820_files:
    # Parse all 820 first
    all_820_data = {}
    for file in uploaded_820_files:
        content = file.read().decode("utf-8")
        all_820_data[file.name] = parse_820(content)

    # Process each 837I and compare
    for file in uploaded_837i_files:
        content = file.read().decode("utf-8")
        providers_data = parse_837i(content)
        provider_results = compare_837i_820(providers_data, all_820_data)
        excel_data = create_excel(provider_results, file.name)
        
        st.download_button(
            label=f"Download Excel for {file.name}",
            data=excel_data,
            file_name=f"{os.path.splitext(file.name)[0]}_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )









import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO

st.set_page_config(page_title="837I vs 820 Claim Comparison", layout="wide")

# --- Helper functions to parse 837I ---
def parse_837i(content):
    claims = []
    segments = content.split("~")
    
    first_name = last_name = claim_id = claim_amount = None
    
    for seg in segments:
        if seg.startswith("NM1*IL"):
            parts = seg.split("*")
            last_name = parts[3] if len(parts) > 3 else ""
            first_name = parts[4] if len(parts) > 4 else ""
        
        elif seg.startswith("CLM"):
            parts = seg.split("*")
            claim_id = parts[1] if len(parts) > 1 else ""
            claim_amount = parts[2] if len(parts) > 2 else ""
            
            if first_name and last_name and claim_id and claim_amount:
                claims.append({
                    "FirstName": first_name,
                    "LastName": last_name,
                    "ClaimID": claim_id,
                    "ClaimAmount": claim_amount
                })
                claim_id = claim_amount = None  # reset for next claim
                
    return pd.DataFrame(claims)

# --- Helper functions to parse 820 ---
def parse_820(content):
    payments = []
    segments = content.split("~")
    
    first_name = last_name = claim_id = claim_amount = None
    
    for seg in segments:
        if seg.startswith("NM1*QE"):
            parts = seg.split("*")
            last_name = parts[3] if len(parts) > 3 else ""
            first_name = parts[4] if len(parts) > 4 else ""
        
        elif seg.startswith("RMR*IK"):
            parts = seg.split("*")
            claim_id = parts[2] if len(parts) > 2 else ""
        
        elif seg.startswith("AMT*"):
            parts = seg.split("*")
            if parts[0] == "AMT" or parts[0].startswith("AMT"):
                claim_amount = parts[2] if len(parts) > 2 else ""
            
            if first_name and last_name and claim_id and claim_amount:
                payments.append({
                    "FirstName": first_name,
                    "LastName": last_name,
                    "ClaimID": claim_id,
                    "ClaimAmount": claim_amount
                })
                claim_id = claim_amount = None
                
    return pd.DataFrame(payments)

# --- Comparison logic ---
def compare_claims(df837, df820):
    results = []
    for _, row in df837.iterrows():
        match = df820[
            (df820["FirstName"].str.lower() == row["FirstName"].lower()) &
            (df820["LastName"].str.lower() == row["LastName"].lower()) &
            (df820["ClaimID"].str.strip() == row["ClaimID"].strip()) &
            (df820["ClaimAmount"].astype(str).str.strip() == str(row["ClaimAmount"]).strip())
        ]
        if match.empty:
            status = "Member not found in 820"
        else:
            status = "Match found"
        
        result_row = row.to_dict()
        result_row["Status"] = status
        results.append(result_row)
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.sidebar.header("Upload EDI Files")
file_837 = st.sidebar.file_uploader("Upload 837I EDI File", type=["txt", "edi"])
file_820 = st.sidebar.file_uploader("Upload 820 EDI File", type=["txt", "edi"])

if file_837 and file_820:
    content_837 = file_837.read().decode("utf-8", errors="ignore")
    content_820 = file_820.read().decode("utf-8", errors="ignore")
    
    df837 = parse_837i(content_837)
    df820 = parse_820(content_820)
    
    st.write("### Parsed 837I Claims")
    st.dataframe(df837)
    st.write("### Parsed 820 Claims")
    st.dataframe(df820)
    
    comparison_df = compare_claims(df837, df820)
    st.write("### Comparison Results")
    st.dataframe(comparison_df)
    
    # Create Excel with one sheet per subscriber
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, group in comparison_df.groupby(["FirstName", "LastName"]):
            sheet_name = f"{name[0]}_{name[1]}"[:31]  # Excel sheet name limit
            group.to_excel(writer, index=False, sheet_name=sheet_name)
    
    st.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name="837I_vs_820_Comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )








# app.py
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import tempfile
import json

# ---------------------------
# Streamlit sidebar inputs
# ---------------------------
st.set_page_config(page_title="EDI Companion Guide Rule Ingestion", layout="wide")
st.sidebar.header("Upload Companion Guide")

edi_type = st.sidebar.selectbox("EDI Type", ["837i", "820"])
uploaded_file = st.sidebar.file_uploader("Upload Companion Guide (PDF)", type=["pdf"])
test_query = st.sidebar.text_input("Test retrieval query")
process_button = st.sidebar.button("Extract & Store Rules")

# ---------------------------
# Azure settings from env/secrets
# ---------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", st.secrets.get("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", st.secrets.get("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", st.secrets.get("AZURE_OPENAI_API_VERSION"))
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", st.secrets.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", st.secrets.get("AZURE_OPENAI_GPT_DEPLOYMENT"))

# ---------------------------
# LangChain helpers
# ---------------------------
def get_embeddings():
    return AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model="text-embedding-3-large",
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_OPENAI_API_VERSION
    )

def get_llm():
    return AzureChatOpenAI(
        deployment_name=CHAT_DEPLOYMENT,
        temperature=0,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_OPENAI_API_VERSION
    )

def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)
    return docs

def chunk_documents(documents, chunk_size=1200, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def save_faiss_index(vectorstore, edi_type):
    save_path = f"vectorstores/{edi_type}"
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

def load_faiss_index(edi_type):
    path = f"vectorstores/{edi_type}"
    if os.path.exists(path):
        return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
    return None

def extract_rules_from_chunk(text, edi_type):
    llm = get_llm()
    prompt = f"""
You are an expert in EDI {edi_type} companion guides.
Given the following guide content, extract ALL segment rules in JSON format.
Each rule must have:
- segment (like "NM1*IL")
- description (1 short sentence)
- fields (list of objects with field_name and description)

Content:
{text}

Return ONLY JSON array, no extra text.
"""
    response = llm([HumanMessage(content=prompt)])
    try:
        rules = json.loads(response.content)
        return rules
    except Exception:
        return []

# ---------------------------
# Processing
# ---------------------------
if process_button and uploaded_file:
    st.write(f"Extracting rules from {edi_type} companion guide...")

    docs = load_pdf(uploaded_file)
    chunks = chunk_documents(docs)

    all_rules_texts = []
    for chunk in chunks:
        rules = extract_rules_from_chunk(chunk.page_content, edi_type)
        for r in rules:
            r_text = json.dumps(r, ensure_ascii=False)
            all_rules_texts.append(r_text)

    if not all_rules_texts:
        st.error("No rules extracted. Check guide format or LLM prompt.")
    else:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_texts(all_rules_texts, embeddings, docstore=InMemoryDocstore({}))
        save_faiss_index(vectorstore, edi_type)
        st.success(f"Extracted {len(all_rules_texts)} rules and saved FAISS index for {edi_type}.")

# ---------------------------
# Test retrieval
# ---------------------------
if test_query:
    vs = load_faiss_index(edi_type)
    if vs:
        results = vs.similarity_search(test_query, k=3)
        st.write("### Top matching rules from FAISS:")
        for i, r in enumerate(results, 1):
            try:
                parsed = json.loads(r.page_content)
                st.json(parsed)
            except:
                st.text(r.page_content)
    else:
        st.error(f"No FAISS index found for {edi_type}. Please extract rules first.")
















# app.py
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import tempfile
import shutil

# ---------------------------
# Streamlit sidebar inputs
# ---------------------------
st.set_page_config(page_title="EDI Companion Guide Ingestion", layout="wide")
st.sidebar.header("Upload Companion Guide")

edi_type = st.sidebar.selectbox("EDI Type", ["837i", "820"])
uploaded_file = st.sidebar.file_uploader("Upload Companion Guide (PDF)", type=["pdf"])
test_query = st.sidebar.text_input("Test retrieval query")
process_button = st.sidebar.button("Process Guide")

# ---------------------------
# Azure settings from env/secrets
# ---------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", st.secrets.get("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", st.secrets.get("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", st.secrets.get("AZURE_OPENAI_API_VERSION"))
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", st.secrets.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", st.secrets.get("AZURE_OPENAI_GPT_DEPLOYMENT"))

# ---------------------------
# Helper functions
# ---------------------------
def get_embeddings():
    return AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model="text-embedding-3-large",
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_OPENAI_API_VERSION
    )

def get_llm():
    return AzureChatOpenAI(
        deployment_name=CHAT_DEPLOYMENT,
        temperature=0,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_OPENAI_API_VERSION
    )

def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def save_faiss_index(vectorstore, edi_type):
    save_path = f"vectorstores/{edi_type}"
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

def load_faiss_index(edi_type):
    path = f"vectorstores/{edi_type}"
    if os.path.exists(path):
        return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
    return None

# ---------------------------
# Processing
# ---------------------------
if process_button and uploaded_file:
    st.write(f"Processing {edi_type} companion guide...")

    # Load & chunk
    docs = load_pdf(uploaded_file)
    chunks = chunk_documents(docs)

    # Create vectorstore
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings, docstore=InMemoryDocstore({}))

    # Save index
    save_faiss_index(vectorstore, edi_type)
    st.success(f"FAISS index for {edi_type} saved successfully.")

# ---------------------------
# Test retrieval
# ---------------------------
if test_query:
    vs = load_faiss_index(edi_type)
    if vs:
        results = vs.similarity_search(test_query, k=3)
        st.write("### Top matches from FAISS index:")
        for i, r in enumerate(results, 1):
            st.markdown(f"**Result {i}:**\n```\n{r.page_content}\n```")
    else:
        st.error(f"No FAISS index found for {edi_type}. Please process a guide first.")
















import pandas as pd
import json

def extract_bold_field_name(description):
    """Extract field name from bold markdown text like '**ISA01 - Authorization Info Qualifier**'."""
    import re
    match = re.search(r"\*\*(.*?)\*\*", description)
    if match:
        return match.group(1).strip()
    return None

def export_output(data, base_name):
    # Group rows by segment
    segment_rows = {}

    for item in data:
        segment = item["segment"]
        fields = item["fields"]

        row = {}
        for field_desc, field_value in fields:
            field_name = extract_bold_field_name(field_desc)
            if field_name:
                row[field_name] = field_value

        # Append the row to the corresponding segment's list
        segment_rows.setdefault(segment, []).append(row)

    # Write to Excel with multiple sheets
    excel_file = f"{base_name}.xlsx"
    with pd.ExcelWriter(excel_file) as writer:
        for segment, rows in segment_rows.items():
            df = pd.DataFrame(rows)
            # Sheet names can't be more than 31 characters or contain some symbols
            safe_segment = segment[:31].replace('/', '_').replace('\\', '_')
            df.to_excel(writer, sheet_name=safe_segment, index=False)

    # Write full JSON as backup
    json_file = f"{base_name}.json"
    with open(json_file, "w") as jf:
        json.dump(data, jf, indent=4)

















def enrich_with_descriptions(segments_dict, chain):
    enriched_segments = {}

    for segment, data in segments_dict.items():
        # Join all field names together and send one LLM call per segment
        fields = data.get("fields", [])
        
        # Prepare input prompt: ask for description of all fields at once
        joined_fields = ", ".join(fields)
        prompt = f"Describe the following fields for the EDI segment '{segment}': {joined_fields}"

        # Trigger LLM ONCE per segment
        enriched_text = chain.run(prompt)

        # Split back the enriched descriptions intelligently (you can refine this logic)
        enriched_fields = enriched_text.split(",") if "," in enriched_text else [enriched_text]

        # Map each original field with its enriched description (fallback if count mismatch)
        descriptions = {}
        for i, field in enumerate(fields):
            desc = enriched_fields[i].strip() if i < len(enriched_fields) else ""
            descriptions[field] = desc

        # Store back in result
        enriched_segments[segment] = {
            "fields": fields,
            "descriptions": descriptions
        }

    return enriched_segments













def export_output(data_rows, output_path="out/output_segments.xlsx"):
    """
    Export rows to separate Excel sheets based on segment extracted by `extract_bold_field_name`.
    """
    from collections import defaultdict
    import pandas as pd
    import os

    segment_sheets = defaultdict(list)

    for row in data_rows:
        for field in row:
            segment = extract_bold_field_name(field)
            segment_sheets[segment].append(row)
            break  # only consider the first field for segment grouping

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for segment, rows in segment_sheets.items():
            df = pd.DataFrame(rows)
            safe_segment = segment[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=safe_segment, index=False)

    print(f"Exported to {output_path}")


def extract_bold_field_name(text):
    """
    Extract bold field names enclosed in '**' and return the segment.
    If multiple '**' are found, use the last one.
    Fallback to extracting segment from text like 'ENT01 - something'
    """
    import re
    matches = re.findall(r'\*\*(.*?)\*\*', text)
    if matches:
        segment_name = matches[-1].strip().split()[0]  # Use last match and take first word
    else:
        segment_name = text.strip().split()[0]  # Fallback, take first word (like 'ENT01' → 'ENT')

    return segment_name.upper()














import re
import pandas as pd
import json

def extract_bold_field_name(text):
    """Extract the **bold** part of the field description."""
    match = re.search(r"\*\*(.*?)\*\*", text)
    return match.group(1).strip() if match else None

def export_output(data, base_name):
    rows = []
    for item in data:
        row = {"Segment": item["segment"]}
        for field_desc, field_value in item["fields"]:
            field_name = extract_bold_field_name(field_desc)
            if field_name:
                row[field_name] = field_value
        rows.append(row)

    df = pd.DataFrame(rows)
    excel_file = f"{base_name}.xlsx"
    json_file = f"{base_name}.json"

    df.to_excel(excel_file, index=False)
    
    # Full JSON with descriptions retained
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    return excel_file, json_file













import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import pandas as pd
import json
import os

# === OCR FUNCTION FOR PDF ===
def extract_text_from_pdf_with_ocr(pdf_path, poppler_path=None):
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    docs = []
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img)
            page_text = f"\n[Page {i+1}]\n{text.strip()}"
            docs.append(Document(page_content=page_text))
        except Exception as e:
            print(f"OCR failed on page {i+1}: {e}")
    return docs

# === VECTOR INDEX ===
def create_vector_index(docs, index_path):
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(index_path)
    return db

# === LOAD VECTOR INDEX + QA CHAIN ===
def get_retrieval_qa_chain(index_path, azure_api_key, azure_endpoint):
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    db = FAISS.load_local(index_path, embedding_model)
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name="gpt-4",
        api_version="2023-05-15",
        temperature=0
    )
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

# === PARSE RAW EDI ===
def parse_edi_file(edi_text):
    segments = edi_text.split("~")
    data = []
    for seg in segments:
        parts = seg.strip().split("*")
        if len(parts) > 1:
            data.append({"segment": parts[0], "fields": parts[1:]})
    return data

# === LLM-ASSISTED SEGMENT + FIELD ANNOTATION ===
def enrich_with_descriptions(data, chain):
    enriched = []
    for entry in data:
        segment = entry["segment"]
        fields = entry["fields"]
        try:
            seg_question = f"In an EDI file, what does the segment '{segment}' represent?"
            seg_description = chain.run(seg_question)
        except Exception:
            seg_description = "Not available"
        
        field_names = []
        for idx, field in enumerate(fields):
            try:
                field_question = f"In the EDI segment '{segment}', what does field number {idx + 1} represent?"
                field_description = chain.run(field_question)
            except Exception:
                field_description = f"Field{idx+1}"
            field_names.append((field_description, field))
        
        enriched.append({
            "segment": segment,
            "description": seg_description,
            "fields": field_names
        })
    return enriched

# === EXPORT TO EXCEL + JSON ===
def export_output(data, base_name):
    rows = []
    for item in data:
        row = {
            "Segment": item["segment"],
            "Description": item["description"]
        }
        for desc, val in item["fields"]:
            row[desc] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    excel_file = f"{base_name}.xlsx"
    json_file = f"{base_name}.json"
    df.to_excel(excel_file, index=False)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    return excel_file, json_file


# ===================
# ==== STREAMLIT ====
# ===================

st.set_page_config(page_title="EDI Data Extractor", layout="wide")
st.title("📑 Intelligent EDI Data Extractor")

# === SIDEBAR CONFIG ===
st.sidebar.header("⚙️ Configuration")
edi_type = st.sidebar.selectbox("Select EDI Format", ["837I", "820"])
pdf_file = st.sidebar.file_uploader("Upload Companion Guide PDF", type=["pdf"], key="pdf")
generate_index = st.sidebar.button("🔁 Generate/Update Vector Index")

azure_key = st.sidebar.text_input("Azure OpenAI API Key", type="password")
azure_endpoint = st.sidebar.text_input("Azure OpenAI Endpoint")

# === INDEX CHECK ===
index_dir = "indices"
index_path = os.path.join(index_dir, f"{edi_type}_index")
index_exists = os.path.exists(os.path.join(index_path, "index.faiss"))

# === GENERATE INDEX IF NEEDED ===
if generate_index:
    if not pdf_file:
        st.sidebar.warning("Please upload a companion guide PDF.")
    else:
        os.makedirs(index_dir, exist_ok=True)
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        with st.spinner("Running OCR and creating vector index..."):
            docs = extract_text_from_pdf_with_ocr("temp.pdf")
            create_vector_index(docs, index_path)
        st.sidebar.success("Vector index created successfully.")
        index_exists = True

# === MAIN APP AREA ===
st.subheader(f"📂 Upload and Extract from {edi_type} EDI File")
edi_file = st.file_uploader("Upload EDI File (.txt)", type=["txt"], key="edi")
extract_button = st.button("📤 Extract Data to Excel")

if extract_button:
    if not edi_file:
        st.error("Please upload an EDI file.")
    elif not azure_key or not azure_endpoint:
        st.error("Please provide Azure API Key and Endpoint in the sidebar.")
    elif not index_exists:
        st.warning(f"No index found for {edi_type}. Please upload PDF and generate the index from sidebar.")
    else:
        with st.spinner("Processing..."):
            edi_text = edi_file.read().decode("utf-8")
            segments = parse_edi_file(edi_text)
            chain = get_retrieval_qa_chain(index_path, azure_key, azure_endpoint)
            enriched = enrich_with_descriptions(segments, chain)
            xls_file, json_file = export_output(enriched, edi_type)
        st.success("Extraction complete!")
        with open(xls_file, "rb") as f:
            st.download_button("📥 Download Excel", f, file_name=xls_file)
        with open(json_file, "rb") as f:
            st.download_button("📥 Download JSON", f, file_name=json_file)













def process_edi_files(edi_files):
    import json

    # Load Cached Guide Data
    with open("cached_guide_segments.json", "r") as f:
        guide_data = json.load(f)

    # --- FIX 1: Flatten Nested Lists in guide_data ---
    flattened_guide_data = []
    for item in guide_data:
        if isinstance(item, list):
            flattened_guide_data.extend(item)  # Unpack nested list
        elif isinstance(item, dict):
            flattened_guide_data.append(item)  # Keep dict entries
    guide_data = flattened_guide_data

    # --- Main Processing Loop ---
    extracted_claims = []

    for edi_file in edi_files:
        content = edi_file.read().decode('utf-8', errors='ignore')
        segments = content.split('~')

        current_claim = {}
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue  # Skip empty lines

            seg_id = segment.split('*')[0].strip()

            # --- FIX 2: Safe Segment Lookup ---
            guide_segment = next(
                (item for item in guide_data if isinstance(item, dict) and item.get('segment_id') == seg_id),
                None
            )

            # Debugging
            st.write(f"Processing Segment ID: {seg_id}")
            if guide_segment:
                st.write(f"Matched Guide Segment: {guide_segment['segment_id']}")
            else:
                st.warning(f"No mapping found for segment: {seg_id}")
                continue

            # Extract Data using LLM
            extracted = extract_data_from_segment(segment, guide_segment)

            # Update Current Claim Dict
            current_claim.update(extracted)

            # Detect End of Claim (For Example: CLM Segment starts a new claim)
            if seg_id == 'CLM' and current_claim:
                extracted_claims.append(current_claim.copy())
                current_claim = {}

    # In case there's any remaining data
    if current_claim:
        extracted_claims.append(current_claim)

    return extracted_claims








def extract_data_from_segment(segment_line, guide_segment):
    # Convert fields list to flat dictionary for prompt clarity
    field_mappings = {field['code']: field['name'] for field in guide_segment['fields']}

    prompt = f"""
    You are an EDI Data Extractor.

    EDI Segment Line:
    {segment_line}

    Companion Guide Mapping:
    Segment ID: {guide_segment['segment_id']}
    Description: {guide_segment['description']}
    Fields Mapping:
    {json.dumps(field_mappings, indent=2)}

    Extract only the data elements from the segment based on this mapping.
    Return as JSON key-value pairs.
    """
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"Failed to parse extraction for segment: {segment_line[:30]}... Error: {e}")
        return {}










import streamlit as st
import os
import json
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from openai import AzureOpenAI

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = "YOUR_AZURE_OPENAI_KEY"
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT)

# --- UI Layout ---
st.title("📄 EDI Companion Guide & Data Extractor")

# Session State
if 'guide_processed' not in st.session_state:
    st.session_state['guide_processed'] = os.path.exists("cached_guide_segments.json")

# --- Companion Guide Upload ---
companion_guide_file = st.file_uploader("Upload Companion Guide (PDF/DOCX)", type=["pdf", "docx"])

if st.button("🔄 Re-Extract Companion Guide Data"):
    if companion_guide_file:
        with st.spinner("Processing Companion Guide using AI..."):
            guide_data = extract_segments_from_guide(companion_guide_file)

            with open("cached_guide_segments.json", "w") as f:
                json.dump(guide_data, f, indent=2)

            st.session_state['guide_processed'] = True
            st.success("Companion Guide processed and cached successfully!")
    else:
        st.error("Please upload a Companion Guide file first.")

# --- EDI Files Upload ---
uploaded_edi_files = st.file_uploader("Upload EDI Files", type=["txt"], accept_multiple_files=True)

if st.button("🚀 Extract Data from EDI Files"):
    if uploaded_edi_files and st.session_state['guide_processed']:
        with st.spinner("Extracting data from EDI files..."):
            extracted_data = process_edi_files(uploaded_edi_files)
            export_to_excel(extracted_data)
            st.success("✅ Data extracted and saved to Extracted_EDI_Data.xlsx!")
            st.download_button("📥 Download Excel", data=open("Extracted_EDI_Data.xlsx", "rb"), file_name="Extracted_EDI_Data.xlsx")
    else:
        st.error("Upload EDI files and ensure Companion Guide is processed.")


# ------------------ FUNCTION DEFINITIONS -------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_segments_from_guide(file):
    # Extract raw text from file
    if file.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        raw_text = extract_text_from_docx(file)
    else:
        return []

    chunks = split_into_segments(raw_text)

    extracted_data = []
    for chunk in chunks:
        prompt = f"""
        You are an EDI Companion Guide Parser.

        Given Text:
        {chunk}

        Extract Segment ID, Description, and Field Mappings.
        Return JSON like:
        {{
            "segment_id": "SEGMENT_ID",
            "description": "Description",
            "fields": {{
                "ElementCode1": "Field Name 1",
                "ElementCode2": "Field Name 2"
            }}
        }}
        """
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            extracted_data.append(json.loads(response.choices[0].message.content))
        except Exception as e:
            st.warning(f"Failed to parse segment: {chunk[:30]}... Error: {e}")
    return extracted_data

def split_into_segments(raw_text):
    # Example: split by 'Segment:' or segment heading patterns in your guide
    segments = []
    current_segment = ""
    for line in raw_text.split('\n'):
        if line.strip().startswith("Segment") or line.strip().startswith("ISA") or line.strip().startswith("CLM"):
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = line + "\n"
        else:
            current_segment += line + "\n"
    if current_segment:
        segments.append(current_segment.strip())
    return segments

def process_edi_files(edi_files):
    with open("cached_guide_segments.json", "r") as f:
        guide_data = json.load(f)

    extracted_claims = []
    for edi_file in edi_files:
        content = edi_file.read().decode('utf-8', errors='ignore')
        segments = content.split('~')

        current_claim = {}
        for segment in segments:
            seg_id = segment.split('*')[0]
            guide_segment = next((item for item in guide_data if item['segment_id'] == seg_id), None)

            if guide_segment:
                extracted = extract_data_from_segment(segment, guide_segment)
                current_claim.update(extracted)

            if seg_id == 'CLM' and current_claim:
                extracted_claims.append(current_claim.copy())
                current_claim = {}

    return extracted_claims

def extract_data_from_segment(segment_line, guide_segment):
    prompt = f"""
    You are an EDI 837i Data Extractor.

    EDI Segment Line:
    {segment_line}

    Companion Guide Mapping:
    {guide_segment}

    Extract only the data elements from the segment based on the mapping. Return as JSON key-value pairs.
    """
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"Failed to extract data from segment: {segment_line[:30]}... Error: {e}")
        return {}

def export_to_excel(extracted_data):
    df = pd.DataFrame(extracted_data)
    df.to_excel("Extracted_EDI_Data.xlsx", index=False)


















ISA*00*          *00*          *ZZ*SENDERID       *ZZ*RECEIVERID     *240805*1230*^*00501*000000905*0*T*:~
GS*HC*SENDERID*RECEIVERID*20240805*1230*1*X*005010X223A2~
ST*837*0001*005010X223A2~
BHT*0019*00*0123*20240805*1230*CH~
NM1*41*2*SENDER COMPANY*****46*SENDERID~
PER*IC*CONTACT NAME*TE*1234567890~
NM1*40*2*RECEIVER COMPANY*****46*RECEIVERID~
HL*1**20*1~
NM1*85*2*PROVIDER NAME*****XX*1234567893~
N3*123 PROVIDER ADDRESS~
N4*CITY*ST*12345~
CLM*123456789*500***11:B:1*Y*A*Y*I~
DTP*434*RD8*20240801-20240801~
NM1*QC*1*DOE*JOHN****MI*987654321~
N3*456 PATIENT STREET~
N4*PATIENTCITY*ST*98765~
CLM*987654321*300***11:B:1*Y*A*Y*I~
DTP*434*RD8*20240802-20240802~
NM1*QC*1*SMITH*JANE****MI*123123123~
N3*789 PATIENT AVE~
N4*ANOTHERCITY*ST*67890~
SE*20*0001~
GE*1*1~
IEA*1*000000905~









ISA*00*          *00*          *ZZ*PAYERSENDER    *ZZ*PAYEERECEIVER  *240805*1245*^*00501*000000789*0*T*:~
GS*RA*PAYERSENDER*PAYEERECEIVER*20240805*1245*1*X*005010X218~
ST*820*0001*005010X218~
BPR*C*1500*C*ACH*CTX*01*123456789*DA*987654321*20240805~
TRN*1*1234567890*9876543210~
REF*EV*9876543210~
DTM*097*20240805~
N1*PR*PAYER NAME~
N1*PE*PAYEE NAME~
ENT*1*2N*987654321~
RMR*IV*123456789*PO*500~
DTM*003*20240730~
ENT*2*2N*123123123~
RMR*IV*987654321*PO*1000~
DTM*003*20240731~
SE*15*0001~
GE*1*1~
IEA*1*000000789~




@st.cache_data
def build_faiss_index(guide_files):
    chunks = []
    for guide_file in guide_files:
        try:
            content = guide_file.read().decode('utf-8')
        except UnicodeDecodeError:
            guide_file.seek(0)
            content = guide_file.read().decode('latin1')  # fallback to latin1 (windows-1252)
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                chunks.append(line.strip())
    embeddings = OpenAIEmbeddings(openai_api_key=AZURE_OPENAI_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore







import streamlit as st
import pandas as pd
import os
import faiss
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import AzureOpenAI

# Azure OpenAI Config
AZURE_OPENAI_KEY = "YOUR_AZURE_OPENAI_KEY"
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"

# Initialize OpenAI Client
client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT)

st.title("EDI Data Extractor - RAG Powered")

# Sidebar - Upload Companion Guides
st.sidebar.header("Upload Companion Guides")
uploaded_guides = st.sidebar.file_uploader("Upload Guide Files (TXT/PDF)", type=["txt", "pdf"], accept_multiple_files=True)

# Main - Upload EDI Files
uploaded_edi_files = st.file_uploader("Upload EDI Files", type=["txt"], accept_multiple_files=True)

# File Type Selection for EDI Files
file_types = {}
if uploaded_edi_files:
    for edi_file in uploaded_edi_files:
        file_type = st.selectbox(f"Select EDI Type for {edi_file.name}", ["837i", "820"], key=edi_file.name)
        file_types[edi_file.name] = file_type

# Load Companion Guides into FAISS
@st.cache_data
def build_faiss_index(guide_files):
    chunks = []
    for guide_file in guide_files:
        content = guide_file.read().decode('utf-8')
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                chunks.append(line.strip())
    embeddings = OpenAIEmbeddings(openai_api_key=AZURE_OPENAI_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

if uploaded_guides:
    st.sidebar.success("Companion Guides Uploaded")
    vectorstore = build_faiss_index(uploaded_guides)

def preprocess_edi(content):
    edi_cleaned = content.replace('\n', '').replace('\r', '')
    return [seg.strip() for seg in edi_cleaned.split('~') if seg.strip()]

def retrieve_context(segment_line):
    docs = vectorstore.similarity_search(segment_line, k=1)
    return docs[0].page_content if docs else ""

def extract_data(segment_line, context):
    prompt = f"""
    You are an EDI Data Extractor.
    Companion Guide Context:
    {context}

    Given EDI Segment Line:
    {segment_line}

    Extract data elements as key-value JSON.
    """
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Process Button
if st.button("Process EDI Files") and uploaded_edi_files and uploaded_guides:
    all_claims_data = []

    for edi_file in uploaded_edi_files:
        content = edi_file.read().decode('utf-8')
        segments = preprocess_edi(content)

        current_claim = {}
        for segment in segments:
            context = retrieve_context(segment)
            extracted_json = extract_data(segment, context)
            try:
                segment_data = eval(extracted_json)
                if 'Claim_ID' in segment_data:  # Detecting new claim start
                    if current_claim:
                        all_claims_data.append(current_claim)
                    current_claim = {}
                current_claim.update(segment_data)
            except:
                continue
        if current_claim:
            all_claims_data.append(current_claim)

    # Export to Excel
    df = pd.DataFrame(all_claims_data)
    tmp_download_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    df.to_excel(tmp_download_file.name, index=False)

    st.success("Extraction Complete!")
    st.download_button(label="Download Extracted Data", data=open(tmp_download_file.name, 'rb'), file_name='extracted_claims.xlsx')


















def record_text_input():
    global typed_text, last_clicked_control
    if not typed_text.strip():
        return
    if last_clicked_control:
        action = {
            "event": "text_input",
            "text": typed_text,
            "time": time.time(),
            "control": last_clicked_control  # We use last clicked control!
        }
        recorded_actions.append(action)
        print(f"Typed Text: {typed_text} in {last_clicked_control['name']}")
    else:
        print("No control clicked before typing. Text recorded without control info.")
    typed_text = ""









from pywinauto.uia_defines import IUIA

def record_text_input():
    global typed_text
    if not typed_text.strip():
        return
    try:
        focused_elem = IUIA().get_focused_element()
        info = focused_elem.element_info
        action = {
            "event": "text_input",
            "text": typed_text,
            "time": time.time(),
            "control": {
                "name": info.name,
                "control_type": info.control_type,
                "automation_id": info.automation_id,
                "rectangle": str(info.rectangle)
            }
        }
        recorded_actions.append(action)
        print(f"Typed Text: {typed_text} in {info.name}")
    except Exception as e:
        print(f"Failed to get focused control: {e}")
    typed_text = ""
















def record_text_input():
    global typed_text
    if not typed_text.strip():
        return
    try:
        focused_ctrl = Desktop(backend="uia").get_focus().element_info
        action = {
            "event": "text_input",
            "text": typed_text,
            "time": time.time(),
            "control": {
                "name": focused_ctrl.name,
                "control_type": focused_ctrl.control_type,
                "automation_id": focused_ctrl.automation_id,
                "rectangle": str(focused_ctrl.rectangle)
            }
        }
        recorded_actions.append(action)
        print(f"Typed Text: {typed_text} in {focused_ctrl.name}")
    except Exception as e:
        print(f"Failed to get focused control: {e}")
    typed_text = ""









import mouse
import keyboard
import json
import time
import threading
from pywinauto import Desktop
import win32gui

recorded_actions = []
typed_text = ""
recording = True  # Flag to stop recording on ESC

print("Recording started... Perform actions and press ESC to stop.")

# Get control from current mouse position
def control_from_current_position():
    try:
        x, y = mouse.get_position()
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle),
            "coords": (x, y)
        }
    except Exception as e:
        print(f"Failed to get control at ({x},{y}): {e}")
        return None

# Mouse Event Listener
def mouse_listener():
    while recording:
        # Mouse is event-driven, no need for polling
        pass  # Hook is handled outside (non-blocking)
    print("Mouse listener stopped.")

# Mouse Hook Callback
def on_mouse_event(event):
    if hasattr(event, 'event_type') and hasattr(event, 'button'):
        if event.event_type == 'down' and event.button in ['left', 'right']:
            time.sleep(0.2)
            ctrl = control_from_current_position()
            if ctrl:
                recorded_actions.append({
                    "event": "mouse_click",
                    "button": event.button,
                    "time": time.time(),
                    "control": ctrl
                })
                print(f"Clicked on {ctrl['name']} at {ctrl['coords']}")

# Keyboard Hook Callback
def on_key_event(event):
    global typed_text, recording
    if event.name == 'esc' and event.event_type == 'down':
        # Stop Recording
        print("ESC pressed. Stopping recording.")
        recording = False
        if typed_text:
            record_text_input()
        # Unhook listeners
        mouse.unhook(on_mouse_event)
        keyboard.unhook_all()
        # Save actions
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        print("Actions saved to recorded_user_actions.json")
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    if not typed_text.strip():
        return
    foreground_hwnd = win32gui.GetForegroundWindow()
    active_ctrl = Desktop(backend="uia").window(handle=foreground_hwnd).element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": active_ctrl.name,
            "control_type": active_ctrl.control_type,
            "automation_id": active_ctrl.automation_id,
            "rectangle": str(active_ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {active_ctrl.name}")
    typed_text = ""

# Start Mouse Hook (non-blocking)
mouse.hook(on_mouse_event)

# Start Keyboard Hook (non-blocking)
keyboard.hook(on_key_event)

# Keep main thread alive while recording
while recording:
    time.sleep(0.5)

print("Recorder stopped.")





















import mouse
import keyboard
import json
import time
from pywinauto import Desktop
import win32gui

recorded_actions = []

print("Recording actions... Open your app and perform actions. Press ESC to stop.")

# Get control from current mouse position
def control_from_current_position():
    try:
        x, y = mouse.get_position()
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle),
            "coords": (x, y)
        }
    except Exception as e:
        print(f"Failed to get control at ({x},{y}): {e}")
        return None

# Mouse Event Hook
def on_mouse_event(event):
    if hasattr(event, 'event_type') and hasattr(event, 'button'):
        if event.event_type == 'down' and event.button in ['left', 'right']:
            time.sleep(0.2)  # Small UI stabilization delay
            ctrl = control_from_current_position()
            if ctrl:
                recorded_actions.append({
                    "event": "mouse_click",
                    "button": event.button,
                    "time": time.time(),
                    "control": ctrl
                })
                print(f"Clicked on {ctrl['name']} at {ctrl['coords']}")

# Keyboard Hook
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    foreground_hwnd = win32gui.GetForegroundWindow()
    active_ctrl = Desktop(backend="uia").window(handle=foreground_hwnd).element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": active_ctrl.name,
            "control_type": active_ctrl.control_type,
            "automation_id": active_ctrl.automation_id,
            "rectangle": str(active_ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {active_ctrl.name}")
    typed_text = ""

mouse.hook(on_mouse_event)
keyboard.hook(on_key)

keyboard.wait('esc')















import mouse
import keyboard
import json
import time
from pywinauto import Desktop
import win32gui

recorded_actions = []

print("Recording actions... Open your app and perform actions. Press ESC to stop.")

# Resolve control dynamically at click position
def control_from_point(x, y):
    try:
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle)
        }
    except:
        return None

# Mouse Event Hook
def on_mouse_event(event):
    if hasattr(event, 'event_type') and hasattr(event, 'button'):
        if event.event_type == 'down' and event.button in ['left', 'right']:
            time.sleep(0.2)  # Small delay to let UI settle
            ctrl = control_from_point(event.x, event.y)
            if ctrl:
                recorded_actions.append({
                    "event": "mouse_click",
                    "button": event.button,
                    "time": time.time(),
                    "control": ctrl
                })
                print(f"Clicked on {ctrl['name']} at ({event.x},{event.y})")

# Keyboard Hook
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    foreground_hwnd = win32gui.GetForegroundWindow()
    active_ctrl = Desktop(backend="uia").window(handle=foreground_hwnd).element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": active_ctrl.name,
            "control_type": active_ctrl.control_type,
            "automation_id": active_ctrl.automation_id,
            "rectangle": str(active_ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {active_ctrl.name}")
    typed_text = ""

mouse.hook(on_mouse_event)
keyboard.hook(on_key)

keyboard.wait('esc')













import mouse
import keyboard
import json
import time
from pywinauto import Desktop
from pywinauto.controls.uiawrapper import UIAWrapper

recorded_actions = []
control_map = []

print("Loading UI Tree...")

def dump_control_tree(element: UIAWrapper, depth=0):
    node = {
        "name": element.element_info.name,
        "control_type": element.element_info.control_type,
        "automation_id": element.element_info.automation_id,
        "rectangle": element.element_info.rectangle,
        "depth": depth
    }
    control_map.append(node)
    try:
        for child in element.children():
            dump_control_tree(child, depth + 1)
    except:
        pass

# Load Active App Tree
active_app = Desktop(backend="uia").get_active()
dump_control_tree(active_app)

print(f"Loaded {len(control_map)} controls.")

# Find Control by Mouse Position
def find_control_by_position(x, y):
    for ctrl in control_map:
        rect = ctrl["rectangle"]
        if rect.left <= x <= rect.right and rect.top <= y <= rect.bottom:
            return ctrl
    return None

# Mouse Event Hook
def on_mouse_event(event):
    if hasattr(event, 'event_type') and hasattr(event, 'button'):
        if event.event_type == 'down' and event.button in ['left', 'right']:
            time.sleep(0.2)  # Let system stabilize UI state
            ctrl = find_control_by_position(event.x, event.y)
            if ctrl:
                recorded_actions.append({
                    "event": "mouse_click",
                    "button": event.button,
                    "time": time.time(),
                    "control": {
                        "name": ctrl['name'],
                        "control_type": ctrl['control_type'],
                        "automation_id": ctrl['automation_id'],
                        "rectangle": str(ctrl['rectangle'])
                    }
                })
                print(f"Clicked on {ctrl['name']} at ({event.x},{event.y})")

# Keyboard Hook
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    active_ctrl = Desktop(backend="uia").get_active().element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": active_ctrl.name,
            "control_type": active_ctrl.control_type,
            "automation_id": active_ctrl.automation_id,
            "rectangle": str(active_ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {active_ctrl.name}")
    typed_text = ""

mouse.hook(on_mouse_event)
keyboard.hook(on_key)

keyboard.wait('esc')












import mouse
import keyboard
import json
import time
from pywinauto import Desktop

recorded_actions = []

print("Recording actions... Perform actions and press ESC to stop.")

def control_from_point(x, y):
    try:
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle)
        }
    except:
        return None

# Mouse Event Hook
def on_mouse_event(event):
    if hasattr(event, 'event_type') and hasattr(event, 'button'):
        if event.event_type == 'down' and event.button in ['left', 'right']:
            if hasattr(event, 'x') and hasattr(event, 'y'):
                ctrl = control_from_point(event.x, event.y)
                if ctrl:
                    recorded_actions.append({
                        "event": "mouse_click",
                        "button": event.button,
                        "time": time.time(),
                        "control": ctrl
                    })
                    print(f"Clicked on {ctrl['name']} at ({event.x},{event.y})")

# Keyboard Hook
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    ctrl = Desktop(backend="uia").get_active().element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": ctrl.name,
            "control_type": ctrl.control_type,
            "automation_id": ctrl.automation_id,
            "rectangle": str(ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {ctrl.name}")
    typed_text = ""

mouse.hook(on_mouse_event)
keyboard.hook(on_key)

keyboard.wait('esc')









..
import mouse
import keyboard
import json
import time
from pywinauto import Desktop

recorded_actions = []

print("Recording actions... Perform actions on app and press ESC to stop.")

def control_from_point(x, y):
    try:
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle)
        }
    except:
        return None

# Mouse Hook for click events
def on_mouse_event(event):
    if event.event_type == 'down' and event.button in ['left', 'right']:
        ctrl = control_from_point(event.x, event.y)
        if ctrl:
            recorded_actions.append({
                "event": "mouse_click",
                "button": event.button,
                "time": time.time(),
                "control": ctrl
            })
            print(f"Clicked on {ctrl['name']} at ({event.x},{event.y})")

# Keyboard Listener to capture text inputs
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    ctrl = Desktop(backend="uia").get_active().element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": ctrl.name,
            "control_type": ctrl.control_type,
            "automation_id": ctrl.automation_id,
            "rectangle": str(ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {ctrl.name}")
    typed_text = ""

mouse.hook(on_mouse_event)
keyboard.hook(on_key)

keyboard.wait('esc')










import mouse
import keyboard
import json
import time
from pywinauto import Desktop

recorded_actions = []

print("Recording actions... Click on app and Press ESC to stop.")

def control_from_point(x, y):
    try:
        elem = Desktop(backend="uia").from_point(x, y)
        info = elem.element_info
        return {
            "name": info.name,
            "control_type": info.control_type,
            "automation_id": info.automation_id,
            "rectangle": str(info.rectangle)
        }
    except:
        return None

# Mouse Click Listener using from_point
def on_click(event):
    ctrl = control_from_point(event.x, event.y)
    if ctrl:
        recorded_actions.append({
            "event": "mouse_click",
            "button": event.button,
            "time": time.time(),
            "control": ctrl
        })
        print(f"Clicked on {ctrl['name']} at ({event.x},{event.y})")

# Keyboard Listener to capture text inputs
typed_text = ""
def on_key(event):
    global typed_text
    if event.name == 'esc':
        if typed_text:
            record_text_input()
        print("Recording stopped.")
        mouse.unhook_all()
        keyboard.unhook_all()
        with open("recorded_user_actions.json", "w") as f:
            json.dump(recorded_actions, f, indent=4)
        exit()
    elif event.event_type == 'down':
        if event.name == 'enter':
            record_text_input()
        else:
            typed_text += event.name if len(event.name) == 1 else ''

def record_text_input():
    global typed_text
    ctrl = Desktop(backend="uia").get_active().element_info
    action = {
        "event": "text_input",
        "text": typed_text,
        "time": time.time(),
        "control": {
            "name": ctrl.name,
            "control_type": ctrl.control_type,
            "automation_id": ctrl.automation_id,
            "rectangle": str(ctrl.rectangle)
        }
    }
    recorded_actions.append(action)
    print(f"Typed Text: {typed_text} in {ctrl.name}")
    typed_text = ""

mouse.on_click(on_click)
keyboard.hook(on_key)

keyboard.wait('esc')









import streamlit as st
import subprocess
import pyautogui
import keyboard
import time
import json
from pywinauto import Desktop
import requests
import os

# ========== CONFIG ==========
AZURE_OPENAI_ENDPOINT = 'https://<YOUR-ENDPOINT>.openai.azure.com/'
AZURE_OPENAI_KEY = '<YOUR-API-KEY>'
DEPLOYMENT_NAME = '<YOUR-DEPLOYMENT-NAME>'  # GPT-4 deployment name in Azure
# ============================

# List of apps you want to automate
apps = {
    "Notepad": r"C:\Windows\System32\notepad.exe",
    "Calculator": r"C:\Windows\System32\calc.exe"
}

st.title("AI-Powered RPA Recorder (Streamlit + Azure OpenAI)")

# Streamlit UI Elements
selected_app = st.selectbox("Select Application", list(apps.keys()))

if st.button("Start Recording"):
    subprocess.Popen(apps[selected_app])
    st.success(f"Recording started for {selected_app}. Perform your actions. Press 'Esc' key to stop.")
    recorded_actions = []

    st.info("Recording in Progress... (Press 'Esc' to Stop Recording)")

    while True:
        # Stop recording if ESC is pressed
        if keyboard.is_pressed('esc'):
            st.success("Recording Stopped.")
            break
        
        # Capture Mouse Clicks
        if pyautogui.mouseDown():
            x, y = pyautogui.position()
            element_info = {}
            try:
                elem = Desktop(backend="uia").get_active()
                element_info = {
                    'name': elem.element_info.name,
                    'control_type': elem.element_info.control_type,
                    'automation_id': elem.element_info.automation_id,
                    'rectangle': str(elem.element_info.rectangle),
                }
            except Exception:
                pass

            action = {
                'type': 'click',
                'position': (x, y),
                'element': element_info,
                'timestamp': time.time()
            }
            recorded_actions.append(action)
            time.sleep(0.5)

        # Capture Keypresses
        keys = keyboard.get_hotkey_name()
        if keys and keys != 'esc':
            element_info = {}
            try:
                elem = Desktop(backend="uia").get_active()
                element_info = {
                    'name': elem.element_info.name,
                    'control_type': elem.element_info.control_type,
                    'automation_id': elem.element_info.automation_id,
                    'rectangle': str(elem.element_info.rectangle),
                }
            except Exception:
                pass

            action = {
                'type': 'keypress',
                'keys': keys,
                'element': element_info,
                'timestamp': time.time()
            }
            recorded_actions.append(action)
            time.sleep(0.5)

    # Save recorded actions
    with open("recorded_actions.json", "w") as f:
        json.dump(recorded_actions, f, indent=4)

    st.success("Actions Recorded Successfully!")

# Generate Script using Azure OpenAI API
if st.button("Generate Script with Azure OpenAI"):
    if not os.path.exists("recorded_actions.json"):
        st.error("No recorded actions found. Please record first.")
    else:
        with open("recorded_actions.json", "r") as f:
            recorded_data = json.load(f)

        prompt = f"""
You are an expert RPA Automation Script Generator using Python's pywinauto library.
Based on the following recorded user actions, generate a Python script to automate them:
{json.dumps(recorded_data, indent=2)}

- Launch the application using pywinauto Application(backend="uia").
- For clicks, try to identify controls by title, control_type, automation_id.
- If identification fails, fallback to click_input(coords=(x, y)).
- For keypress actions, use type_keys.
- Add appropriate waits for control readiness.
- Generate clean executable Python code.
"""

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_KEY
        }

        payload = {
            "messages": [
                {"role": "system", "content": "You are a Python automation script writer."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.2
        }

        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            ai_response = response.json()
            generated_code = ai_response['choices'][0]['message']['content']
            st.code(generated_code, language="python")

            with open("generated_script.py", "w") as f:
                f.write(generated_code)

            st.download_button("Download Script", generated_code, file_name="generated_script.py")
        else:
            st.error(f"Failed to get response from Azure OpenAI: {response.text}")

# Replay Script
if st.button("Replay Generated Script"):
    if not os.path.exists("generated_script.py"):
        st.error("No generated script found. Please generate it first.")
    else:
        subprocess.run(["python", "generated_script.py"])
        st.success("Replay Completed.")


















import streamlit as st
import subprocess
import pyautogui
import keyboard
import time
import json
from pywinauto import Desktop
import requests
import os

# ========== CONFIG ==========
AZURE_OPENAI_ENDPOINT = 'https://<YOUR-ENDPOINT>.openai.azure.com/'
AZURE_OPENAI_KEY = '<YOUR-API-KEY>'
DEPLOYMENT_NAME = '<YOUR-DEPLOYMENT-NAME>'  # GPT-4 deployment name in Azure
# ============================

# List of apps you want to automate
apps = {
    "Notepad": r"C:\Windows\System32\notepad.exe",
    "Calculator": r"C:\Windows\System32\calc.exe"
}

st.title("AI-Powered RPA Recorder (Streamlit + Azure OpenAI)")

# Streamlit UI Elements
selected_app = st.selectbox("Select Application", list(apps.keys()))

if st.button("Start Recording"):
    subprocess.Popen(apps[selected_app])
    st.success(f"Recording started for {selected_app}. Perform your actions. Press 'Esc' key to stop.")
    recorded_actions = []

    st.info("Recording in Progress... (Press 'Esc' to Stop Recording)")

    while True:
        if keyboard.is_pressed('esc'):
            st.success("Recording Stopped.")
            break
        
        x, y = pyautogui.position()
        if pyautogui.mouseDown():
            element_info = {}
            try:
                elem = Desktop(backend="uia").get_active()
                element_info = {
                    'name': elem.element_info.name,
                    'control_type': elem.element_info.control_type,
                    'automation_id': elem.element_info.automation_id,
                    'rectangle': str(elem.element_info.rectangle),
                }
            except Exception:
                pass

            action = {
                'type': 'click',
                'position': (x, y),
                'element': element_info,
                'timestamp': time.time()
            }
            recorded_actions.append(action)
            time.sleep(0.5)

        if keyboard.is_pressed():
            keys = keyboard.get_hotkey_name()
            element_info = {}
            try:
                elem = Desktop(backend="uia").get_active()
                element_info = {
                    'name': elem.element_info.name,
                    'control_type': elem.element_info.control_type,
                    'automation_id': elem.element_info.automation_id,
                    'rectangle': str(elem.element_info.rectangle),
                }
            except Exception:
                pass

            action = {
                'type': 'keypress',
                'keys': keys,
                'element': element_info,
                'timestamp': time.time()
            }
            recorded_actions.append(action)
            time.sleep(0.5)

    # Save recorded actions
    with open("recorded_actions.json", "w") as f:
        json.dump(recorded_actions, f, indent=4)

    st.success("Actions Recorded Successfully!")

# Generate Script using Azure OpenAI API
if st.button("Generate Script with Azure OpenAI"):
    if not os.path.exists("recorded_actions.json"):
        st.error("No recorded actions found. Please record first.")
    else:
        with open("recorded_actions.json", "r") as f:
            recorded_data = json.load(f)

        prompt = f"""
You are an RPA Script Generator using Python's pywinauto library.
Based on the following recorded user actions, generate a Python script that automates them:
{json.dumps(recorded_data, indent=2)}

Use pywinauto Application(backend="uia") to launch apps.
For each click, locate the control using title, control_type, automation_id.
If precise element details are unavailable, fallback to click_input(coords=(x, y)).
Handle keypresses using type_keys.
Ensure waits and proper flow is maintained.
Generate clean executable Python code.
"""

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_KEY
        }

        payload = {
            "messages": [
                {"role": "system", "content": "You are a Python automation script writer."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.2
        }

        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            ai_response = response.json()
            generated_code = ai_response['choices'][0]['message']['content']
            st.code(generated_code, language="python")

            with open("generated_script.py", "w") as f:
                f.write(generated_code)

            st.download_button("Download Script", generated_code, file_name="generated_script.py")
        else:
            st.error(f"Failed to get response from Azure OpenAI: {response.text}")

# Replay Script
if st.button("Replay Generated Script"):
    if not os.path.exists("generated_script.py"):
        st.error("No generated script found. Please generate it first.")
    else:
        subprocess.run(["python", "generated_script.py"])
        st.success("Replay Completed.")














| **S.No** | **Achievement Category**                      | **Details**                                                                                                                                                         |
| -------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Multi-PDF Ingestion & Processing              | Successfully designed a workflow to ingest multiple PDFs and extract relevant content chunk-wise using a combination of text and image (OCR) layers.                |
| 2        | Hybrid Chunking (Text + OCR)                  | Implemented fallback OCR extraction using `pdf2image` and `pytesseract` to handle image-based PDFs in absence of text layer.                                        |
| 3        | Vector Store Migration to FAISS               | Transitioned from Qdrant DB (which couldn't be installed on VDI) to FAISS for local vector storage with persistent index management across multiple sessions.       |
| 4        | Metadata-based File Isolation in FAISS        | Maintained file-level segregation in FAISS using metadata (filename tags) to avoid mixing chunks from different PDFs during similarity searches.                    |
| 5        | Retrieval-Augmented Generation (RAG) Pipeline | Successfully built an end-to-end RAG flow: Extract relevant content from contract → Summarize → Generate test cases → Refine them iteratively using LLM.            |
| 6        | Azure OpenAI Integration                      | Integrated with Azure OpenAI endpoint for GPT-4o completions, switching from Groq API to ensure enterprise compliance.                                              |
| 7        | Streamlit UI for Non-Technical Users          | Developed a Streamlit-based tool that allows users to upload PDFs, trigger chunk extraction, and generate refined test scenarios in a user-friendly interface.      |
| 8        | FAISS Persistent Index Save/Load              | Enabled saving/loading of FAISS indexes to disk (local JSON and binary files), ensuring persistence across sessions.                                                |
| 9        | Chunk Querying with Filename Filtering        | Enhanced search mechanism to retrieve only relevant chunks based on filename and query similarity, simulating Qdrant's filter feature.                              |
| 10       | Test Case Refinement & Expansion              | Developed logic to post-process LLM outputs to ensure sufficient depth of clinical journey test cases, dynamically adding service lines based on scenarios.         |
| 11       | DataFrame-driven Post Processing              | Leveraged Pandas for refining, merging, and validating the generated test cases, ensuring alignment with contract-provided codes and SDOH billing data.             |
| 12       | Flexible Deployment without Admin Access      | All code changes were made to work within VDI constraints (no environment variable edits, no Docker, no Qdrant), using local binary paths and in-memory operations. |





import faiss
import numpy as np
import pickle
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document

faiss_index = None
docstore = None
index_to_docstore_id = None
current_id = 0  # Global unique ID across all docs

def store_chunks_in_faiss(chunks, filename, embedding_model):
    global faiss_index, docstore, index_to_docstore_id, current_id

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"filename": filename, "text": chunk.page_content} for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    dim = len(vectors[0])
    if faiss_index is None:
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        faiss_index = index
        docstore = {}
        index_to_docstore_id = {}

    ids = []
    for i, (vec, meta, text) in enumerate(zip(vectors, metadatas, texts)):
        doc_id = str(current_id)
        faiss_index.add_with_ids(np.array([vec]).astype('float32'), np.array([current_id]))
        docstore[doc_id] = Document(page_content=text, metadata=meta)
        index_to_docstore_id[current_id] = doc_id
        ids.append(current_id)
        current_id += 1

    print(f"Added {len(ids)} chunks from {filename} to FAISS.")


def query_relevant_chunks(query, filename, embedding_model, threshold=0.1, top_k=30):
    query_vector = embedding_model.embed_query(query)

    scores, indices = faiss_index.search(np.array([query_vector]).astype('float32'), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score > threshold:
            continue  # Filter low scores
        doc_id = index_to_docstore_id.get(idx)
        if doc_id:
            doc = docstore[doc_id]
            if doc.metadata.get("filename") == filename:
                results.append(doc.page_content)
    return results

def save_faiss_index(path="faiss_data/multi_pdf"):
    faiss.write_index(faiss_index, f"{path}.index")
    data = {
        "docstore": docstore,
        "index_to_docstore_id": index_to_docstore_id,
        "current_id": current_id
    }
    with open(f"{path}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"FAISS index saved at {path}.index and {path}.pkl")

def load_faiss_index(path="faiss_data/multi_pdf"):
    global faiss_index, docstore, index_to_docstore_id, current_id
    faiss_index = faiss.read_index(f"{path}.index")
    with open(f"{path}.pkl", "rb") as f:
        data = pickle.load(f)
    docstore = data["docstore"]
    index_to_docstore_id = data["index_to_docstore_id"]
    current_id = data["current_id"]
    print(f"FAISS index loaded with {len(index_to_docstore_id)} entries.")
    










def query_relevant_chunks(query, filename, threshold=0.1, top_k=30):
    if faiss_index is None:
        print("FAISS index is not built.")
        return []

    # Embed and Normalize Query Vector
    query_vector = embedding_model.embed_query(query)
    query_vector = query_vector / np.linalg.norm(query_vector)

    scores, indices = faiss_index.index.search(np.array([query_vector]).astype("float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # No result
        doc_id = faiss_index.index_to_docstore_id.get(idx)
        if doc_id is None:
            continue
        doc = faiss_index.docstore.search(doc_id)
        if doc and doc.metadata.get("filename") == filename:
            print(f"[Match] Filename: {doc.metadata.get('filename')} | Score: {score}")
            if score >= threshold:
                results.append(doc.page_content)

    if not results:
        print("No relevant chunks found with current threshold/filename filter.")

    return results







import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

faiss_index = None  # Global FAISS Index

def store_chunks_in_faiss(chunks, filename):
    global faiss_index

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"filename": filename, "text": chunk.page_content} for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)
    # Normalize vectors for Cosine Similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)  # Inner Product for Cosine Similarity
    index.add(np.array(vectors).astype("float32"))

    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    faiss_index = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    print(f"FAISS index built with {len(vectors)} vectors for file: {filename}")







import os
import json
import faiss
import numpy as np
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Global FAISS index
faiss_index = None

# Embedding model (Assumed to be initialized elsewhere)
embedding_model = None  # You should initialize your HuggingFaceBgeEmbeddings here

# ---------- Store Chunks in FAISS and Persist ----------
def store_chunks_in_faiss(chunks, filename):
    global faiss_index

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"filename": filename, "text": chunk.page_content} for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)

    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
    dim = len(vectors[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    os.makedirs("faiss_indices", exist_ok=True)
    faiss.write_index(index, f"faiss_indices/{filename}.index")

    # Persist docstore metadata
    with open(f"faiss_indices/{filename}_docstore.json", "w", encoding="utf-8") as f:
        json.dump([
            {"id": i, "text": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(documents)
        ], f, indent=2)

    # In-memory index for immediate use
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    faiss_index = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

# ---------- Load FAISS Index & Docstore ----------
def load_faiss_index(filename):
    global faiss_index

    index = faiss.read_index(f"faiss_indices/{filename}.index")

    with open(f"faiss_indices/{filename}_docstore.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docstore = InMemoryDocstore({str(d["id"]): Document(page_content=d["text"], metadata=d["metadata"]) for d in data})
    index_to_docstore_id = {d["id"]: str(d["id"]) for d in data}

    faiss_index = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

# ---------- Query Relevant Chunks ----------
def query_relevant_chunks(query, filename, threshold=0.75, top_k=30):
    global faiss_index

    query_vector = embedding_model.embed_query(query)

    scores, indices = faiss_index.index.search(
        np.array([query_vector]).astype("float32"), top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        doc_id = faiss_index.index_to_docstore_id.get(idx)
        if doc_id is None:
            continue

        doc = faiss_index.docstore.search(doc_id)
        if doc and doc.metadata.get("filename") == filename and score >= threshold:
            results.append(doc.page_content)

    return results

# ---------- Usage Example ----------
# 1. Store Chunks:
# store_chunks_in_faiss(chunks, filename="contract1")

# 2. Load FAISS index when app restarts:
# load_faiss_index(filename="contract1")

# 3. Query:
# results = query_relevant_chunks(query="HIV SNP services", filename="contract1")
# print(results)





















from pywinauto import Application, Desktop
import time

# Start Calculator
Application(backend="uia").start("calc.exe")
time.sleep(2)  # Allow time for full load

# Connect to Calculator window
calc_win = Desktop(backend="uia").window(title_re=".*Calculator.*")
calc_win.wait("visible", timeout=10)
calc_win.set_focus()

# Perform 2 + 3 =
calc_win.child_window(title="Two", control_type="Button").click_input()
calc_win.child_window(title="Plus", control_type="Button").click_input()
calc_win.child_window(title="Three", control_type="Button").click_input()
calc_win.child_window(title="Equals", control_type="Button").click_input()

# Extract and clean result
result_text = calc_win.child_window(auto_id="CalculatorResults", control_type="Text").window_text()
result = result_text.replace("Display is", "").strip()

print("Result is:", result)



from pywinauto import Desktop
windows = Desktop(backend="uia").windows()
for w in windows:
    print(w.window_text())
    


from pywinauto.application import Application
from pywinauto.findwindows import find_windows

app = Application(backend="uia").start("calc.exe")
time.sleep(2)

windows = find_windows(process=app.process, backend="uia")
print("Handles found:", windows)

from pywinauto import Desktop
import time

# Launch Calculator manually first or via script
app = Application(backend="uia").start("calc.exe")
time.sleep(2)

# Use Desktop to list top-level windows
win = Desktop(backend="uia").window(title_re=".*Calculator.*")
win.wait("visible", timeout=10)
win.set_focus()

win.child_window(title="Two", control_type="Button").click_input()
win.child_window(title="Plus", control_type="Button").click_input()
win.child_window(title="Three", control_type="Button").click_input()
win.child_window(title="Equals", control_type="Button").click_input()

result = win.child_window(auto_id="CalculatorResults", control_type="Text").window_text()
print("Result is:", result.replace("Display is", "").strip())











from pywinauto.application import Application
from pywinauto.findwindows import ElementNotFoundError
import time

# Start Calculator
app = Application(backend="uia").start("calc.exe")
time.sleep(1)  # Initial wait

# Print all windows in the app (to debug titles)
windows = app.windows()
print(f"Found {len(windows)} windows:")
for w in windows:
    print(w.window_text())

# Try to attach to the first visible window (fallback logic)
main_win = None
for _ in range(10):  # Try for 5 seconds max
    try:
        windows = app.windows()
        for w in windows:
            if w.is_visible() and "calc" in w.window_text().lower():
                main_win = w
                break
        if main_win:
            break
        time.sleep(0.5)
    except Exception as e:
        print("Retrying due to:", e)
        time.sleep(0.5)

if not main_win:
    raise TimeoutError("Calculator window not found or visible.")

# Focus and operate
main_win.set_focus()
main_win.child_window(title="Two", control_type="Button").click_input()
main_win.child_window(title="Plus", control_type="Button").click_input()
main_win.child_window(title="Three", control_type="Button").click_input()
main_win.child_window(title="Equals", control_type="Button").click_input()

# Get and clean the result
result = main_win.child_window(auto_id="CalculatorResults", control_type="Text").window_text()
print("Result is:", result.replace("Display is", "").strip())












from pywinauto.application import Application
from pywinauto.findwindows import ElementNotFoundError
import time

# Start the Calculator app
app = Application(backend="uia").start("calc.exe")
time.sleep(1)

# Retry to find the window for a few seconds
for _ in range(10):
    try:
        calc = app.window(title_re=".*Calculator.*")
        calc.wait("visible", timeout=2)
        break
    except ElementNotFoundError:
        time.sleep(0.5)
else:
    raise Exception("Calculator window not found after waiting.")

# Bring to front
calc.set_focus()

# Do operation: 2 + 3 =
calc.child_window(title="Two", control_type="Button").click_input()
calc.child_window(title="Plus", control_type="Button").click_input()
calc.child_window(title="Three", control_type="Button").click_input()
calc.child_window(title="Equals", control_type="Button").click_input()

# Get result
result = calc.child_window(auto_id="CalculatorResults", control_type="Text").window_text()
print("Result is:", result)














from pywinauto.application import Application
import time

# Start the Calculator app
app = Application(backend="uia").start('calc.exe')
time.sleep(1)  # wait for it to open

# Connect to the calculator window
calc = app.window(title_re='Calculator')

# Make sure it's in focus
calc.set_focus()

# Perform 2 + 3 =
calc.child_window(title="Two", control_type="Button").click_input()
calc.child_window(title="Plus", control_type="Button").click_input()
calc.child_window(title="Three", control_type="Button").click_input()
calc.child_window(title="Equals", control_type="Button").click_input()

# Wait for result to update
time.sleep(0.5)

# Get the result
result = calc.child_window(auto_id="CalculatorResults", control_type="Text").window_text()

print("Result is:", result)

# Optional: Close Calculator
# app.kill()



from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
import numpy as np
import faiss

def store_chunks_in_faiss(chunks, filename):
    global faiss_index  # Optional: if you want to reuse or update it later

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"filename": filename, "text": chunk.page_content} for chunk in chunks]

    # 1. Embed the documents manually (just like you do in Qdrant)
    vectors = embedding_model.embed_documents(texts)

    # 2. Create Document objects (LangChain uses these with metadata)
    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

    # 3. Create FAISS index with same vector size
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)

    # 4. Add vectors to FAISS
    index.add(np.array(vectors).astype("float32"))

    # 5. Create LangChain-compatible FAISS object
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    faiss_index = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def query_relevant_chunks(query, filename, threshold=0.75, top_k=30):
    # Embed the query
    query_vector = embedding_model.embed_query(query)

    # Search FAISS index
    scores, indices = faiss_index.index.search(
        np.array([query_vector]).astype("float32"), top_k
    )

    # Filter results by score threshold and filename match
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # No result

        doc_id = faiss_index.index_to_docstore_id.get(idx)
        if doc_id is None:
            continue

        doc = faiss_index.docstore.search(doc_id)
        if doc and doc.metadata.get("filename") == filename and score >= threshold:
            results.append(doc.page_content)

    return results











max_rows_per_file = 10000

# Convert columns to a single header string
header = "".join(col for col in all_columns)

# Total number of rows
total_rows = len(all_data_rows)

# Calculate how many output files are needed
total_files = (total_rows + max_rows_per_file - 1) // max_rows_per_file

for file_num in range(total_files):
    start_index = file_num * max_rows_per_file
    end_index = min(start_index + max_rows_per_file, total_rows)
    chunk_rows = all_data_rows[start_index:end_index]
    
    filename = f"output_claims25-1_latest_{file_num + 1}.txt"
    
    with open(filename, "w") as file:
        file.write(header + "\n")  # write header in every file
        
        for row in chunk_rows:
            s = "|".join(row_val for row_val in row)
            file.write(s + "\n")

















from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

def fallback_ocr_loader(pdf_path):
    images = convert_from_path(pdf_path)
    docs = []
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img)
            page_text = f"\n[OCR Page {i+1}]\n{text.strip()}"
            docs.append(Document(page_content=page_text))
        except Exception as e:
            print(f"OCR failed on page {i+1}: {e}")
    return docs

def merge_text_and_ocr(text_docs, ocr_docs):
    merged_docs = []
    total_pages = max(len(text_docs), len(ocr_docs))
    
    for i in range(total_pages):
        text_content = text_docs[i].page_content.strip() if i < len(text_docs) else ""
        ocr_content = ocr_docs[i].page_content.strip() if i < len(ocr_docs) else ""
        
        combined = f"{text_content}\n\n{ocr_content}".strip()
        merged_docs.append(Document(page_content=combined))
    return merged_docs

def load_and_chunk_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()
    except:
        text_docs = []

    if not text_docs or not text_docs[0].page_content.strip():
        text_docs = []

    ocr_docs = fallback_ocr_loader(pdf_path)

    # Merge both sources (text + OCR)
    merged_docs = merge_text_and_ocr(text_docs, ocr_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(merged_docs)

    return chunks











import streamlit as st
import os
import tempfile
import json
import pandas as pd
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract

# Configuration
QDRANT_HOST = "localhost"
COLLECTION_NAME = "hiv_snp_chunks"
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en", model_kwargs={"device": "cpu"})
qdrant = QdrantClient(QDRANT_HOST)

# OCR fallback
def fallback_ocr_loader(pdf_path):
    images = convert_from_path(pdf_path)
    all_text = ""
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img)
            all_text += f"\nPage {i+1}\n{text}"
        except Exception as e:
            print(f"OCR failed on page {i+1}: {e}")
    return all_text

# Load and chunk PDF
def load_and_chunk_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except:
        docs = [fallback_ocr_loader(pdf_path)]
    if not docs or not docs[0].page_content.strip():
        docs = [fallback_ocr_loader(pdf_path)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(docs)

# Store chunks in Qdrant with file name tag
def store_chunks_in_qdrant(chunks, filename):
    vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={"text": chunk.page_content, "filename": filename}
        )
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
    )
    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Query Qdrant for relevant chunks
def query_relevant_chunks(query, filename, threshold=0.75, top_k=30):
    query_vector = embedding_model.embed_query(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
        score_threshold=threshold
    )
    return [res.payload["text"] for res in results]

# Choose LLM client
def get_llm_client(source, groq_key=None, azure_key=None, azure_url=None, azure_deployment=None):
    if source == "Groq":
        return ChatGroq(api_key=groq_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        return AzureChatOpenAI(
            azure_endpoint=azure_url,
            openai_api_key=azure_key,
            deployment_name=azure_deployment,
            model_name="gpt-4o",
            api_version="2025-01-01-preview"
        )

# Streamlit App
st.title("📄 HIV SNP Test Case Generator")

with st.sidebar:
    st.header("⚙️ LLM Settings")
    llm_source = st.selectbox("LLM Provider", ["Groq", "Azure OpenAI"])
    groq_key = st.text_input("Groq API Key", type="password")
    azure_key = st.text_input("Azure API Key", type="password")
    azure_url = st.text_input("Azure Endpoint")
    azure_deployment = st.text_input("Azure Deployment (e.g. gpt-4o)")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded in uploaded_files:
        with st.spinner(f"Processing {uploaded.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                chunks = load_and_chunk_pdf(tmp.name)
                store_chunks_in_qdrant(chunks, uploaded.name)
            st.success(f"{uploaded.name} indexed successfully with {len(chunks)} chunks.")

    query = st.text_area("Enter your query", value="HIV SNP services in compensation/payment sections")

    if st.button("Generate Summary & Test Cases"):
        for uploaded in uploaded_files:
            st.subheader(f"📄 {uploaded.name}")
            relevant_chunks = query_relevant_chunks(query, uploaded.name)
            if not relevant_chunks:
                st.warning("No relevant content found.")
                continue

            combined_text = "\n--\n".join(relevant_chunks)
            llm = get_llm_client(
                llm_source, groq_key=groq_key, azure_key=azure_key,
                azure_url=azure_url, azure_deployment=azure_deployment
            )

            # Step 1: Summarize
            prompt_summary = f"""You are reviewing extracted paragraphs from a provider contract.

Text:
"""{combined_text}"""

Summarize HIV SNP compensation terms, services, codes, units, rates, and limits."""
            summary = llm([HumanMessage(content=prompt_summary)]).content.strip()
            st.info(summary)

            # Step 2: Generate test cases
            prompt_testcases = f"""You are a healthcare QA engineer. Based on the contract summary:

{summary}

Generate at least 6 test case scenarios in JSON list format with fields:
- Summary
- Test Scenario
- Line Number
- Requirement
- Service Type
- Revenue Code
- Diagnosis Code
- Units
- POS
- Bill Amount
- Expected Output

Respond with only a valid JSON array."""
            raw = llm([HumanMessage(content=prompt_testcases)]).content.strip()
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(0))
                    df = pd.DataFrame(json_data)
                    st.dataframe(df)
                    output_file = f"out/testcases_{uploaded.name}.xlsx"
                    df.to_excel(output_file, index=False)
                    with open(output_file, "rb") as f:
                        st.download_button("⬇️ Download Excel", f, file_name=output_file)
                except json.JSONDecodeError as e:
                    st.error(f"JSON Parse Error: {e}")
                    st.code(raw[:500])
            else:
                st.error("No valid JSON found in LLM response.")
