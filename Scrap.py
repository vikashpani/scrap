import json
from typing import List, Dict
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# --------------------------
# 1. Load and convert rules
# --------------------------

def convert_rules(json_rules: List[Dict]) -> Dict:
    """
    Convert flat list of rules from companion guide JSON
    into a nested dictionary: {Segment: {Position: {rule details}}}
    """
    rules_dict = {}
    for rule in json_rules:
        seg = rule["Segment Name"]
        pos = rule["Field Position"]
        usage = rule.get("Usage", "")
        desc = rule.get("Short Description", "")
        codes = [c["Code"] for c in rule.get("Accepted Codes", [])]

        if seg not in rules_dict:
            rules_dict[seg] = {}

        rules_dict[seg][pos] = {
            "usage": usage,
            "description": desc,
            "accepted_codes": codes
        }
    return rules_dict


# --------------------------
# 2. Setup LLM (Azure OpenAI)
# --------------------------

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # replace with your deployment
    model="gpt-4o",
    temperature=0
)


# --------------------------
# 3. Validation Logic
# --------------------------

REQUIRED_SEGMENTS = ["ISA", "GS", "ST", "BPR", "TRN", "N1", "CLP", "SE", "GE", "IEA"]

def validate_segment(segment: str, rules: Dict) -> Dict:
    """
    Validate one EDI segment and return a single dictionary
    with overall status and per-field reasons.
    """
    elements = segment.split("*")
    seg_id = elements[0].strip()

    field_reasons = {}
    overall_status = "Matched"

    # ---------- Syntax check ----------
    if not seg_id.isalpha():
        return {
            "edi_line": segment,
            "status": "Invalid",
            "rule_line": seg_id,
            "reason": {"Syntax": "Invalid segment ID (not alphabetic)"}
        }

    # ---------- Mandatory check ----------
    if seg_id in REQUIRED_SEGMENTS:
        REQUIRED_SEGMENTS.remove(seg_id)

    # ---------- Field-level validation ----------
    if seg_id in rules:
        seg_rules = rules[seg_id]
        for pos, rule in seg_rules.items():
            idx = int(pos) - 1
            desc = rule["description"]

            try:
                value = elements[idx]
            except IndexError:
                field_reasons[f"{seg_id}{pos}"] = f"Invalid - Missing required field {desc}"
                overall_status = "Invalid"
                continue

            # Required field check
            if rule.get("usage") == "Required" and not value.strip():
                field_reasons[f"{seg_id}{pos}"] = f"Invalid - {desc} is required"
                overall_status = "Invalid"
                continue

            # Accepted codes check
            if rule.get("accepted_codes"):
                if value not in rule["accepted_codes"]:
                    llm_reason = llm([HumanMessage(
                        content=f"Field {seg_id}{pos} has invalid code '{value}'. "
                                f"Valid codes are {rule['accepted_codes']}. "
                                f"Describe the error in one sentence."
                    )])
                    reason = llm_reason.content if llm_reason else f"Invalid code {value}"
                    field_reasons[f"{seg_id}{pos}"] = f"Invalid - {reason}"
                    overall_status = "Invalid"
                    continue

            # If passed all checks
            field_reasons[f"{seg_id}{pos}"] = f"Valid - {desc} = {value}"
    else:
        field_reasons[seg_id] = "No field-level rules; syntax OK"

    return {
        "edi_line": segment,
        "status": overall_status,
        "rule_line": seg_id,
        "reason": field_reasons
    }


def validate_edi_file(edi_text: str, rules: Dict) -> List[Dict]:
    """
    Validate full EDI file line by line
    """
    segments = [seg.strip() for seg in edi_text.split("~") if seg.strip()]
    all_results = [validate_segment(seg, rules) for seg in segments]

    # Final mandatory check for missing segments
    if REQUIRED_SEGMENTS:
        for seg in REQUIRED_SEGMENTS:
            all_results.append({
                "edi_line": "N/A",
                "status": "Invalid",
                "rule_line": seg,
                "reason": {seg: f"Mandatory segment {seg} missing in file"}
            })
    return all_results


# --------------------------
# 4. Run Example
# --------------------------
if __name__ == "__main__":
    # Load rules.json (list of dicts)
    with open("rules.json") as f:
        json_rules = json.load(f)

    RULES = convert_rules(json_rules)

    # Load 835 EDI file
    with open("835.edi") as f:
        edi_text = f.read()

    results = validate_edi_file(edi_text, RULES)

    # Save output to JSON
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results[:3], indent=2))  # show first 3 results






import re

def parse_segments(text):
    segments = []
    # Split on segment terminator (~) or newline if PDF extracted
    raw_segments = re.split(r'~|\n', text)
    
    for raw in raw_segments:
        raw = raw.strip()
        if not raw:
            continue
        # Segment name = first 2-3 chars
        match = re.match(r'^([A-Z0-9]{2,3})(.*)', raw)
        if not match:
            continue
        seg_name, rest = match.groups()
        elements = rest.strip().split('*')[1:] if '*' in rest else []
        
        seg_dict = {
            "segment": seg_name,
            "elements": [
                {
                    "position": str(i+1).zfill(2),
                    "text": elem.strip(),
                    "usage": "",
                    "short_description": "",
                    "accepted_codes": []
                }
                for i, elem in enumerate(elements)
            ]
        }
        segments.append(seg_dict)
    return segments





def parse_segments(docs):
    """
    Input: docs (list of LangChain Document objects with .page_content)
    Output: structured segments dict for LLM
    """
    segments = []

    for doc in docs:
        text = doc.page_content

        # Split into lines for processing
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        current_segment = None
        current_fields = []

        for line in lines:
            # If line looks like start of a new segment (e.g., ISA, GS, ST, SE, etc.)
            if len(line) >= 2 and line[:2].isalpha() and line[:2].isupper():
                # save previous segment if exists
                if current_segment:
                    segments.append({
                        "segment": current_segment,
                        "fields": current_fields
                    })
                # start new segment
                parts = line.split()
                current_segment = parts[0]  # e.g., ISA
                current_fields = [{"field_position": "01", "text": " ".join(parts[1:])}]
            
            else:
                # looks like continuation or element description
                if current_fields:
                    field_pos = str(len(current_fields) + 1).zfill(2)
                    current_fields.append({
                        "field_position": field_pos,
                        "text": line
                    })

        # last segment flush
        if current_segment:
            segments.append({
                "segment": current_segment,
                "fields": current_fields
            })

    return segments







import json
from collections import defaultdict
from openai import OpenAI

client = OpenAI()

def build_segment_dicts(elements):
    """
    Convert element-level docs into segment-wise dictionary
    """
    segments = defaultdict(list)
    for elem in elements:
        element_name = elem["Element"]  # e.g., ISA01
        segment = element_name[:3]      # ISA
        field_position = element_name[3:]  # 01
        text = elem["text"]

        segments[segment].append({
            "field_position": field_position,
            "text": text
        })
    return segments


def query_llm_for_segment(segment_name, fields):
    """
    Send one segment (with all its fields) to the LLM
    and get enriched rules back.
    """
    prompt = f"""
    You are analyzing an EDI X12 segment called {segment_name}.
    Each field has a position and some extracted text.
    For each field, return:
    - field_position
    - usage (required/optional/conditional if identifiable)
    - short_description
    - accepted_codes with description (if applicable)

    Input fields:
    {json.dumps(fields, indent=2)}

    Return JSON only, structured like:
    {{
      "segment": "{segment_name}",
      "fields": [
        {{
          "field_position": "01",
          "usage": "Required",
          "short_description": "Authorization Information Qualifier",
          "accepted_codes": {{"00": "No Authorization", "01": "Authorization Present"}}
        }}
      ]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def build_rules(elements):
    """
    Main pipeline: process all elements → segment dict → query LLM → merge into rules.json
    """
    segments = build_segment_dicts(elements)
    rules = {"segments": []}

    for segment_name, fields in segments.items():
        enriched = query_llm_for_segment(segment_name, fields)
        rules["segments"].append(enriched)

    return rules


# Example usage
elements = [
    {"Element": "ISA01", "text": "Authorization Information Qualifier"},
    {"Element": "ISA02", "text": "Authorization Information"},
    {"Element": "ISA03", "text": "Security Information Qualifier"},
    {"Element": "ISA04", "text": "Security Information"},
    {"Element": "GS01", "text": "Functional Identifier Code"},
    {"Element": "GS02", "text": "Application Sender’s Code"}
]

rules_json = build_rules(elements)

with open("rules.json", "w") as f:
    json.dump(rules_json, f, indent=2)










import re

def element_based_chunking(text: str):
    """
    Splits EDI implementation guide text into chunks per element (ISA01, ISA02...).
    """
    # Pattern: Usage + ISAxx
    element_pattern = re.compile(r"\b(REQUIRED|SITUATIONAL|NOT USED)\s+(ISA\d{2})", re.IGNORECASE)

    chunks = []
    current_chunk = []
    current_element = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = element_pattern.search(line)
        if match:
            # save previous element chunk
            if current_chunk and current_element:
                chunks.append({
                    "Element": current_element,
                    "Text": " ".join(current_chunk).strip()
                })
                current_chunk = []

            # start new element
            current_element = match.group(2)
            current_chunk.append(line)
        else:
            # continuation line
            if current_element:
                current_chunk.append(line)

    # save last
    if current_chunk and current_element:
        chunks.append({
            "Element": current_element,
            "Text": " ".join(current_chunk).strip()
        })

    return chunks











prompt = f"""
You are given part of an EDI implementation guideline (example: ISA segment fields 01–16).

Your task: Extract *all* validation rules for every field in every segment.

Output format: A JSON array of objects. No markdown, no comments, no explanations.

Each object must contain exactly these keys:
- "Segment Name": string
- "Field Position": integer (1–N, no leading zeros, numeric only)
- "Usage": one of ["Required","Situational","Not Used"]
- "Short Description": string
- "Accepted Codes": list of {{"code": string, "description": string}} (empty list if none)

STRICT INSTRUCTIONS:
1. If the segment has 16 fields, you must return **all 16 entries** (Field Position 1 through 16). Never skip a field, even if it has no accepted codes.
2. Do not merge different fields. Each field must have its own entry.
3. Capture every accepted code that appears, from tables, inline text, bullets, or parentheses. Do not drop or shorten them. If description is not available, use "" (empty string).
4. Field Position must always increase sequentially (1, 2, 3…) without missing numbers.
5. Usage must exactly match the guideline text (e.g., "Required" if marked as M, "Situational" if marked as S, "Not Used" if marked as X).
6. Short Description should be the full label or definition (not abbreviated unless that is exactly how it appears).
7. Accepted Codes list must contain all codes for that field, no duplicates, and with full descriptions.

Validation before output:
- Ensure Segment Name + Field Position combination is unique.
- Ensure numeric field positions are sequential (no skips).
- Ensure Accepted Codes are captured completely.

Return only the JSON array.

Text:
{chunk.page_content}
"""







prompt = f"""
You are given part of an EDI implementation guideline.

Goal: Extract validation rules for segments and fields with NO duplicates and NO missing accepted codes.

Output: Return ONLY a JSON array of objects. No markdown, no comments, no extra text.

For each rule include exactly these keys:
- "Segment Name": string
- "Field Position": integer (numeric only)
- "Usage": one of ["Required","Situational","Not Used"]
- "Short Description": string
- "Accepted Codes": list of {{"code": string, "description": string}} (empty list if none specified)

STRICT PROCESS (follow exactly; do not output these steps):
1) SILENTLY build a dictionary keyed by K = "<Segment Name>|<Field Position>" while reading the text.
2) When you see the same K again, MERGE into the existing entry instead of creating a new one:
   - Short Description: keep the longest/most informative version.
   - Usage: if conflicting, choose the strictest (Required > Situational > Not Used).
   - Accepted Codes: UNION of all codes found for K (no duplicates).
3) EXHAUSTIVE CODE CAPTURE:
   - Capture codes from any phrasing like "Accepted Codes", "Valid Values", "Values", "must be one of", "qualifier values", "(code – description)", tables, bullets, comma-separated lists, or text in parentheses after the field name.
   - Include codes even if descriptions are missing (use empty string for description).
   - Normalize codes by trimming spaces; keep original case as shown (usually uppercase).
4) ONLY AFTER finishing the entire text, convert the dictionary values to a JSON list.
5) Sort final list by "Segment Name" (A→Z) then by "Field Position" (ascending).
6) Ensure there is exactly ONE object per unique ("Segment Name","Field Position").

Validation before output:
- No duplicate ("Segment Name","Field Position") pairs.
- "Field Position" is an integer.
- "Accepted Codes" lists contain unique codes only.

Text:
{chunk.page_content}
"""












def parse_segment(seg: ET.Element) -> str:
    seg_id = seg.attrib["id"]
    elements = [seg_id]

    # dictionary to hold element values with proper position
    element_dict = {}

    for child in seg:
        if child.tag == "ele":
            # extract numeric part (position index)
            idx_str = child.attrib["id"].replace(seg_id, "")
            if idx_str.isdigit():
                idx = int(idx_str)
                element_dict[idx] = child.text or ""
        elif child.tag == "comp":
            idx_str = child.attrib["id"].replace(seg_id, "")
            if idx_str.isdigit():
                idx = int(idx_str)
                sub_values = [sub.text or "" for sub in child if sub.tag == "subele"]
                element_dict[idx] = ":".join(sub_values)

    # Build elements in proper order, filling missing with ""
    max_idx = max(element_dict.keys(), default=0)
    for i in range(1, max_idx + 1):
        elements.append(element_dict.get(i, ""))

    return "*".join(elements) + "~"









import xml.etree.ElementTree as ET

def parse_segment(seg: ET.Element) -> str:
    seg_id = seg.attrib["id"]

    # dictionary for elements by position
    positions = {}

    for child in seg:
        if child.tag == "ele":
            idx = int(child.attrib["id"].replace(seg_id, ""))  # e.g. CLM02 → 2
            positions[idx] = child.text or ""
        elif child.tag == "comp":
            idx = int(child.attrib["id"].replace(seg_id, ""))  # e.g. CLM05 → 5
            sub_values = [sub.text or "" for sub in child if sub.tag == "subele"]
            positions[idx] = ":".join(sub_values)

    # fill gaps with empty "" so *** appear
    max_idx = max(positions.keys())
    elements = [seg_id] + [positions.get(i, "") for i in range(1, max_idx + 1)]

    return "*".join(elements) + "~"
















#!/usr/bin/env python3
"""
Extract matched claims from large 837I/837P XMLs and build EDI .DAT files grouped by claim type + ISA06 (trading partner).

- Deduplicates identical ISA_LOOP (full ISA->IEA interchange).
- Matches claims using Excel lookup: (CLM01, NM109, CLM02, StartDate).
- Groups matched claims by (ClaimType, ISA06) and by subscriber (2000A).
- Writes proper EDI .DAT files reusing original ISA/GS/ST/.../SE/GE/IEA segments.

CONFIG: set INPUT_XML_FOLDER, EXCEL_FILE, OUTPUT_FOLDER
"""
import os
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
import pandas as pd

# -----------------------
# CONFIG - change these
# -----------------------
INPUT_XML_FOLDER = "edi_claim_extractor_source"   # folder with 837I/837P XMLs
EXCEL_FILE = "edi_claim_extractor_input_file.xlsx"
OUTPUT_FOLDER = "edi_claim_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Excel column names tolerance (try to auto-detect common variations)
EXCEL_CANDIDATES = {
    "clm01": ["Patient Control Number", "CLM01", "patient_control_number", "patient_control_no"],
    "member": ["Member ID", "NM109", "MemberID", "member_id"],
    "amount": ["Total Bill Amount", "CLM02", "TotalAmount", "total_amount"],
    "startdate": ["Start Date", "Start Date of Service", "StartDate", "start_date"]
}


# -----------------------
# Helpers
# -----------------------
def pick_excel_columns(df):
    """Return column names for clm01, member, amount, startdate by checking candidate headers."""
    cols = {k: None for k in EXCEL_CANDIDATES}
    lower_map = {c.lower().strip(): c for c in df.columns}
    for key, candidates in EXCEL_CANDIDATES.items():
        for cand in candidates:
            if cand in df.columns:
                cols[key] = cand
                break
            if cand.lower() in lower_map:
                cols[key] = lower_map[cand.lower()]
                break
    # If any missing, try fuzzy fallback: find columns containing substrings
    for key, val in cols.items():
        if val is None:
            for col in df.columns:
                lc = col.lower()
                if key == "clm01" and ("clm" in lc and "01" in lc or "control" in lc):
                    cols[key] = col; break
                if key == "member" and ("member" in lc or "nm1" in lc):
                    cols[key] = col; break
                if key == "amount" and ("amount" in lc or "clm02" in lc or "total" in lc):
                    cols[key] = col; break
                if key == "startdate" and ("date" in lc):
                    cols[key] = col; break
    return cols


def normalize_date_str(s: str) -> str:
    """Return digits-only normalized date for comparisons (e.g., 20250103 or 01032025 etc)."""
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    # remove non-digits
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits


def normalize_value(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip()


def fingerprint_element(elem: ET.Element) -> str:
    """Fingerprint an element (and its child seg/ele structure) deterministically."""
    parts = []
    for node in elem.iter():
        if node.tag == 'seg':
            segid = node.attrib.get('id', '')
            ele_texts = []
            for ele in node.findall('ele'):
                ele_texts.append((ele.text or '').strip())
            parts.append(segid + "|" + "|".join(ele_texts))
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def seg_to_edi_text(seg: ET.Element) -> str:
    """Convert <seg id='ISA'> <ele id='ISA01'>..</ele> ... </seg> -> 'ISA*ele1*ele2*...~'"""
    if seg is None:
        return ""
    segid = seg.attrib.get('id', '').strip()
    # preserve element order
    ele_texts = []
    for ele in seg.findall('ele'):
        # keep empty elements as empty strings to preserve field positions
        ele_texts.append((ele.text or '').strip())
    if ele_texts:
        return segid + "*" + "*".join(ele_texts) + "~"
    else:
        return segid + "~"


def get_seg_ele(scope: ET.Element, seg_id: str, ele_id: str) -> str:
    """Find first <seg id='seg_id'>/<ele id='ele_id'> descendant under scope and return text."""
    node = scope.find(f".//seg[@id='{seg_id}']/ele[@id='{ele_id}']")
    return (node.text or "").strip() if node is not None and node.text else ""


def find_all_isa_loops(root: ET.Element):
    """Return list of ISA_LOOP elements. Try common XPaths/fallbacks."""
    loops = root.findall(".//loop[@id='ISA_LOOP']")
    if loops:
        return loops
    # try just seg grouping
    return root.findall(".//seg[@id='ISA']/..")  # parent of ISA seg (likely a loop element)


def get_isa06_from_isa_loop(isa_loop: ET.Element) -> str:
    return get_seg_ele(isa_loop, "ISA", "ISA06")


def get_gs08_from_isa_loop(isa_loop: ET.Element) -> str:
    return get_seg_ele(isa_loop, "GS", "GS08")


def get_st_implref_from_st(st_seg: ET.Element) -> str:
    # ST3 (implementation reference) sometimes stored as ST03
    if st_seg is None:
        return ""
    node = st_seg.find("ele[@id='ST03']")
    if node is not None and (node.text or "").strip():
        return (node.text or "").strip()
    return ""


def get_dtp_start_date(loop2000b: ET.Element, target_tag: str) -> str:
    """Return DTP03 where DTP01 matches target_tag (e.g., '472' or '434')."""
    for dtp in loop2000b.findall(".//seg[@id='DTP']"):
        dtp01 = dtp.find("ele[@id='DTP01']")
        if dtp01 is not None and (dtp01.text or "").strip() == target_tag:
            dtp03 = dtp.find("ele[@id='DTP03']")
            return (dtp03.text or "").strip() if dtp03 is not None and dtp03.text else ""
    return ""


# -----------------------
# Load Excel lookup
# -----------------------
print("Loading Excel file:", EXCEL_FILE)
df = pd.read_excel(EXCEL_FILE, dtype=str).fillna("")
cols_map = pick_excel_columns(df)
if not cols_map['clm01'] or not cols_map['member'] or not cols_map['amount']:
    raise SystemExit("ERROR: Could not auto-detect necessary Excel columns. Please ensure Excel has Patient Control Number, Member ID, Total Bill Amount columns.")

excel_keys = set()
for _, row in df.iterrows():
    clm01 = normalize_value(row[cols_map['clm01']])
    nm109 = normalize_value(row[cols_map['member']])
    clm02 = normalize_value(row[cols_map['amount']])
    # start date may be optional in Excel; if present normalize
    sdate = normalize_date_str(row[cols_map['startdate']]) if cols_map['startdate'] else ""
    excel_keys.add((clm01, nm109, clm02, sdate))
print(f"Excel entries loaded: {len(excel_keys)}")

# -----------------------
# Process XML files: dedupe ISA_LOOP, match claims
# -----------------------
seen_isa_hashes = set()
# structure: claims_by_partner[(claim_type, isa06)] -> { subscriber_fp: (subscriber_2000a_elem, set_of_2000b_elems) }
claims_by_partner = defaultdict(lambda: defaultdict(lambda: {"2000a": None, "2000b_set": [], "seen_b_fp": set()}))

xml_files = [f for f in os.listdir(INPUT_XML_FOLDER) if f.lower().endswith(".xml")]
print("Found XML files:", xml_files)

for xmlf in xml_files:
    path = os.path.join(INPUT_XML_FOLDER, xmlf)
    print("Parsing:", path)
    tree = ET.parse(path)
    root = tree.getroot()

    isa_loops = find_all_isa_loops(root)
    if not isa_loops:
        print("  No ISA_LOOP found in", xmlf)
        continue

    for isa_loop in isa_loops:
        isa_fp = fingerprint_element(isa_loop)
        if isa_fp in seen_isa_hashes:
            # skip identical full interchange
            # print("Skipping duplicate ISA_LOOP fingerprint:", isa_fp[:8])
            continue
        seen_isa_hashes.add(isa_fp)

        isa06 = get_isa06_from_isa_loop(isa_loop)
        if not isa06:
            # cannot group without trading partner; skip
            # print("  No ISA06; skipping interchange")
            continue

        # find header/footer segs
        isa_seg = isa_loop.find(".//seg[@id='ISA']")
        iea_seg = isa_loop.find(".//seg[@id='IEA']")

        # find GS groups
        gs_loops = isa_loop.findall(".//loop[@id='GS_LOOP']")
        if not gs_loops:
            # fallback: try to build from GS segments under this ISA
            gs_segs = isa_loop.findall(".//seg[@id='GS']")
            for seg in gs_segs:
                l = ET.Element("loop", attrib={"id": "GS_LOOP"})
                l.append(seg)
                # try to attach GE if present under ISA
                ge = isa_loop.find(".//seg[@id='GE']")
                if ge is not None:
                    l.append(ge)
                gs_loops.append(l)

        for gs_loop in gs_loops:
            gs_seg = gs_loop.find(".//seg[@id='GS']")
            ge_seg = gs_loop.find(".//seg[@id='GE']")

            # find ST loops under this GS
            st_loops = gs_loop.findall(".//loop[@id='ST_LOOP']")
            if not st_loops:
                st_segs = gs_loop.findall(".//seg[@id='ST']")
                for sseg in st_segs:
                    st_loop = ET.Element("loop", attrib={"id": "ST_LOOP"})
                    st_loop.append(sseg)
                    # attach matching SE if exists
                    se_candidate = gs_loop.find(".//seg[@id='SE']")
                    if se_candidate is not None:
                        st_loop.append(se_candidate)
                    st_loops.append(st_loop)

            for st_loop in st_loops:
                st_seg = st_loop.find(".//seg[@id='ST']")
                se_seg = st_loop.find(".//seg[@id='SE']")
                if st_seg is None:
                    continue

                # determine claim type: prefer GS08 then ST03
                gs08 = (gs_seg.find("ele[@id='GS08']").text or "").strip() if gs_seg is not None and gs_seg.find("ele[@id='GS08']") is not None else ""
                st_impl = get_st_implref_from_st(st_seg)
                claim_type = None
                if "X222" in gs08 or "005010X222" in gs08 or "X222" in st_impl or "005010X222" in st_impl:
                    claim_type = "Professional"
                elif "X223" in gs08 or "005010X223" in gs08 or "X223" in st_impl or "005010X223" in st_impl:
                    claim_type = "Institutional"
                else:
                    # fallback: if filename contains 'p' or 'i'
                    if "p" in xmlf.lower():
                        claim_type = "Professional"
                    elif "i" in xmlf.lower():
                        claim_type = "Institutional"
                    else:
                        # unknown, skip
                        continue

                # iterate 2000A loops inside this ST
                for loop2000a in st_loop.findall(".//loop[@id='2000A']"):
                    # fingerprint subscriber-level 2000A (to dedupe later)
                    a_fp = fingerprint_element(loop2000a)

                    # we'll collect any matched 2000B under this 2000A
                    matched_b_list = []

                    for loop2000b in loop2000a.findall(".//loop[@id='2000B']"):
                        # extract CLM01, NM109, CLM02
                        clm01 = get_seg_ele(loop2000b, "CLM", "CLM01")
                        nm109 = get_seg_ele(loop2000b, "NM1", "NM109")
                        if not nm109:
                            # sometimes member id is stored under different ele id; try known variants
                            nm109 = get_seg_ele(loop2000b, "NM1", "NM1_09") or get_seg_ele(loop2000b, "NM1", "NM1_08")
                        clm02 = get_seg_ele(loop2000b, "CLM", "CLM02")

                        # get start dates for both tags
                        dtp472 = get_dtp_start_date(loop2000b, "472")  # professional
                        dtp434 = get_dtp_start_date(loop2000b, "434")  # institutional

                        # normalize strings for matching
                        n_clm01 = normalize_value(clm01)
                        n_nm109 = normalize_value(nm109)
                        n_clm02 = normalize_value(clm02)
                        n_dtp472 = normalize_date_str(dtp472)
                        n_dtp434 = normalize_date_str(dtp434)

                        # Match rules:
                        # - if ST type is Professional try match using dtp472; else dtp434
                        matched_as = None
                        if claim_type == "Professional" and n_dtp472:
                            if (n_clm01, n_nm109, n_clm02, n_dtp472) in excel_keys:
                                matched_as = "Professional"
                        elif claim_type == "Institutional" and n_dtp434:
                            if (n_clm01, n_nm109, n_clm02, n_dtp434) in excel_keys:
                                matched_as = "Institutional"
                        # fallback: try matching ignoring dtp if Excel startdate empty or not provided
                        if not matched_as:
                            # try match where Excel maybe has empty start date
                            if (n_clm01, n_nm109, n_clm02, "") in excel_keys:
                                matched_as = claim_type

                        if matched_as:
                            # we will keep this 2000B
                            matched_b_list.append(loop2000b)

                    # if we found any matched 2000B under this 2000A, record them grouped by (claim_type, isa06)
                    if matched_b_list:
                        key = (claim_type, isa06)
                        # ensure this subscriber (by a_fp) is recorded
                        entry = claims_by_partner[key][a_fp]
                        if entry["2000a"] is None:
                            # store a shallow copy of the 2000A loop without inner 2000B children (we will append matched ones)
                            # Create new element and copy seg children except nested 2000B loops
                            a_copy = ET.Element("loop", attrib={"id": "2000A"})
                            for seg in loop2000a:
                                # if seg is a loop 2000B skip; if seg is seg element, append
                                if seg.tag == "loop" and seg.attrib.get("id","").startswith("2000B"):
                                    continue
                                a_copy.append(seg)
                            entry["2000a"] = a_copy
                        # For each matched 2000B, append if not duplicate
                        for b in matched_b_list:
                            b_fp = fingerprint_element(b)
                            if b_fp not in entry["seen_b_fp"]:
                                entry["seen_b_fp"].add(b_fp)
                                # append a deep copy of b (to avoid references)
                                entry["2000b_set"].append(b)

# -----------------------
# Build and write .DAT files
# -----------------------
def write_dat_files(claims_by_partner_map, out_dir):
    today = datetime.now().strftime("%m%d%Y")
    for (claim_type, isa06), subs_dict in claims_by_partner_map.items():
        # collect all non-empty subscribers
        subs = [v for v in subs_dict.values() if v["2000b_set"]]
        if not subs:
            continue

        # Use header/footer from first matched subscriber's stored tuple? We stored only 2000a/2000b; need header from original scanning
        # But in our earlier collection we didn't keep isa/gs/st/se/ge/iea per key to avoid repetition.
        # For this script we will find the first occurrence in XML files again to get header/footer segments.
        # Simpler: reconstruct header/footer from any matched 2000b's ancestor segs.

        # For header reuse, extract the ISA/GS/ST/SE/GE/IEA segments from the original 2000a/2000b elements' ancestors.
        # We'll find ancestor segs by searching upwards — but ElementTree does not have parent.
        # Instead, to keep it simple and robust: search the entire XML folder again for the first ISA_LOOP that contains any of these 2000B fingerprints and reuse its headers.
        header_isa_seg = None
        header_gs_seg = None
        header_st_seg = None
        footer_se_seg = None
        footer_ge_seg = None
        footer_iea_seg = None

        # find one original location for header/footer (search input XMLs)
        found_header = False
        # Build set of b fingerprints for search
        all_b_fps = set()
        for sub in subs:
            for b in sub["2000b_set"]:
                all_b_fps.add(fingerprint_element(b))

        # search xml files to find a matching containership
        for xmlf in xml_files:
            path = os.path.join(INPUT_XML_FOLDER, xmlf)
            tree = ET.parse(path)
            root = tree.getroot()
            isa_loops = find_all_isa_loops(root)
            for isa_loop in isa_loops:
                isa_loop_fps_b = set()
                for loop2000b in isa_loop.findall(".//loop[@id='2000B']"):
                    isa_loop_fps_b.add(fingerprint_element(loop2000b))
                if all_b_fps & isa_loop_fps_b:
                    # use header/footer from this isa_loop
                    header_isa_seg = isa_loop.find(".//seg[@id='ISA']")
                    header_gs_seg = isa_loop.find(".//seg[@id='GS']")
                    header_st_seg = isa_loop.find(".//seg[@id='ST']")
                    footer_se_seg = isa_loop.find(".//seg[@id='SE']")
                    footer_ge_seg = isa_loop.find(".//seg[@id='GE']")
                    footer_iea_seg = isa_loop.find(".//seg[@id='IEA']")
                    found_header = True
                    break
            if found_header:
                break

        # fallback: if not found, skip writing (shouldn't happen)
        if not found_header:
            print("Warning: could not locate header/footer for", claim_type, isa06)
            continue

        # Build EDI lines: ISA, GS, ST, optional HEADER segs, then 2000A blocks with their included 2000B children, then SE, GE, IEA
        lines = []
        lines.append(seg_to_edi_text(header_isa_seg))
        lines.append(seg_to_edi_text(header_gs_seg))
        lines.append(seg_to_edi_text(header_st_seg))
        # optional: if any 'HEADER' seg exists under header_st_seg's context, include - try find under st siblings
        header_seg = None
        # search for 'HEADER' seg near ST (under same ST_LOOP)
        # Try: find seg id HEADER under the root (first instance)
        header_seg_generic = None
        for xmlf in xml_files:
            path = os.path.join(INPUT_XML_FOLDER, xmlf)
            tree = ET.parse(path)
            root = tree.getroot()
            h = root.find(".//seg[@id='HEADER']")
            if h is not None:
                header_seg_generic = h
                break
        if header_seg_generic is not None:
            lines.append(seg_to_edi_text(header_seg_generic))

        # append unique 2000A blocks with their matched 2000B children
        for sub in subs:
            a_elem = sub["2000a"]
            if a_elem is None:
                continue
            # append segs inside a_elem (non-loop segs)
            for seg in a_elem.findall(".//seg"):
                lines.append(seg_to_edi_text(seg))
            # append each matched 2000B under this subscriber
            for b in sub["2000b_set"]:
                # b may be a <loop id='2000B'> containing many seg children
                for seg in b.findall(".//seg"):
                    lines.append(seg_to_edi_text(seg))

        # append trailers
        if footer_se_seg is not None:
            lines.append(seg_to_edi_text(footer_se_seg))
        if footer_ge_seg is not None:
            lines.append(seg_to_edi_text(footer_ge_seg))
        if footer_iea_seg is not None:
            lines.append(seg_to_edi_text(footer_iea_seg))

        # write file
        fname = f"{claim_type}_{isa06}_{today}.DAT"
        outpath = os.path.join(out_dir, fname)
        with open(outpath, "w", encoding="utf-8") as w:
            # join with newline for human readability; EDI segment terminator `~` is already included
            w.write("\n".join(lines))
        print(f"Wrote {len(subs)} subscribers -> {outpath}")

write_dat_files(claims_by_partner, OUTPUT_FOLDER)
print("Completed. Outputs in:", OUTPUT_FOLDER)







import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

# -------------------
# Config
# -------------------
input_xml_folder = "edi_claim_extractor_source"
excel_file = "edi_claim_extractor_input_file.xlsx"
output_folder = "edi_claim_outputs"
os.makedirs(output_folder, exist_ok=True)

# -------------------
# Load Excel claims of interest
# -------------------
df = pd.read_excel(excel_file, dtype=str).fillna("")

claims_lookup = set(
    (str(row["Patient Control Number"]).strip(),
     str(row["Member ID"]).strip(),
     str(row["Total Bill Amount"]).strip(),
     str(row["Start Date"]).strip())
    for _, row in df.iterrows()
)

# -------------------
# Utilities
# -------------------
def get_text(element, path):
    node = element.find(path)
    return node.text.strip() if node is not None and node.text else ""

def match_claim(clm01, nm109, clm02, start_date):
    return (clm01.strip(), nm109.strip(), clm02.strip(), start_date.strip()) in claims_lookup

# -------------------
# Process XML files
# -------------------
claims_by_partner = defaultdict(lambda: {"Professional": [], "Institutional": []})

for file in os.listdir(input_xml_folder):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_xml_folder, file))
    root = tree.getroot()

    # Extract trading partner from ISA06
    isa06 = get_text(root, ".//ISA06")
    if not isa06:
        continue

    # Determine claim type by looking at ST segment
    st_val = get_text(root, ".//ST01")
    if st_val == "837":
        # Professional vs Institutional indicator
        claim_type = "Professional" if "HC" in file else "Institutional"
    else:
        continue

    # Get headers and footers
    isa = root.find(".//ISA")
    gs = root.find(".//GS")
    st = root.find(".//ST")
    se = root.find(".//SE")
    ge = root.find(".//GE")
    iea = root.find(".//IEA")

    # Traverse 2000A → 2000B
    for loop2000a in root.findall(".//2000A"):
        a_copy = ET.Element("2000A", attrib=loop2000a.attrib)
        added_b = False

        for loop2000b in loop2000a.findall(".//2000B"):
            clm01 = get_text(loop2000b, ".//CLM01")
            nm109 = get_text(loop2000b, ".//NM109")
            clm02 = get_text(loop2000b, ".//CLM02")

            # DTP start date (472 for professional, 434 for institutional)
            dtp_tag = "472" if claim_type == "Professional" else "434"
            start_date = ""
            for dtp in loop2000b.findall(".//DTP"):
                if dtp_tag in (dtp.text or ""):
                    start_date = dtp.find("DTP03").text if dtp.find("DTP03") is not None else ""

            if match_claim(clm01, nm109, clm02, start_date):
                a_copy.append(loop2000b)
                added_b = True

        if added_b:
            claims_by_partner[isa06][claim_type].append((isa, gs, st, a_copy, se, ge, iea))

# -------------------
# Write outputs
# -------------------
today = datetime.now().strftime("%m%d%Y")

for partner, types in claims_by_partner.items():
    for claim_type, blocks in types.items():
        if not blocks:
            continue

        # Build new XML root
        new_root = ET.Element("EDI")
        # take header/footer from first block
        isa, gs, st, _, se, ge, iea = blocks[0]
        new_root.append(isa)
        new_root.append(gs)
        new_root.append(st)

        # add all unique 2000A
        seen_a = set()
        for _, _, _, a_block, _, _, _ in blocks:
            a_str = ET.tostring(a_block)
            if a_str not in seen_a:
                seen_a.add(a_str)
                new_root.append(a_block)

        new_root.append(se)
        new_root.append(ge)
        new_root.append(iea)

        out_file = f"{claim_type}_{partner}_{today}.DAT"
        ET.ElementTree(new_root).write(os.path.join(output_folder, out_file), encoding="utf-8", xml_declaration=True)

print(f"✅ Extraction completed. Output written to {output_folder}")












import os
import xml.etree.ElementTree as ET
from io import StringIO
from x12.core import x12n_document, ParamsBase

def change_file_into_segment_list(edi_file_path):
    with open(edi_file_path, 'r', encoding='utf-8') as file:
        segment_list = [line.strip() for line in file.readlines()]
    return segment_list

def load_edi_segments_as_xml_obj(edi_segments_list):
    edibuffer = StringIO('\n'.join(edi_segments_list))
    param = ParamsBase()
    xmlbuffer = StringIO()
    x12n_document(param, edibuffer, fd_997=None, fd_html=None, fd_xmldoc=xmlbuffer, map_path=None)
    xmlbuffer.seek(0)
    rootnode = ET.fromstring(xmlbuffer.getvalue())
    return rootnode

def identify_edi_type(segments):
    """
    Returns 'I' for Institutional, 'P' for Professional, or None if unsupported (like Dental).
    """
    for seg in segments:
        if seg.startswith("ST*837"):
            if "005010X223" in seg:  # Institutional
                return "I"
            elif "005010X222" in seg:  # Professional
                return "P"
            elif "005010X224" in seg:  # Dental (skip)
                return None
    return None

def build_combined_xml(input_folder, output_folder):
    institutional_root = ET.Element("Institutional837")
    professional_root = ET.Element("Professional837")

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt") and not filename.endswith(".edi"):
            continue

        file_path = os.path.join(input_folder, filename)
        segments = change_file_into_segment_list(file_path)
        edi_type = identify_edi_type(segments)

        if edi_type is None:  # Skip dental or unknown
            continue

        rootnode = load_edi_segments_as_xml_obj(segments)

        if edi_type == "I":
            institutional_root.append(rootnode)
        elif edi_type == "P":
            professional_root.append(rootnode)

    # Write Institutional
    if len(institutional_root):
        inst_tree = ET.ElementTree(institutional_root)
        inst_tree.write(os.path.join(output_folder, "837I_combined.xml"), encoding="utf-8", xml_declaration=True)

    # Write Professional
    if len(professional_root):
        prof_tree = ET.ElementTree(professional_root)
        prof_tree.write(os.path.join(output_folder, "837P_combined.xml"), encoding="utf-8", xml_declaration=True)

    print("✅ Combined XMLs generated successfully.")















"""
Combine multiple X12 EDI .dat files into TWO XML outputs (Institutional 837I vs Professional 837P)
using pyx12's x12n_document.

Features:
- Scans a source folder for EDI files (*.dat).
- Auto-detects transaction set version (ICN/ICVN) from ISA/GS/ST segments.
- Loads the correct pyx12 map folder automatically.
- Separates Institutional (837I, e.g. 005010X223A2) and Professional (837P, e.g. 005010X222A1).
- Writes two XML files: combined_institutional.xml and combined_professional.xml.

Usage:
------
python combine_edi_to_xml.py \
    --source "/path/to/source/folder" \
    --outdir "/path/to/output/folder"

Notes:
------
- The 2000A/2000B loops are preserved by pyx12.
- Institutional vs Professional are split into two separate XML files.
"""

from __future__ import annotations
import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET

# --- pyx12 imports ---
try:
    from pyx12.params import params
    from pyx12.x12n_document import x12n_document
except Exception as e:
    print("[ERROR] pyx12 not available or unexpected import error:", e, file=sys.stderr)
    raise


def find_edi_files(src_dir: Path, patterns: Tuple[str, ...] = ("*.dat",)) -> Iterable[Path]:
    for pat in patterns:
        yield from src_dir.rglob(pat)


def detect_version_and_txset(edi_path: Path) -> Tuple[str, str, str]:
    """
    Detect version (ICVN), transaction set (ST01), and convention (ST03).
    Returns (icvn, txset, implref).
    """
    with open(edi_path, "rt", encoding="utf-8", errors="ignore") as f:
        content = f.read(5000)
    segments = content.split("~")
    icvn = ""
    txset = ""
    implref = ""
    for seg in segments:
        fields = seg.split("*")
        if fields[0] == "GS" and len(fields) > 8:
            icvn = fields[8]
        elif fields[0] == "ST" and len(fields) > 2:
            txset = fields[1]
            implref = fields[2]
        if icvn and txset:
            break
    return icvn, txset, implref


def get_default_map_dir(icvn: str) -> Path:
    import pyx12
    base = Path(pyx12.__file__).resolve().parent / "maps"
    return base / icvn[:6]


def convert_edi_to_xml_bytes(edi_path: Path, p: "params") -> bytes:
    buf = io.BytesIO()
    with open(edi_path, "rb") as fp_in:
        x12n_document(fp_in, buf, p, str(edi_path))
    return buf.getvalue()


def parse_xml_bytes(xml_bytes: bytes) -> ET.Element:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        snippet = xml_bytes[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to parse XML: {e}\nSnippet:\n{snippet}")
    return root


def build_combined_tree(doc_roots: Iterable[Tuple[Path, ET.Element]]) -> ET.ElementTree:
    combined_root = ET.Element("CombinedX12")
    for src_path, root in doc_roots:
        wrapper = ET.SubElement(combined_root, "Document", attrib={"file": str(src_path)})
        wrapper.append(root)
    return ET.ElementTree(combined_root)


def pretty_write(tree: ET.ElementTree, out_path: Path) -> None:
    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def combine_folder(src_dir: Path, out_dir: Path, file_patterns: Tuple[str, ...] = ("*.dat",)) -> Tuple[int, int]:
    edi_files = list(find_edi_files(src_dir, file_patterns))
    if not edi_files:
        raise FileNotFoundError(f"No EDI files found in {src_dir} matching {file_patterns}")

    institutional_roots = []
    professional_roots = []

    for edi_path in sorted(edi_files):
        icvn, txset, implref = detect_version_and_txset(edi_path)
        if txset != "837":
            continue  # skip non-837s

        map_dir = get_default_map_dir(icvn)
        p = params()
        p.set("map_path", str(map_dir))
        p.set("icvn", icvn)

        xml_bytes = convert_edi_to_xml_bytes(edi_path, p)
        root = parse_xml_bytes(xml_bytes)

        # Classify Professional vs Institutional based on implementation reference
        if "X222" in icvn or "X222" in implref:
            professional_roots.append((edi_path, root))
        elif "X223" in icvn or "X223" in implref:
            institutional_roots.append((edi_path, root))
        else:
            # Default to professional if not clearly institutional
            professional_roots.append((edi_path, root))

    if institutional_roots:
        inst_tree = build_combined_tree(institutional_roots)
        pretty_write(inst_tree, out_dir / "combined_institutional.xml")
    if professional_roots:
        prof_tree = build_combined_tree(professional_roots)
        pretty_write(prof_tree, out_dir / "combined_professional.xml")

    return len(institutional_roots), len(professional_roots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine multiple EDI files into two XMLs: Institutional vs Professional")
    parser.add_argument("--source", required=True, type=Path, help="Folder containing EDI .dat files")
    parser.add_argument("--outdir", required=True, type=Path, help="Folder to write XML outputs")
    parser.add_argument("--glob", nargs="+", default=["*.dat"], help="Glob patterns to include (default: *.dat)")
    args = parser.parse_args(argv)

    inst_count, prof_count = combine_folder(src_dir=args.source, out_dir=args.outdir, file_patterns=tuple(args.glob))
    print(f"Wrote {inst_count} institutional and {prof_count} professional files into {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())













import os
import pandas as pd
from pyx12.x12file import X12Reader
from pyx12.writer import X12Writer
from datetime import datetime

# Load claims of interest from Excel
claims_df = pd.read_excel("edi_claim_extractor_input_file.xlsx")

# Create dict for fast lookup
claims_lookup = {
    (str(row["PatientControlNumber"]),
     str(row["MemberID"]),
     str(row["TotalBillAmount"]),
     str(row["StartDateOfService"]))
    for _, row in claims_df.iterrows()
}

source_folder = "edi_claim_extractor_source_folder"
output_folder = "Extracted_EDIs"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(source_folder):
    if not file.endswith(".edi"):
        continue
    
    with open(os.path.join(source_folder, file), 'r') as f:
        edi = X12Reader(f)

        current_claim = {}
        collected_claims = {}
        isa_sender = None
        st_type = None

        for seg in edi:
            seg_id = seg.get_seg_id()

            if seg_id == "ISA":
                isa_sender = seg.get_value('ISA06')

            if seg_id == "ST":
                st_type = seg.get_value('ST01')  # 837P or 837I

            if seg_id == "CLM":
                current_claim["PatientControlNumber"] = seg.get_value("CLM01")
                current_claim["TotalBillAmount"] = seg.get_value("CLM02")

            if seg_id == "NM1" and seg.get_value("NM101") == "IL":  # Subscriber
                current_claim["MemberID"] = seg.get_value("NM109")

            if seg_id == "DTP":
                if seg.get_value("DTP01") in ["472", "434"]:  # Service dates
                    dt_str = seg.get_value("DTP03")
                    current_claim["StartDateOfService"] = dt_str

            if seg_id == "SE":  # End of claim
                key = (current_claim.get("PatientControlNumber"),
                       current_claim.get("MemberID"),
                       current_claim.get("TotalBillAmount"),
                       current_claim.get("StartDateOfService"))

                if key in claims_lookup:
                    claim_type = "Professional" if st_type == "837P" else "Institutional"
                    collected_claims.setdefault((claim_type, isa_sender), []).append(edi.copy_current_loop())

                current_claim = {}

        # Write filtered EDI files
        for (claim_type, partner), claims in collected_claims.items():
            out_file = f"{claim_type}_{partner}_{datetime.now().strftime('%m%d













import os
import re
import json
from typing import List, Dict, Any, Tuple
from langchain_openai import AzureChatOpenAI

# =========================
# Azure OpenAI (LangChain)
# =========================
def get_azure_llm() -> AzureChatOpenAI:
    """
    Configure Azure OpenAI via LangChain. Reads creds from environment.
    Required env vars:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION (e.g. 2024-05-01-preview)
      - AZURE_OPENAI_DEPLOYMENT (e.g. "gpt-4o")
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing Azure OpenAI configuration. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment,
        temperature=0,
    )
    return llm

# =========================
# EDI Parsing Utilities
# =========================
def parse_edi_file(edi_text: str) -> List[Dict[str, Any]]:
    """
    Split by '~' into segments. Each segment split by '*'.
    Returns list of dicts: {segment, line, fields}
    - fields[0] is the segment name (e.g., 'ISA')
    - positions 1..n map to fields[1..n]
    """
    segments = []
    for raw in edi_text.split("~"):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("*")
        seg = parts[0].strip()
        segments.append({"segment": seg, "line": line, "fields": parts})
    return segments

def normalize_position(field_position: Any, segment_name: str) -> int:
    """
    Normalize a rule's FieldPosition to a 1-based integer:
    Accepts: 1, "1", "01", "001", "ISA01", "ISA1", "SegmentName01", "NM101", etc.
    Returns integer position (1..N). Raises ValueError if no digits found.
    """
    if field_position is None:
        raise ValueError("FieldPosition is None")

    s = str(field_position).strip()

    # If it's purely digits (e.g., "1", "01", "001")
    if s.isdigit():
        return int(s)

    # Common forms: "ISA01", "ISA1", "NM101", "REF02", "SegmentName15"
    # Extract trailing digits
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))

    # Edge case: "segmentname01" with varying case
    s_upper = s.upper()
    seg_upper = segment_name.upper()
    if s_upper.startswith(seg_upper):
        tail = s[len(segment_name):]
        m2 = re.search(r"(\d+)$", tail)
        if m2:
            return int(m2.group(1))

    raise ValueError(f"Cannot normalize FieldPosition='{field_position}' for {segment_name}")

def unique_rule_positions(rules_for_segment: List[Dict[str, Any]], segment_name: str) -> List[int]:
    """
    Deduplicate rule positions that are the same spot with different notations (01, 1, ISA01, etc.)
    Returns sorted unique integer positions.
    """
    pos_set = set()
    for r in rules_for_segment:
        try:
            p = normalize_position(r.get("FieldPosition"), segment_name)
            pos_set.add(p)
        except Exception:
            # Ignore malformed positions; we won't validate those
            pass
    return sorted(pos_set)

# =========================
# Validation Core
# =========================
def validate_required(value: str) -> Tuple[bool, str]:
    if value is None:
        return False, "Required field missing (no element at this position)."
    if str(value).strip() == "":
        return False, "Required field empty."
    return True, "Required field present."

def validate_not_used(value: str) -> Tuple[bool, str]:
    if value is None or str(value).strip() == "":
        return True, "Field correctly not used."
    return False, f"Field marked 'Not Used' but value '{value}' found."

def validate_situational_with_llm(
    llm: AzureChatOpenAI,
    segment_name: str,
    position: int,
    value: str,
    description: str,
    all_fields: List[str],
    edi_line: str
) -> Dict[str, Any]:
    """
    Ask LLM to judge situational validity following your rules:
      - If description has an explicit condition referencing other fields, obey it.
      - If no explicit condition is present in description, BOTH present or absent are valid.
      - 'Not Used' logic never applies here (this is situational branch).
      - Treat standard EDI codes like '00','01','ZZ','T','P' as legitimate values (not 'empty').
    """
    # Build field map for clarity (1-based)
    field_map = {f"{segment_name}{i}": (all_fields[i] if i < len(all_fields) else "")
                 for i in range(1, len(all_fields))}

    prompt = f"""
You are an EDI validator. Validate one situational field based ONLY on the given description and values.

Rules for interpretation:
- "Situational" = conditionally required. If the description specifies a condition (e.g., depends on another field), check it.
- If NO explicit condition is present in the description, BOTH present and absent are valid.
- Do not treat legitimate codes like "00", "01", "ZZ", "T", "P" as empty.
- Only use information from the provided inputs. Do not invent extra conditions.

Segment: {segment_name}
Field Position: {position}
Field Tag: {segment_name}{position}
Field Value: {value}
Description: {description}

All Fields (1-based map):
{json.dumps(field_map, indent=2)}

EDI Line:
{edi_line}

Respond ONLY JSON, no markdown, no extra text:
{{
  "status": "Matched" or "Invalid",
  "rule_line": "{segment_name}{position}",
  "reason": "one-line reason"
}}
""".strip()

    resp = llm.invoke(prompt)
    content = (resp.content or "").strip()
    try:
        data = json.loads(content)
        # Guardrails for schema
        st = data.get("status")
        rl = data.get("rule_line", f"{segment_name}{position}")
        rs = data.get("reason", "Situational check done.")
        if st not in ("Matched", "Invalid"):
            raise ValueError("Bad status from LLM.")
        return {"status": st, "rule_line": rl, "edi_line": edi_line, "reason": rs}
    except Exception:
        # Fallback: if LLM fails, default to permissive per your policy
        ok = (value is None) or (str(value).strip() == "") or (str(value).strip() != "")
        return {
            "status": "Matched" if ok else "Invalid",
            "rule_line": f"{segment_name}{position}",
            "edi_line": edi_line,
            "reason": "Situational check fallback: no explicit condition enforced."
        }

def validate_segment_against_rules(
    segment: Dict[str, Any],
    rules_for_segment: List[Dict[str, Any]],
    llm: AzureChatOpenAI
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validate one parsed segment against its rules.
    Returns:
      - field_results: list of {status, rule_line, edi_line, reason}
      - segment_summary: {status, ediline, reason}
    """
    seg_name = segment["segment"]
    line = segment["line"]
    fields = segment["fields"]  # fields[0] == seg_name; positions start at 1

    # Deduplicate positions & pre-check for extra elements
    rule_positions = unique_rule_positions(rules_for_segment, seg_name)
    max_rule_pos = max(rule_positions) if rule_positions else 0

    field_results: List[Dict[str, Any]] = []

    # If there are more elements than rules cover (beyond max position), mark as extras
    if len(fields) - 1 > max_rule_pos:
        # For each extra position > max_rule_pos, mark as invalid (no rule defined)
        for i in range(max_rule_pos + 1, len(fields)):
            field_results.append({
                "status": "Invalid",
                "rule_line": f"{seg_name}{i}",
                "edi_line": line,
                "reason": f"Extra element at position {i} without a defined rule."
            })

    # Validate per rule
    for rule in rules_for_segment:
        usage = str(rule.get("Usage", "")).strip().lower()
        desc = str(rule.get("ShortDescription", "")).strip()
        try:
            pos = normalize_position(rule.get("FieldPosition"), seg_name)
        except Exception:
            # If we cannot normalize, skip but record an issue
            field_results.append({
                "status": "Invalid",
                "rule_line": f"{seg_name}{rule.get('FieldPosition')}",
                "edi_line": line,
                "reason": "Unrecognized FieldPosition format in rules."
            })
            continue

        # Fetch value at position (1-based)
        value = fields[pos] if pos < len(fields) else None
        rule_line = f"{seg_name}{pos}"

        # Required
        if usage == "required":
            ok, reason = validate_required(value)
            field_results.append({
                "status": "Matched" if ok else "Invalid",
                "rule_line": rule_line,
                "edi_line": line,
                "reason": reason
            })
            continue

        # Not Used
        if usage == "not used":
            ok, reason = validate_not_used(value)
            field_results.append({
                "status": "Matched" if ok else "Invalid",
                "rule_line": rule_line,
                "edi_line": line,
                "reason": reason
            })
            continue

        # Situational → LLM
        if usage == "situational":
            field_results.append(
                validate_situational_with_llm(
                    llm=llm,
                    segment_name=seg_name,
                    position=pos,
                    value=value if value is not None else "",
                    description=desc,
                    all_fields=fields,
                    edi_line=line
                )
            )
            continue

        # Any other usage → treat as optional/valid
        field_results.append({
            "status": "Matched",
            "rule_line": rule_line,
            "edi_line": line,
            "reason": "Optional field (no strict validation)."
        })

    # Build segment summary
    invalids = [r for r in field_results if r["status"] == "Invalid"]
    if not invalids:
        segment_summary = {
            "status": "Matched",
            "ediline": line,
            "reason": "All required, situational, and not-used fields validated successfully."
        }
    else:
        segment_summary = {
            "status": "Invalid",
            "ediline": line,
            "reason": "; ".join(dict.fromkeys([r["reason"] for r in invalids]))  # unique-preserving
        }

    return field_results, segment_summary

# =========================
# Segment Summary via LLM
# =========================
def summarize_segment_with_llm(
    llm: AzureChatOpenAI,
    edi_line: str,
    field_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Ask LLM to produce the final single-dict summary with:
      - status: "Matched" if all fields matched, else "Invalid"
      - ediline: the original segment line
      - reason: one-line reason (either all good, or concise failures)
    """
    prompt = f"""
Summarize these field-level EDI validation results into ONE JSON dictionary.
Rules:
- If ALL fields are "Matched" -> status = "Matched".
- If ANY field is "Invalid" -> status = "Invalid".
- 'ediline' must equal the original EDI line verbatim.
- 'reason' must be a single concise line. If invalid, mention the most important failure(s).

EDI line:
{edi_line}

Field results:
{json.dumps(field_results, indent=2)}

Respond ONLY JSON, no markdown:
{{
  "status": "Matched" or "Invalid",
  "ediline": "{edi_line}",
  "reason": "<one line>"
}}
""".strip()

    resp = llm.invoke(prompt)
    content = (resp.content or "").strip()
    try:
        data = json.loads(content)
        # guardrails
        if "status" not in data or "ediline" not in data or "reason" not in data:
            raise ValueError("Missing keys.")
        if data["status"] not in ("Matched", "Invalid"):
            raise ValueError("Bad status.")
        return data
    except Exception:
        # Local fallback mirroring the same rules
        any_invalid = any(r["status"] == "Invalid" for r in field_results)
        if any_invalid:
            return {
                "status": "Invalid",
                "ediline": edi_line,
                "reason": "; ".join(dict.fromkeys([r["reason"] for r in field_results if r["status"] == "Invalid"]))
            }
        return {
            "status": "Matched",
            "ediline": edi_line,
            "reason": "All required, situational, and not-used fields validated successfully."
        }

# =========================
# Example Driver
# =========================
def load_rules_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Expects a list of dicts, each having:
      - SegmentName
      - FieldPosition (e.g., 1, "01", "ISA01", "NM101")
      - Usage ("Required" | "Situational" | "Not Used" | ...)
      - ShortDescription
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_validation(edi_text: str, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates all segments in the EDI text with their respective rules.
    Returns list of per-segment summaries (each: {status, ediline, reason}).
    """
    llm = get_azure_llm()
    segments = parse_edi_file(edi_text)

    # Group rules by segment
    rules_by_segment: Dict[str, List[Dict[str, Any]]] = {}
    for r in rules:
        seg = str(r.get("SegmentName", "")).strip()
        if not seg:
            continue
        rules_by_segment.setdefault(seg, []).append(r)

    summaries: List[Dict[str, Any]] = []

    for seg in segments:
        seg_name = seg["segment"]
        if seg_name not in rules_by_segment:
            # No rules for this segment -> single Invalid record or treat as pass?
            # We'll mark as Invalid with reason for transparency.
            summaries.append({
                "status": "Invalid",
                "ediline": seg["line"],
                "reason": f"No rules defined for segment '{seg_name}'."
            })
            continue

        field_results, local_summary = validate_segment_against_rules(
            segment=seg,
            rules_for_segment=rules_by_segment[seg_name],
            llm=llm
        )

        # Option: Let LLM craft the final single-dict summary (with fallback)
        final_summary = summarize_segment_with_llm(llm, seg["line"], field_results)

        # Ensure we always return exactly the requested keys/types
        summaries.append({
            "status": final_summary["status"],
            "ediline": final_summary["ediline"],
            "reason": final_summary["reason"]
        })

    return summaries

# =========================
# __main__
# =========================
if __name__ == "__main__":
    # ---- Example usage ----
    # 1) Load rules from a local JSON file (produced from your PDF pipeline)
    #    e.g., rules.json = [
    #      {"SegmentName":"ISA","FieldPosition":"ISA01","Usage":"Required","ShortDescription":"Authorization Information Qualifier"},
    #      {"SegmentName":"ISA","FieldPosition":"ISA14","Usage":"Situational","ShortDescription":"Acknowledgment can be requested through data element ISA14."},
    #      {"SegmentName":"ISA","FieldPosition":"ISA15","Usage":"Required","ShortDescription":"Test Indicator"},
    #      {"SegmentName":"ISA","FieldPosition":"ISA16","Usage":"Required","ShortDescription":"Component Element Separator"}
    #    ]
    RULES_PATH = "rules.json"
    rules_list = load_rules_from_json(RULES_PATH)

    # 2) Read an EDI file (raw text)
    #    Example single line shown; in real use, read a whole file with multiple segments.
    edi_text = (
        "ISA*00*          *00*          *ZZ*SENDERID      *ZZ*RECEIVERID    *250101*1253*^*00501*000000905*1*T*:~"
        "GS*HC*SENDER*RECEIVER*20250101*1253*1*X*005010X222A1~"
        "ST*837*0001*005010X222A1~"
    )

    # 3) Run validation (returns list of per-segment summaries)
    summaries = run_validation(edi_text, rules_list)

    # 4) Print results
    print(json.dumps(summaries, indent=2))
















import json

isa_rules = [
    {'SegmentName': 'ISA', 'FieldPosition': 1, 'Usage': 'Required',
     'ShortDescription': 'Authorization Information Qualifier'},
    {'SegmentName': 'ISA', 'FieldPosition': 2, 'Usage': 'Required',
     'ShortDescription': 'Authorization Information'},
    {'SegmentName': 'ISA', 'FieldPosition': 3, 'Usage': 'Required',
     'ShortDescription': 'Security Information Qualifier'},
    {'SegmentName': 'ISA', 'FieldPosition': 4, 'Usage': 'Required',
     'ShortDescription': 'Security Information'},
    {'SegmentName': 'ISA', 'FieldPosition': 14, 'Usage': 'Situational',
     'ShortDescription': 'Acknowledgment requested'},
    {'SegmentName': 'ISA', 'FieldPosition': 15, 'Usage': 'Required',
     'ShortDescription': 'Test Indicator'},
    {'SegmentName': 'ISA', 'FieldPosition': 16, 'Usage': 'Required',
     'ShortDescription': 'Subelement Separator'}
]

edi_line = "ISA*00*          *01*SECRET    *ZZ*SUBMITTERS.ID*ZZ*RECEIVERS.ID*030101*1253*^*00501*000000905*1*T*:~"
fields = edi_line.split("*")

def validate_isa(fields, rules):
    results = []
    for rule in rules:
        pos = rule['FieldPosition']
        idx = pos  # our rules already give int position
        
        # Prepare rule line text
        rule_line = f"{rule['SegmentName']}{pos} - {rule['ShortDescription']} ({rule['Usage']})"
        
        if idx >= len(fields):
            results.append({
                "status": "Invalid",
                "rule_line": rule_line,
                "reason": "Field missing in EDI"
            })
            continue

        value = fields[idx].strip()
        if rule['Usage'] == "Required" and not value:
            results.append({
                "status": "Invalid",
                "rule_line": rule_line,
                "reason": "Required field empty"
            })
        else:
            results.append({
                "status": "Matched",
                "rule_line": rule_line,
                "reason": "Field present and valid"
            })
    return results

# Run validation
validation_output = validate_isa(fields, isa_rules)

# Pretty print JSON
print(json.dumps(validation_output, indent=2))









def validate_isa(fields, rules):
    errors = []
    for rule in rules:
        pos = rule['FieldPosition']
        # Handle if rule uses numbers
        if isinstance(pos, int):
            idx = pos  # field index in split
        elif pos.startswith("ISA"):
            idx = int(pos.replace("ISA", ""))  # e.g. ISA14 → 14
        else:
            continue

        if idx >= len(fields):
            errors.append(f"Missing field {pos}: {rule['ShortDescription']}")
            continue

        value = fields[idx]
        if rule['Usage'] == "Required" and not value.strip():
            errors.append(f"Field {pos} ({rule['ShortDescription']}) is required but empty")

    return errors















prompt = f"""
You are validating an EDI segment against rules.

Interpretation rules:
- "Required" means the field must exist in the segment. Even if the value is "00" or empty string symbols like "00", "ZZ", it is still valid as long as it is a standard code or allowed value.
- "Situational" means the field may be present or absent depending on conditions. If no condition is given in the rule, then both present and absent are acceptable.
- "Not Used" means the field must be blank or omitted.
- Never flag standard EDI codes like "00", "01", "ZZ" as empty values.
- Only mark as invalid if the field is missing entirely, in the wrong position, or violates a clearly defined situational dependency.

EDI line:
{line.strip()}

Fields:
{fields}

Rules:
{json.dumps(matching_rules, indent=2)}

Respond only in JSON (no explanation):

{{
  "status": "Matched" or "Invalid",
  "rule_line": "matching rule or None",
  "reason": "one-line reason"
}}
"""













prompt = f"""
You are an EDI 837/ISA segment validator.

Validate this EDI line against the provided rules and fields.

EDI line:
{line.strip()}

Fields extracted (with positions):
{fields}

Rules for this segment (from implementation guide):
{json.dumps(matching_rules, indent=2)}

Validation rules to apply:
1. If a field is marked as "Required", it must have a non-empty value in the EDI line. If missing → Invalid.
2. If a field is marked as "Not Used", it must always be empty. If it has a value → Invalid.
3. If a field is marked as "Situational":
   - If the rule description specifies a condition (e.g. "Required when field X = Y"), check that condition.
   - If condition is met → value must be present.
   - If condition is not met or no condition is described → the field may be empty or filled, both are valid.

Respond strictly in JSON with the following format:
{{
  "status": "Matched" or "Invalid",
  "rule_line": "The rule line from the guide that matched or failed",
  "reason": "One-line explanation of why it is valid or invalid"
}}
"""



















ISA*00*

*00*

*ZZ*PAYERSENDER

*ZZ*PROVIDERRECV

*240601*1200*^*00501*000000001*0*T*:~

GS*HP*PAYERSENDER*PROVIDERRECV*20240601*1200*1*X*005010X221A1~

ST*835*0001~

BPR*I*1500.00*C*CHK*123456789*DA*987654321*1234567890~

TRN*1*1234567890*9876543210~

N1*PR* INSURANCE COMPANY~

N3*123 MAIN STREET~

N4*ANYTOWN*CA*90210~

N1*PE*PROVIDER NAME~

N3*456 MEDICAL AVE~

N4*OTHERTOWN*CA*90001~

CLP 12345*1*200.00*150.00*50.00*MC*1234567890~

CAS CO*45*50.00~

NM1*QC*1*DOE JOHN****MI*987654321~

DTM*405*20240601~

REF EVA123456789~

SE*14*0001~

GE*1*1~

IEA*1*000000001~

















from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize Azure OpenAI model
llm = AzureChatOpenAI(
    deployment_name="your-deployment-name",  # e.g., "gpt-4"
    model="gpt-4",
    temperature=0,
    openai_api_version="2023-05-15",
    azure_endpoint="https://YOUR-RESOURCE-NAME.openai.azure.com/",
    api_key="YOUR-AZURE-OPENAI-KEY"
)

# Prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are validating EDI segments against rules.

### Segment:
{segment}

### Rules:
{rules}

### Task:
1. Check if this segment follows the rules.
2. If valid, return JSON:
   {{
     "segment": "{segment}",
     "matched_rule": "<exact rule line>",
     "reason": "Valid because ..."
   }}
3. If invalid, return JSON:
   {{
     "segment": "{segment}",
     "matched_rule": null,
     "reason": "Invalid because <explanation>"
   }}
""")

def validate_segment(segment: str, rules: str):
    prompt = prompt_template.format(segment=segment, rules=rules)
    response = llm.predict(prompt)
    return response



















import os
import re
import json
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============== STEP 1: PDF Extraction ==================
def extract_pdf(pdf_path, poppler_path=None):
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


# =============== STEP 2: Chunking ==================
def chunk_documents(docs, chunk_size=1500, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_documents([doc]))
    return chunks


# =============== STEP 3: Build Rules with LLM ==================
def build_rules_from_chunks(chunks, llm):
    rules = []
    for i, chunk in enumerate(chunks):
        prompt = f"""
        You are given part of an EDI implementation guideline.
        Extract validation rules for segments and fields.
        Rules must include:
        - Segment name
        - Field position
        - Usage (Required / Situational / Not Used)
        - Short description

        Text:
        {chunk.page_content}

        Return JSON list of rules.
        """
        resp = llm.invoke(prompt)
        try:
            parsed = json.loads(resp.content)
            rules.extend(parsed)
        except Exception:
            print(f"Failed to parse chunk {i}")
    return rules


# =============== STEP 4: Store Rules ==================
def save_rules(rules, output_path="rules.json"):
    with open(output_path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Rules stored at {output_path}")


# =============== STEP 5: Validate Uploaded EDI ==================
def validate_edi(edi_file, rules, llm):
    results = []
    with open(edi_file, "r") as f:
        edi_lines = f.readlines()

    for line in edi_lines:
        seg_id = line.split("*")[0]
        matching_rules = [r for r in rules if r.get("segment") == seg_id]

        if not matching_rules:
            results.append({
                "edi_line": line.strip(),
                "rule_line": None,
                "status": "Invalid",
                "reason": "No matching rule for this segment"
            })
            continue

        # Let LLM validate field-level usage
        prompt = f"""
        Validate this EDI line against the rules.

        EDI line:
        {line.strip()}

        Rules:
        {json.dumps(matching_rules, indent=2)}

        Respond JSON:
        {{
          "status": "Matched/Invalid",
          "rule_line": "...",
          "reason": "one-line reason"
        }}
        """
        resp = llm.invoke(prompt)
        try:
            parsed = json.loads(resp.content)
            parsed["edi_line"] = line.strip()
            results.append(parsed)
        except Exception:
            results.append({
                "edi_line": line.strip(),
                "rule_line": None,
                "status": "Error",
                "reason": "LLM parse failure"
            })

    return results


# =============== STEP 6: Dump to Excel ==================
def dump_to_excel(results, output_path="validation_results.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Validation results saved to {output_path}")


# =============== STEP 7: HTML Highlight ==================
def generate_html(results, output_path="validation_report.html"):
    html_lines = []
    for res in results:
        if res["status"] == "Invalid":
            html_lines.append(
                f"<p style='color:red;' title='{res['reason']}'>{res['edi_line']}</p>"
            )
        else:
            html_lines.append(
                f"<p style='color:green;' title='{res['reason']}'>{res['edi_line']}</p>"
            )
    with open(output_path, "w") as f:
        f.write("<html><body>" + "\n".join(html_lines) + "</body></html>")
    print(f"HTML report saved to {output_path}")


# ================= RUN ==================
if __name__ == "__main__":
    # Azure OpenAI Config
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",  # change to your deployment
        temperature=0,
        api_version="2024-05-01-preview"
    )

    # Step 1 & 2: Extract + Chunk
    docs = extract_pdf("EDI_Guideline.pdf", poppler_path="C:/poppler/bin")
    chunks = chunk_documents(docs)

    # Step 3 & 4: Build + Save Rules
    rules = build_rules_from_chunks(chunks, llm)
    save_rules(rules)

    # Step 5: Validate
    results = validate_edi("sample.edi", rules, llm)

    # Step 6: Excel
    dump_to_excel(results)

    # Step 7: HTML
    generate_html(results)



# =============== STEP 4B: Load Rules ==================
def load_rules(input_path="rules.json"):
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            return json.load(f)
    return None


# ================= RUN ==================
if __name__ == "__main__":
    # Azure OpenAI Config
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",  # change to your deployment name
        temperature=0,
        api_version="2024-05-01-preview"
    )

    # Step 3 & 4: Build rules only if not found locally
    rules = load_rules("rules.json")
    if rules:
        print("✅ Loaded rules from local rules.json")
    else:
        print("⚡ No local rules found, extracting from PDF...")
        docs = extract_pdf("EDI_Guideline.pdf", poppler_path="C:/poppler/bin")
        chunks = chunk_documents(docs)
        rules = build_rules_from_chunks(chunks, llm)
        save_rules(rules, "rules.json")

    # Step 5: Validate EDI
    results = validate_edi("sample.edi", rules, llm)

    # Step 6: Excel
    dump_to_excel(results, "validation_results.xlsx")

    # Step 7: HTML
    generate_html(results, "validation_report.html")












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
