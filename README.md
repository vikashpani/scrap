You are a healthcare QA test case generator working on validating claims for HIV SNP (Special Need Plan) members.

You are given the following contract summary, which includes information about covered services, reimbursement methods, care models, and possibly codes and payment terms:

"""
{summary}
"""

Your task is to extract medically realistic test case scenarios from the summary and build multi-line test cases that reflect how real claims would be submitted and processed.

---

### Step 1: Generate Realistic Medical Scenarios Based on the Summary

- From the above summary, generate realistic and relevant **clinical test scenarios** that reflect the services described.
- Each scenario should:
  - Be unique and **clearly derived from the summary**
  - Represent actual healthcare events — such as psychiatric inpatient care, ALC post-discharge care, substance use rehab, lab testing, prescription medications, HIV follow-up
  - Include combinations of services (e.g., psych + meds + lab), not just a single service
- Do NOT include scenarios unrelated to the summary.
- Avoid generic wording — describe the actual situation the contract supports.

---

### Step 2: For Each Scenario, Generate Multi-line Test Case Data

Each test case scenario should be broken into **multiple service lines** like a real-world claim. Each service line must include the following fields:

- **Line Number**: Starting from 1 and incrementing within each scenario
- **Requirement**: Short description of the line’s purpose (e.g., “Psychiatric room and board”, “Lab test for HIV viral load”)
- **Test Scenario**: The full scenario this line belongs to (repeat the same for all lines in the same case)
- **Service Type**: e.g., Inpatient Psych, Substance Use, ALC, Lab, Pharmacy
- **Service Code**: CPT/HCPCS code (infer if not present in summary)
- **Revenue Code**: Rev code like 0126, 0912, 0300, 0250 (use appropriate one based on service type)
- **Diagnosis Code**: ICD-10 codes tied to the service (use HIV-related, mental health, SUD, etc.)
- **Units**: Number of days, visits, tests, or doses
- **POS**: 2-digit place of service code (e.g., 21 = Inpatient, 55 = ALC)
- **Bill Amount**: Approximate dollar value
- **Expected Output**: What should happen when claim is processed (e.g., “Paid at 105% of OMH rate”, “Paid at $200 per diem”, “Denied - Missing Diagnosis”)

---

### Format

Return the test case lines as a JSON array.

Example format:

[
  {
    "Line Number": "1",
    "Requirement": "Room and board for psychiatric inpatient stay",
    "Test Scenario": "HIV-positive member admitted for inpatient psychiatric treatment for major depression, covered under OMH psychiatric guidelines for HIV SNP.",
    "Service Type": "Inpatient Room & Board",
    "Service Code": "H0019",
    "Revenue Code": "0126",
    "Diagnosis Code": "F32.9",
    "Units": "5",
    "POS": "21",
    "Bill Amount": "10000",
    "Expected Output": "Reimbursed at 105% OMH per diem rate for 5 days"
  },
  {
    "Line Number": "2",
    "Requirement": "Psych evaluation during inpatient stay",
    "Test Scenario": "HIV-positive member admitted for inpatient psychiatric treatment for major depression, covered under OMH psychiatric guidelines for HIV SNP.",
    "Service Type": "Behavioral Health",
    "Service Code": "90791",
    "Revenue Code": "0900",
    "Diagnosis Code": "F32.9",
    "Units": "1",
    "POS": "21",
    "Bill Amount": "250",
    "Expected Output": "Allowed under BH evaluation policy"
  },
  ...
]

---

### Rules:
- Do NOT generate explanations, markdown, or commentary — return only valid JSON
- Do NOT generate more than 15–20 total test case lines per response
- Avoid repeating the same service unless clinically required
- Use realistic values for codes and payment















You are a healthcare QA test engineer responsible for generating detailed test cases for validating claims processing logic.

You are given the following high-level test scenario from a contract for HIV SNP (Special Need Plan) patients:

"""
Test Scenario: {test_scenario}
"""

Generate one detailed test case composed of **multiple service lines**, each line representing a **billable healthcare service** involved in this scenario.

For each line, include:
- Line Number: unique integer starting from 1
- Requirement: explain what service this line is testing (e.g., Room and board, psych eval, prescription, lab)
- Service Type: inpatient, pharmacy, lab, behavioral health, etc.
- Service Code: CPT/HCPCS code (infer if not mentioned)
- Revenue Code: correct rev code based on service type
- Diagnosis Code: ICD-10 code related to the HIV SNP case (realistic and relevant to the service)
- Units: number of days, visits, or tests
- POS: place of service code (2-digit)
- Bill Amount: expected charge
- Expected Output: what should happen (e.g., “Reimbursed at 105% of OMH per diem”, “Denied for missing diagnosis”, “Paid at $200/day”)

Guidelines:
- If the scenario includes **inpatient psychiatric care**, include:
    - Room & board
    - Psych eval
    - Any labs (CBC, HIV viral load)
    - ALC (if applicable)
    - Medication
- If the service type is missing, infer it based on the scenario
- Use realistic codes (revenue, CPT, diagnosis)
- Units must match clinical context (e.g., 5 days inpatient = 5 units room, 1 eval, etc.)
- Expected output should simulate how claim adjudication should behave

Return the result as a JSON array, with each line as a separate object.

DO NOT include explanation, markdown, or tables. Just return:
[
  {
    "Line Number": "1",
    "Requirement": "...",
    "Service Type": "...",
    "Service Code": "...",
    "Revenue Code": "...",
    "Diagnosis Code": "...",
    "Units": "...",
    "POS": "...",
    "Bill Amount": "...",
    "Expected Output": "..."
  },
  ...
]

















You are a test case generation engine for healthcare claims contracts.

You are provided with a summary extracted from a provider contract that covers service information, reimbursement policies, and any relevant data. Your job is to generate highly detailed, realistic test cases for HIV Special Need Plan (SNP) claims.

Use the information below to generate test case scenarios:
"""
(summary)
"""

Instructions:

1. Generate as many **non-redundant** test case scenarios as possible based on the provided summary.
2. Each test case should include detailed service lines and coding logic, not just high-level bullets.
3. If a service does not have explicitly listed codes or units, intelligently infer them based on standard industry knowledge and HIV SNP care delivery practices.
4. Each test case should contain **multiple lines** if applicable (e.g., Room & Board, Substance Use Rehab, Prescription codes).
5. Include alternate levels of care (ALC), psychiatric inpatient, and substance use services if applicable.
6. Base the logic on combinations of: service types, reimbursement models (e.g., 105% OMH per diem, $200/day), service codes, and diagnosis relevance.

Return the output ONLY in this format, as a list of JSON objects:

[
  {
    "Line Number": "1",
    "Requirement": "Inpatient psychiatric treatment for HIV-positive patient under OMH guidelines",
    "Test Scenario": "Verify that inpatient psychiatric services for HIV SNP member are reimbursed at 105% of OMH private psychiatric hospital per diem. Patient admitted for depressive disorder.",
    "Service Type": "Behavioral Health - Inpatient",
    "Service Code": "H0019",
    "Revenue Code": "0126",
    "Diagnosis Code": "F32.9",
    "Units": "8",
    "POS": "21",
    "Bill Amount": "16800",
    "Expected Output": "Expected to process at 105% of per diem for 8 days"
  },
  {
    "Line Number": "2",
    "Requirement": "Alternate Level of Care (ALC) after psych treatment",
    "Test Scenario": "Validate that the patient placed under Alternate Level of Care is reimbursed at $200 per diem after inpatient psychiatric discharge.",
    "Service Type": "ALC - Room and Board",
    "Service Code": "S9484",
    "Revenue Code": "0169",
    "Diagnosis Code": "Z59.0",
    "Units": "5",
    "POS": "55",
    "Bill Amount": "1000",
    "Expected Output": "Expected to process at $200 per diem for 5 days"
  },
  {
    "Line Number": "3",
    "Requirement": "Substance abuse rehab services for HIV SNP patient",
    "Test Scenario": "Check that HIV SNP patient receiving drug rehab therapy is reimbursed using behavioral health and rehab codes. Treatment spans 15 days.",
    "Service Type": "Behavioral Health - Substance Use",
    "Service Code": "H2036",
    "Revenue Code": "0912",
    "Diagnosis Code": "F14.10",
    "Units": "15",
    "POS": "21",
    "Bill Amount": "6000",
    "Expected Output": "Expected to process at 105% per diem rate"
  }
]

Rules:
- Do not repeat or generalize any test case.
- Do not return markdown, tables, or explanations.
- Include only relevant test lines with codes that would actually appear in a 837I/837P claim.
- Return maximum 10-15 scenarios if applicable. Less if only a few services are listed.
















import os
import pandas as pd
from langchain.schema import HumanMessage
from langchain.chat_models import ChatGroq  # or ChatOpenAI
from io import StringIO

# === Configuration ===
INPUT_EXCEL = "/mnt/data/hiv_snp_chunks.xlsx"
OUTPUT_EXCEL = "/mnt/data/hiv_snp_testcases.xlsx"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")


# === Step 1: Summarize Extracted Info ===
def summarize_contract_info(extracted_texts):
    joined = "\n---\n".join(extracted_texts)
    prompt = f"""
You are reviewing multiple extracted paragraphs from a provider contract.
These were marked as possibly related to compensation for HIV SNP services.

Please read all the following and generate a concise but comprehensive summary, combining key points, including:
- HIV SNP-related services offered
- Reimbursement models
- Any service/revenue/diagnosis codes mentioned
- Rates or per-diem values
- Limitations or units (e.g., # of days)

Keep all useful context intact.

Text:
\"\"\"
{joined}
\"\"\"
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()


# === Step 2: Generate Test Cases from Summary ===
def generate_testcases(summary):
    prompt = f"""
You are generating test cases for claims testing based on the following provider contract summary:

\"\"\"
{summary}
\"\"\"

Generate structured test cases as CSV rows with these fields:
Test Scenario,Service Type,Service Code,Revenue Code,Diagnosis Code,Units,POS,Bill Amount,Expected Output

Use realistic values. Infer codes or amounts if not explicitly present.
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()


# === Main Driver ===
def process_all_sheets(input_excel, output_excel):
    xl = pd.ExcelFile(input_excel)
    result_sheets = {}

    for sheet in xl.sheet_names:
        print(f"📄 Processing: {sheet}")
        df = xl.parse(sheet)

        # Filter valid "Extracted Info"
        extracted_blocks = df["Extracted Info"].dropna().tolist()
        extracted_blocks = [b for b in extracted_blocks if "not relevant" not in b.lower()]

        if not extracted_blocks:
            continue

        # Step 1: Summarize all chunks
        summary = summarize_contract_info(extracted_blocks)
        print(f"🧠 Summary for {sheet}:\n{summary[:500]}...\n")

        # Step 2: Generate test cases
        csv_output = generate_testcases(summary)

        try:
            test_df = pd.read_csv(StringIO(csv_output))
            result_sheets[sheet] = test_df
        except Exception as e:
            print(f"⚠️ Failed to parse test case output for {sheet}: {e}")
            continue

    # Save output to Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for sheet, df in result_sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

    print(f"\n✅ Test cases written to: {output_excel}")


# === Run ===
if __name__ == "__main__":
    process_all_sheets(INPUT_EXCEL, OUTPUT_EXCEL)














import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.chat_models import ChatGroq  # or ChatOpenAI
import pandas as pd

# === Settings ===
PDF_FOLDER = "/mnt/data/pdfs"  # your PDF folder
OUTPUT_EXCEL = "/mnt/data/hiv_snp_chunks.xlsx"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")


# === Step 1: Extract and chunk PDF ===
def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return chunks


# === Step 2: Ask LLM if chunk is about HIV SNP compensation ===
def is_relevant_chunk(text):
    prompt = f"""
You are reading a section of a provider contract.

Your job is to answer if this text contains compensation-related information 
for HIV SNP (Special Need Plan) services.

Text:
\"\"\"
{text}
\"\"\"

If yes, extract any service descriptions and rates or codes. Otherwise say "Not relevant".
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()


# === Step 3: Process a single PDF ===
def process_pdf(pdf_path):
    chunks = load_and_chunk_pdf(pdf_path)
    results = []

    for i, chunk in enumerate(chunks):
        print(f"🔍 Checking chunk {i+1}/{len(chunks)}")
        answer = is_relevant_chunk(chunk.page_content)

        if "not relevant" not in answer.lower():
            results.append({
                "Chunk Index": i,
                "Chunk Preview": chunk.page_content[:150],
                "Extracted Info": answer.strip()
            })

    return results


# === Step 4: Process all PDFs and write to Excel ===
def process_all_pdfs(pdf_folder, output_excel):
    data_per_pdf = {}
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            print(f"\n📄 Processing: {file}")
            results = process_pdf(file_path)
            if results:
                df = pd.DataFrame(results)
                sheet_name = os.path.splitext(file)[0][:31]  # Excel limit
                data_per_pdf[sheet_name] = df

    # Save to Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for sheet_name, df in data_per_pdf.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ Output saved to: {output_excel}")


# === Run It ===
if __name__ == "__main__":
    process_all_pdfs(PDF_FOLDER, OUTPUT_EXCEL)















import os
import fitz  # PyMuPDF
import pandas as pd
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage
from openpyxl import Workbook

# === Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # or set manually here
PDF_FOLDER = "/path/to/pdf_folder"  # change this
OUTPUT_EXCEL = "hiv_snp_extraction.xlsx"

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")


# === 1. Extract full text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# === 2. Extract only Compensation section ===
def extract_compensation_section(text):
    lines = text.split("\n")
    compensation_text = []
    inside = False
    for line in lines:
        if "compensation" in line.lower():
            inside = True
        elif inside and any(kw in line.lower() for kw in ["section", "exhibit", "appendix", "termination", "provider manual"]):
            break
        elif inside:
            compensation_text.append(line.strip())
    return "\n".join(compensation_text)


# === 3. Ask LLM to extract only HIV SNP services from compensation ===
def extract_hiv_snp_services(comp_text):
    prompt = f"""
You are a contract analyst.

Given this compensation section from a provider agreement:

--- START ---
{comp_text}
--- END ---

Please extract ONLY the services that are related to the HIV Special Need Plan (HIV SNP). 
Do not include anything unrelated.

Return as:
- Service name
- Rate or reimbursement detail (if present)
- Any notes or codes mentioned
"""
    try:
        response = llm([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"


# === 4. Main pipeline ===
def process_all_pdfs_and_store(pdf_folder, output_excel_path):
    all_data = {}
    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, filename)
        print(f"🔍 Processing {filename}")
        raw_text = extract_text_from_pdf(path)
        comp_text = extract_compensation_section(raw_text)
        if not comp_text:
            print(f"⚠️ No compensation section found in {filename}")
            continue
        hiv_snp_data = extract_hiv_snp_services(comp_text)
        sheet_name = os.path.splitext(filename)[0][:31]  # Excel limit
        all_data[sheet_name] = hiv_snp_data

    # Save to Excel
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        for sheet, content in all_data.items():
            df = pd.DataFrame({"HIV SNP Services Extracted": [content]})
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"✅ Saved to {output_excel_path}")


# === Run it ===
if __name__ == "__main__":
    process_all_pdfs_and_store(PDF_FOLDER, OUTPUT_EXCEL)





















Hi Team,

I’d like to schedule a quick walkthrough session to demonstrate the new automation tool we’ve developed for comparing MetroPlus documents.

This tool helps us:

Automate PDF comparisons between data and template documents

Visualize differences in HTML format

Map multiple file pairs at once and process them in batches

Manage mappings efficiently (add, remove, or clear all)













import streamlit as st
import os
import json
import re
from glob import glob
from validation import generate_html_diff

DATA_FOLDER = "data"
TEMPLATE_FOLDER = "templates"
OUTPUT_FOLDER = "out"
MAPPING_FILE = "mapping.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load or initialize mapping
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r") as f:
        file_mappings = json.load(f)
else:
    file_mappings = []

st.set_page_config(page_title="📑 PDF Compare (Batch)", layout="wide")
st.title("📄 MetroPlus PDF Comparison Tool")

# Load PDF files (case-insensitive)
def get_pdfs(folder):
    return [f for f in os.listdir(folder) if re.search(r"\.pdf$", f, re.IGNORECASE)]

data_files = get_pdfs(DATA_FOLDER)
template_files = get_pdfs(TEMPLATE_FOLDER)

st.subheader("📥 Select File Pairs to Map")

col1, col2 = st.columns(2)
with col1:
    data_file = st.selectbox("Select a PDF from 'data' folder", data_files)
with col2:
    template_file = st.selectbox("Select a PDF from 'templates' folder", template_files)

pair = {"data": data_file, "template": template_file}

col3, col4 = st.columns(2)
with col3:
    if st.button("➕ Add Mapping"):
        if pair not in file_mappings:
            file_mappings.append(pair)
            with open(MAPPING_FILE, "w") as f:
                json.dump(file_mappings, f, indent=2)
            st.success(f"✅ Mapping added: {data_file} ↔ {template_file}")
        else:
            st.warning("⚠️ This mapping already exists.")

with col4:
    if st.button("🧹 Clear Selected Mapping"):
        if pair in file_mappings:
            file_mappings.remove(pair)
            with open(MAPPING_FILE, "w") as f:
                json.dump(file_mappings, f, indent=2)
            st.success(f"🧹 Removed: {data_file} ↔ {template_file}")
        else:
            st.warning("⚠️ Mapping not found.")

if st.button("🗑️ Clear All Mappings"):
    file_mappings.clear()
    with open(MAPPING_FILE, "w") as f:
        json.dump(file_mappings, f, indent=2)
    st.success("🗑️ All mappings cleared.")

st.subheader("📋 Current Mappings")
if not file_mappings:
    st.info("No mappings added yet.")
else:
    for i, pair in enumerate(file_mappings):
        st.write(f"{i+1}. `{pair['data']}` ⟷ `{pair['template']}`")

if st.button("🚀 Generate All Comparisons"):
    for pair in file_mappings:
        file1_path = os.path.join(DATA_FOLDER, pair["data"])
        file2_path = os.path.join(TEMPLATE_FOLDER, pair["template"])
        base1 = os.path.splitext(pair["data"])[0]
        base2 = os.path.splitext(pair["template"])[0]
        common = os.path.commonprefix([base1, base2]) or f"{base1}_{base2}"
        output_path = os.path.join(OUTPUT_FOLDER, f"{common}.html")

        try:
            generate_html_diff(file1_path, file2_path, output_path)
            st.success(f"✅ Compared: `{pair['data']}` ↔ `{pair['template']}` → `{common}.html`")
        except Exception as e:
            st.error(f"❌ Error comparing {pair['data']} & {pair['template']}: {e}")











import streamlit as st
import os
import json
from glob import glob
from validation import generate_html_diff

DATA_FOLDER = "data"
TEMPLATE_FOLDER = "templates"
OUTPUT_FOLDER = "out"
MAPPING_FILE = "mapping.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load or initialize mapping
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r") as f:
        file_mappings = json.load(f)
else:
    file_mappings = []

st.set_page_config(page_title="📑 PDF Compare (Batch)", layout="wide")
st.title("📄 MetroPlus PDF Comparison Tool")

# Load file lists
data_files = [os.path.basename(f) for f in glob(os.path.join(DATA_FOLDER, "*.pdf"))]
template_files = [os.path.basename(f) for f in glob(os.path.join(TEMPLATE_FOLDER, "*.pdf"))]

st.subheader("📥 Select File Pairs to Map")

col1, col2 = st.columns(2)
with col1:
    data_file = st.selectbox("Select a PDF from 'data' folder", data_files)
with col2:
    template_file = st.selectbox("Select a PDF from 'templates' folder", template_files)

if st.button("➕ Add Mapping"):
    pair = {"data": data_file, "template": template_file}
    if pair not in file_mappings:
        file_mappings.append(pair)
        with open(MAPPING_FILE, "w") as f:
            json.dump(file_mappings, f, indent=2)
        st.success(f"Mapping added: {data_file} ↔ {template_file}")
    else:
        st.warning("This mapping already exists.")

st.subheader("📋 Current Mappings")
if not file_mappings:
    st.info("No mappings added yet.")
else:
    for i, pair in enumerate(file_mappings):
        st.write(f"{i+1}. `{pair['data']}` ⟷ `{pair['template']}`")

if st.button("🚀 Generate All Comparisons"):
    for pair in file_mappings:
        file1_path = os.path.join(DATA_FOLDER, pair["data"])
        file2_path = os.path.join(TEMPLATE_FOLDER, pair["template"])
        base1 = os.path.splitext(pair["data"])[0]
        base2 = os.path.splitext(pair["template"])[0]
        common = os.path.commonprefix([base1, base2]) or base1
        output_path = os.path.join(OUTPUT_FOLDER, f"{common}.html")

        try:
            generate_html_diff(file1_path, file2_path, output_path)
            st.success(f"✅ Compared: `{pair['data']}` ↔ `{pair['template']}` → `{common}.html`")
        except Exception as e:
            st.error(f"❌ Error comparing {pair['data']} & {pair['template']}: {e}")













# app.py
import streamlit as st
import os
from glob import glob
from validation import generate_html_diff

# Constants
DATA_FOLDER = "data"
TEMPLATE_FOLDER = "templates"
OUTPUT_FOLDER = "out"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Page title
st.set_page_config(page_title="PDF Diff Tool", layout="wide")
st.title("📄 PDF Diff Tool (MetroPlus Validation)")

# List PDFs from folders
data_files = [os.path.basename(f) for f in glob(os.path.join(DATA_FOLDER, "*.pdf"))]
template_files = [os.path.basename(f) for f in glob(os.path.join(TEMPLATE_FOLDER, "*.pdf"))]

if not data_files or not template_files:
    st.warning("❗ Make sure both 'data/' and 'templates/' folders contain PDF files.")
    st.stop()

# UI: Dropdowns
file1 = st.selectbox("📁 Choose PDF from 'data' folder", data_files)
file2 = st.selectbox("📑 Choose PDF from 'templates' folder", template_files)

# Button
if st.button("🔍 Compare PDFs"):
    path1 = os.path.join(DATA_FOLDER, file1)
    path2 = os.path.join(TEMPLATE_FOLDER, file2)

    # Get common base name for output
    base1 = os.path.splitext(file1)[0]
    base2 = os.path.splitext(file2)[0]
    common = os.path.commonprefix([base1, base2]) or base1
    output_path = os.path.join(OUTPUT_FOLDER, f"{common}.html")

    try:
        generate_html_diff(path1, path2, output_path)
        st.success(f"✅ Diff saved: {output_path}")

        # Show HTML in iframe
        with open(output_path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=900, scrolling=True)
    except Exception as e:
        st.error(f"❌ Failed to generate diff: {e}")












import pdfplumber
import fitz  # PyMuPDF
import difflib

def extract_page_text(pdf_path):
    """Extract text per page as a list of lines."""
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text().splitlines() if page.extract_text() else [] for page in pdf.pages]

def mark_differences_on_pdf(original_pdf, modified_pdf, output_pdf="annotated_diff.pdf"):
    original_lines = extract_page_text(original_pdf)
    modified_lines = extract_page_text(modified_pdf)

    doc = fitz.open(original_pdf)

    for page_num, (orig_lines, mod_lines) in enumerate(zip(original_lines, modified_lines)):
        page = doc[page_num]
        sm = difflib.SequenceMatcher(None, orig_lines, mod_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace' or tag == 'delete':
                for i in range(i1, i2):
                    if i < len(orig_lines):
                        line_text = orig_lines[i]
                        highlight_text_on_page(page, line_text, color=(1, 0, 0))  # Red for deleted/replaced
            if tag == 'insert':
                for j in range(j1, j2):
                    if j < len(mod_lines):
                        line_text = mod_lines[j]
                        highlight_text_on_page(page, line_text, color=(0, 1, 0))  # Green for inserted

    doc.save(output_pdf)
    print(f"✅ Annotated diff saved to: {output_pdf}")

def highlight_text_on_page(page, search_text, color=(1, 0, 0)):
    """Highlight all occurrences of a text on a given PDF page."""
    found = page.search_for(search_text)
    for rect in found:
        highlight = page.add_highlight_annot(rect)
        highlight.set_colors(stroke=color)  # RGB
        highlight.update()

# Example usage
mark_differences_on_pdf("a.pdf", "b.pdf")








import pdfplumber
import difflib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def extract_text_lines(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                lines.extend(content.splitlines())
    return lines

def write_diff_to_pdf(file1, file2, output_pdf="pdf_text_diff_report.pdf"):
    text1 = extract_text_lines(file1)
    text2 = extract_text_lines(file2)

    diff = difflib.unified_diff(text1, text2, fromfile=file1, tofile=file2, lineterm="")

    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    y = height - inch
    max_lines = int(height / 12) - 2
    line_count = 0

    for line in diff:
        if line.startswith('+'):
            c.setFillColorRGB(0, 0.5, 0)  # Green
        elif line.startswith('-'):
            c.setFillColorRGB(0.8, 0, 0)  # Red
        elif line.startswith('@'):
            c.setFillColorRGB(0, 0, 0.6)  # Blue
        else:
            c.setFillColorRGB(0, 0, 0)    # Black

        c.drawString(40, y, line[:120])  # Prevent overflow
        y -= 12
        line_count += 1

        if y < 40:
            c.showPage()
            y = height - inch
            line_count = 0

    c.save()
    print(f"✅ PDF diff saved to: {output_pdf}")

# Example usage
write_diff_to_pdf("a.pdf", "b.pdf")

















import pdfplumber
import difflib

def extract_text_from_pdf(path):
    """Extract full text from a PDF file, page by page."""
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text.extend(content.splitlines())
    return text

def generate_html_diff(file1_path, file2_path, output_html="pdf_diff_output.html"):
    text1 = extract_text_from_pdf(file1_path)
    text2 = extract_text_from_pdf(file2_path)

    d = difflib.HtmlDiff(tabsize=4, wrapcolumn=100)
    html_diff = d.make_file(text1, text2, fromdesc='File A', todesc='File B')

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_diff)

    print(f"✅ Diff saved to: {output_html}")

# Example usage:
generate_html_diff("a.pdf", "b.pdf")



















import pdfplumber
import difflib

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def compare_pdfs(file1, file2):
    text1 = extract_text_from_pdf(file1).splitlines()
    text2 = extract_text_from_pdf(file2).splitlines()

    diff = difflib.unified_diff(text1, text2, fromfile='File1', tofile='File2', lineterm='')
    for line in diff:
        print(line)

# Example usage
compare_pdfs("a.pdf", "b.pdf")











import fitz  # PyMuPDF
import difflib

def extract_line_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        lines = []
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if line_text:
                    rect = fitz.Rect(line["bbox"])
                    lines.append((line_text, rect))
        pages.append(lines)
    return pages

def compute_diffs(lines_a, lines_b):
    diffs = []
    max_pages = max(len(lines_a), len(lines_b))
    for i in range(max_pages):
        a_lines = [l[0] for l in lines_a[i]] if i < len(lines_a) else []
        b_lines = [l[0] for l in lines_b[i]] if i < len(lines_b) else []
        diff = list(difflib.ndiff(a_lines, b_lines))
        diffs.append(diff)
    return diffs

def draw_diff_rectangles(pdf_path_b, lines_b, diffs, output_path):
    doc = fitz.open(pdf_path_b)
    for i, diff_page in enumerate(diffs):
        if i >= len(doc):
            continue
        page = doc[i]
        b_line_map = [line[0] for line in lines_b[i]] if i < len(lines_b) else []
        b_rect_map = {line[0]: line[1] for line in lines_b[i]} if i < len(lines_b) else {}

        for line in diff_page:
            if line.startswith("+ "):  # Added in B
                text = line[2:]
                rect = b_rect_map.get(text)
                if rect:
                    page.draw_rect(rect, color=(0, 1, 0), width=1.2)  # Green
            elif line.startswith("- "):  # Missing in B
                # We skip drawing on B’s page since it's not there, or optionally show in margin
                pass
            elif line.startswith("? "):
                continue
    doc.save(output_path)
    print(f"✅ In-place visual diff PDF saved to: {output_path}")













import fitz  # PyMuPDF
import difflib

# Step 1: Extract lines from PDF pages
def extract_pdf_lines(pdf_path):
    doc = fitz.open(pdf_path)
    all_lines = []
    for page in doc:
        blocks = page.get_text("blocks")  # Each block: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort by vertical (Y), then horizontal (X)
        lines = [b[4].strip() for b in blocks if b[4].strip()]
        all_lines.append(lines)
    return all_lines

# Step 2: Compare each page's lines using difflib.ndiff
def compute_diffs(lines_a, lines_b):
    diffs_per_page = []
    max_pages = max(len(lines_a), len(lines_b))
    for i in range(max_pages):
        page_a = lines_a[i] if i < len(lines_a) else []
        page_b = lines_b[i] if i < len(lines_b) else []
        diff = list(difflib.ndiff(page_a, page_b))
        diffs_per_page.append(diff)
    return diffs_per_page

# Step 3: Write the diffs visually on the PDF
def generate_visual_diff(base_pdf_path, diffs_per_page, output_path):
    doc = fitz.open(base_pdf_path)
    while len(doc) < len(diffs_per_page):
        doc.insert_page(-1)

    for i, diff_lines in enumerate(diffs_per_page):
        page = doc[i]
        y = 50  # Start Y position to annotate diffs
        for line in diff_lines:
            if line.startswith('- '):       # Removed in new PDF
                page.insert_text((50, y), line, fontsize=10, color=(1, 0, 0))   # Red
                y += 12
            elif line.startswith('+ '):     # Added in new PDF
                page.insert_text((50, y), line, fontsize=10, color=(0, 0.6, 0))  # Green
                y += 12
            elif line.startswith('? '):     # Change hint (e.g., ^^^^^^^)
                continue  # Skip or log
            else:                          # Unchanged
                page.insert_text((50, y), line, fontsize=10, color=(0.3, 0.3, 0.3))  # Grey
                y += 12

    doc.save(output_path)
    print(f"✅ Diff report saved as: {output_path}")

# Step 4: Run the diffing
def run_pdf_diff(pdf_path_a, pdf_path_b, output_pdf_path):
    print("🔍 Extracting text from PDFs...")
    lines_a = extract_pdf_lines(pdf_path_a)
    lines_b = extract_pdf_lines(pdf_path_b)

    print("🧠 Computing differences...")
    diffs = compute_diffs(lines_a, lines_b)

    print("🖍️ Generating visual diff PDF...")
    generate_visual_diff(pdf_path_b, diffs, output_pdf_path)

# 🧪 Example usage
if __name__ == "__main__":
    pdf_a = "a.pdf"              # Old version
    pdf_b = "b.pdf"              # New version
    output_pdf = "diff_output.pdf"
    run_pdf_diff(pdf_a, pdf_b, output_pdf)















\b(FR|NFR)[\.\s:-]*(\d+(?:[\.\-]\d+)*)?[\s:-]*(.*?)(?=\bFR|\bNFR|$)






import re

def extract_fr_nfr_requirements(text):
    """
    Extracts all functional (FR) and non-functional (NFR) requirements from a full text chunk,
    using regular expressions. Handles multi-line blocks and various formats like 'FR 1.1', 'fr.', 'NFR', etc.
    """
    # This regex captures:
    # - FR or NFR (any case), possibly followed by a number (e.g., FR 1.2, NFR.4)
    # - Followed by text until the next 'FR', 'NFR', or end of chunk
    pattern = re.compile(
        r"""(?i)               # case-insensitive
        \b(FR|NFR)             # match 'FR' or 'NFR'
        [\.\s\-:]*             # optional dot, space, dash, or colon
        (\d+[\.\d]*)?          # optional number or decimal (e.g., 2.1, 1.2.3)
        (.*?)                  # non-greedy match for description
        (?=\bFR|\bNFR|$)       # until next FR/NFR or end of string
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    matches = pattern.findall(text)

    # Reformat to extract cleaned-up blocks
    requirements = []
    for m in matches:
        tag = m[0].upper()
        number = m[1]
        content = m[2].strip()
        full_text = f"{tag} {number}".strip() + ": " + content
        requirements.append(full_text.strip())

    return requirements

















import re

def extract_requirements_from_chunk(text):
    """
    Extracts functional and non-functional requirement blocks from a chunk,
    starting with lines containing 'FR', 'NFR', or similar patterns.
    """
    lines = text.splitlines()
    blocks = []
    current_block = []
    capturing = False

    for line in lines:
        clean_line = line.strip()

        # Match FR/NFR at the start or middle of line (case-insensitive)
        is_new_req = bool(re.match(r"^(FR|NFR)[\.\s\-:]*\d*.*", clean_line, re.IGNORECASE)) or \
                     bool(re.match(r"^.*\b(fr|nfr)\b.*", clean_line, re.IGNORECASE))

        # Start new block
        if is_new_req:
            if current_block:
                blocks.append(" ".join(current_block))
                current_block = []
            capturing = True
            current_block.append(clean_line)

        elif capturing:
            # Continue capturing bullets, subpoints, or non-empty informative lines
            if re.match(r"^\d+[\.\)]", clean_line) or clean_line:
                current_block.append(clean_line)
            else:
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                capturing = False

    if current_block:
        blocks.append(" ".join(current_block))

    return blocks





















import os
import json
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores import Qdrant

# === CONFIG ===
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fdd-testcases"
GROQ_API_KEY = "your-groq-api-key"
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
CHUNKS_PER_TYPE = 5
MAX_CHUNKS = 20

# === LLM ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# === Prompts ===

extract_requirements_prompt = PromptTemplate.from_template("""
You are analyzing an FDD chunk.

Chunk content:
{chunk}

Extract and return only the **individual functional requirements** in plain English.
Output JSON list like:
["Requirement 1...", "Requirement 2...", ...]
""")

generate_testcases_prompt = PromptTemplate.from_template("""
Given the following functional requirement:

{requirement}

Generate at least 2 non-redundant test cases with:
- TestCaseObjective
- ExpectedOutcome
(Leave HowToDo as a placeholder like "To be refined")

Output only JSON:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": ["To be refined"],
    "ExpectedOutcome": "..."
  }},
  ...
]
""")

refinement_prompt = PromptTemplate.from_template("""
You are refining test cases using MetroPlus user guides.

Test Case:
- Objective: {objective}
- Steps: {steps}
- Expected: {expected}

User Guide Context:
{context}

Update HowToDo with clear tool-based steps (login, screen nav, input fields, etc).

Output valid JSON:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": ["...", "..."],
    "ExpectedOutcome": "...",
    "FunctionalRequirement": "{functional_requirement}"
  }}
]
""")

# === Qdrant VectorStore ===
qdrant = Qdrant(
    client=QdrantClient(url=QDRANT_URL),
    collection_name=COLLECTION_NAME,
    embedding=...  # your embedding model here
)

# === Step 1: Extract functional requirements ===
def extract_requirements(fdd_chunk):
    try:
        prompt = extract_requirements_prompt.format(chunk=fdd_chunk)
        res = llm.invoke(prompt)
        raw = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception as e:
        print(f"❌ Requirement extraction failed: {e}")
        return []

# === Step 2: Generate test cases ===
def generate_testcases(req_text):
    try:
        prompt = generate_testcases_prompt.format(requirement=req_text)
        res = llm.invoke(prompt)
        raw = res.content.strip().replace("```json", "").replace("```", "")
        cases = json.loads(raw)
        for c in cases:
            c["FunctionalRequirement"] = req_text
        return cases
    except Exception as e:
        print(f"❌ Testcase gen failed: {e}")
        return []

# === Step 3: Refine test cases ===
def refine_testcase(tc, functional_req):
    all_refined = []

    for offset in range(0, MAX_CHUNKS, CHUNKS_PER_TYPE):
        try:
            docs = qdrant.similarity_search(
                tc["TestCaseObjective"],
                k=CHUNKS_PER_TYPE,
                offset=offset,
                filter=Filter(
                    must=[FieldCondition(
                        key="source_type",
                        match=MatchValue(value="user_guide")
                    )]
                )
            )

            if not docs:
                continue

            context = "\n\n".join(d.page_content for d in docs)
            prompt = refinement_prompt.format(
                objective=tc["TestCaseObjective"],
                steps="\n".join(tc["HowToDo"]),
                expected=tc["ExpectedOutcome"],
                context=context,
                functional_requirement=functional_req
            )
            res = llm.invoke(prompt)
            raw = res.content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(raw)
            if isinstance(parsed, dict): parsed = [parsed]
            all_refined.extend(parsed)
        except Exception as e:
            print(f"⚠️ Refinement error: {e}")
            continue

    if not all_refined:
        tc["FunctionalRequirement"] = functional_req
        return [tc]
    return all_refined

# === Step 4: Main Execution ===
all_final_testcases = []

# Step 4.1: Pull all FDD chunks (source_type = "claims")
for offset in range(0, 100, CHUNKS_PER_TYPE):  # increase limit if needed
    try:
        docs = qdrant.similarity_search(
            query="Generate test cases",
            k=CHUNKS_PER_TYPE,
            offset=offset,
            filter=Filter(
                must=[FieldCondition(
                    key="source_type",
                    match=MatchValue(value="claims")
                )]
            )
        )

        if not docs:
            break

        for doc in docs:
            fdd_chunk = doc.page_content
            requirements = extract_requirements(fdd_chunk)

            for req in requirements:
                print(f"\n🔍 Requirement: {req[:100]}...")
                cases = generate_testcases(req)

                for case in cases:
                    refined = refine_testcase(case, req)
                    all_final_testcases.extend(refined)

    except Exception as e:
        print(f"❌ Error retrieving FDD chunk: {e}")
        break

# === Step 5: Save to Excel ===
df = pd.DataFrame(all_final_testcases)
df.to_excel("FunctionalReq_TestCases_MetroPlus.xlsx", index=False)
print(f"\n✅ All done. {len(df)} test cases written to Excel.")

















refinement_prompt = PromptTemplate.from_template("""
You are refining test cases for MetroPlus by leveraging step-by-step guidance from internal User Guides.

Given:
- Test Case Objective: {objective}
- Current Steps (HowToDo): {steps}
- Expected Outcome: {expected}

And based on the following User Guide context:
{context}

Your task:
1. Understand what the test case is trying to validate.
2. Refer to the User Guide content to determine the exact steps a tester should follow — including:
    - Which tool or system to log into (e.g., Claims Portal, Provider Portal, etc.)
    - Navigation path inside the UI (e.g., "Login → Search Member → View Eligibility")
    - Data inputs, fields, screens, and actions
    - Any validation checkpoints
3. Rewrite the "HowToDo" section in 2 to 4 clear, actionable bullet points based on actual user guide content.

Output only JSON in the format:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": [
        "Step 1...",
        "Step 2...",
        ...
    ],
    "ExpectedOutcome": "..."
  }}
]

No markdown, no explanation, no headings. Output must be valid JSON only.
""")















filter=Filter(
    must=[
        FieldCondition(
            key="source_type",
            match=MatchAny(any=["claims", "rules"])
        )
    ]
)









from qdrant_client.models import Filter, FieldCondition, MatchValue

member_docs = qdrant.similarity_search(
    test_case['TestCaseObjective'],
    k=CHUNKS_PER_TYPE,
    offset=offset,
    filter=Filter(
        must=[
            FieldCondition(
                key="source_type",
                match=MatchValue(value="member_book")
            )
        ]
    )
)












def refine_with_member_book(qdrant, test_case):
    all_refined_outputs = []

    for offset in range(0, MAX_CHUNKS, CHUNKS_PER_TYPE):
        member_docs = qdrant.similarity_search(
            test_case['TestCaseObjective'],
            k=CHUNKS_PER_TYPE,
            offset=offset,
            filter={"source_type": "member_book"}
        )

        if not member_docs:
            continue

        context = "\n\n".join(doc.page_content for doc in member_docs)

        prompt = refinement_prompt.format(
            objective=test_case['TestCaseObjective'],
            steps=test_case['HowToDo'],
            expected=test_case['ExpectedOutcome'],
            context=context
        )

        try:
            res = llm.invoke(prompt)
            response_text = res.content.strip().replace("```json", "").replace("```", "")

            if response_text and (response_text.startswith("{") or response_text.startswith("[")):
                parsed = json.loads(response_text)

                if isinstance(parsed, list):
                    all_refined_outputs.extend(parsed)
                elif isinstance(parsed, dict):
                    all_refined_outputs.append(parsed)

        except Exception as e:
            print(f"⚠️ Refinement failed at offset {offset}: {e}")
            continue

    # If no refinement succeeded, return original as list
    return all_refined_outputs if all_refined_outputs else [test_case]

















def refine_with_member_book(qdrant, test_case):
    all_refined_outputs = []

    for offset in range(0, MAX_CHUNKS, CHUNKS_PER_TYPE):
        member_docs = qdrant.similarity_search(
            test_case['TestCaseObjective'],
            k=CHUNKS_PER_TYPE,
            offset=offset,
            filter={"source_type": "member_book"}
        )

        if not member_docs:
            print(f"⚠️ No member book chunks found at offset {offset}")
            continue

        context = "\n\n".join(doc.page_content for doc in member_docs)

        prompt = refinement_prompt.format(
            objective=test_case['TestCaseObjective'],
            steps=test_case['HowToDo'],
            expected=test_case['ExpectedOutcome'],
            context=context
        )

        try:
            res = llm.invoke(prompt)
            response_text = res.content.strip().replace("```json", "").replace("```", "")

            # Debug log (optional)
            print(f"\n🔎 Refinement LLM response (offset {offset}):\n{response_text[:300]}...\n")

            if response_text and (response_text.startswith("{") or response_text.startswith("[")):
                parsed = json.loads(response_text)

                # Append all valid refinements (assume list or dict)
                if isinstance(parsed, list):
                    all_refined_outputs.extend(parsed)
                elif isinstance(parsed, dict):
                    all_refined_outputs.append(parsed)
                else:
                    print(f"⚠️ Unrecognized JSON format at offset {offset}")

        except Exception as e:
            print(f"❌ Refinement failed at offset {offset}: {e}")
            continue

    # Return first valid refined output, or fallback to original
    return all_refined_outputs[0] if all_refined_outputs else test_case





















all_cases = []

for offset in range(0, MAX_CHUNKS, CHUNKS_PER_TYPE):
    print(f"🔄 Retrieving FDD+Rules chunks: Offset {offset}")
    docs = qdrant.similarity_search(
        query,
        k=CHUNKS_PER_TYPE,
        offset=offset,
        filter={"source_type": {"$in": ["claims", "rules"]}}
    )

    if not docs:
        continue

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = base_prompt.format(context=context)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().replace("```json", "").replace("```", "")

        if raw and (raw.startswith("{") or raw.startswith("[")):
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for case in parsed:
                    refined = refine_with_member_book(qdrant, case)
                    all_cases.append(refined)
            else:
                refined = refine_with_member_book(qdrant, parsed)
                all_cases.append(refined)

        else:
            print(f"⚠️ Skipping offset {offset}: No valid JSON returned")

    except Exception as e:
        print(f"❌ Error at offset {offset}: {e}")
        continue















base_cases = []

for offset in range(0, MAX_CHUNKS, CHUNKS_PER_TYPE):
    print(f"🔄 Retrieving FDD+Rules chunks: Offset {offset}")
    docs = qdrant.similarity_search(
        query,
        k=CHUNKS_PER_TYPE,
        offset=offset,
        filter={"source_type": {"$in": ["claims", "rules"]}}
    )

    if not docs:
        continue

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = base_prompt.format(context=context)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().replace("```json", "").replace("```", "")

        if raw and (raw.startswith("{") or raw.startswith("[")):
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                base_cases.extend(parsed)
            else:
                base_cases.append(parsed)
        else:
            print(f"⚠️ Skipping offset {offset}: No valid JSON returned")

    except Exception as e:
        print(f"❌ Error at offset {offset}: {e}")
        continue


















refinement_prompt = PromptTemplate.from_template("""
You are refining test cases using MetroPlus member handbooks.

Given this test case:
Objective: {objective}
Steps: {steps}
Expected: {expected}

And the following Member Handbook content:
{context}

Refine or enrich the test case if needed based on the member handbook.

Return only JSON in this format:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }}
]

Do not include any markdown, explanation, or extra text. Output ONLY valid JSON.
""")









response_text = res.content.strip().replace("```json", "").replace("```", "")

        # Defensive check: Ensure non-empty and starts with '{' or '['
        if response_text and (response_text.startswith("{") or response_text.startswith("[")):
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed[0]  # take first item if list
            return parsed  # single dict case
        else:
            print("⚠️ Empty or non-JSON response. Skipping refinement.")
            return test_case
import os
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# === CONFIG ===
GROQ_API_KEY = "your_groq_api_key_here"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fdd-testcases"
CHUNKS_PER_TYPE = 5
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# === FILES ===
PDF_FILES = {
    "claims": "data/FDD/Claims & Payment.pdf"
}
RULES_PDF = "data/FDD/Rules.pdf"
MEMBER_FILES = [
    "data/MBR-23.320_2023-MMC-Member-Handbook-Insert-Dental_2023_FINAL.pdf",
    "data/MBR-24.015_2023-Mobile-Crisis-Telephonic-Triage-and-Response-Handbook.pdf",
    "data/MBR-24.106-Final-HCBS-Handbook-Insert_FINAL-1.pdf",
    "data/MBR-24-262-Family-planning-insert_MMC_FINAL.pdf",
    "data/MBR-25.039-Doula-Services_Handbook-Insert_FINAL.pdf",
    "data/MBR-25.133-CDSMP-For-Arthritis_Handbook-Insert_FINAL.pdf"
]

# === Step 1: Load and Chunk Documents ===
print("\n📄 Loading and chunking documents...")
documents = []

# Load Rules
loader = PyPDFLoader(RULES_PDF)
for doc in loader.load():
    doc.metadata["source_type"] = "rules"
    documents.append(doc)

# Load Claims FDD
for name, path in PDF_FILES.items():
    loader = PyPDFLoader(path)
    for doc in loader.load():
        doc.metadata["source_type"] = "claims"
        documents.append(doc)

# Load Member Handbook
for path in MEMBER_FILES:
    loader = PyPDFLoader(path)
    for doc in loader.load():
        doc.metadata["source_type"] = "member_book"
        documents.append(doc)

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# === Step 2: Store in Qdrant ===
print(f"\n🧠 Loaded {len(documents)} documents, split into {len(chunks)} chunks.")
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en", model_kwargs={'device': 'cpu'})

qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME
)

# === Step 3: Setup LLM and Prompt ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

base_prompt = PromptTemplate.from_template("""
You are a test case generation engine for MetroPlus Health Plan.

Given the following Claims FDD and Rules content:

{context}

Generate high-quality, non-redundant test cases. Each test case should include:
- TestCaseObjective
- HowToDo (2–3 bullet points)
- ExpectedOutcome

Return ONLY JSON in this format:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }}
]
""")

refinement_prompt = PromptTemplate.from_template("""
You are refining test cases using MetroPlus member handbooks.

Given this test case:
Objective: {objective}
Steps: {steps}
Expected: {expected}

And the following Member Handbook content:

{context}

Refine or enrich the test case if needed. Return updated JSON format.
""")

# === Step 4: Helper Functions ===
def retrieve_context(qdrant, query):
    claims = qdrant.similarity_search(query, k=CHUNKS_PER_TYPE, filter={"source_type": "claims"})
    rules = qdrant.similarity_search(query, k=CHUNKS_PER_TYPE, filter={"source_type": "rules"})
    return claims + rules

def refine_with_member_book(qdrant, test_case):
    member_docs = qdrant.similarity_search(test_case['TestCaseObjective'], k=CHUNKS_PER_TYPE, filter={"source_type": "member_book"})
    context = "\n\n".join(doc.page_content for doc in member_docs)
    prompt = refinement_prompt.format(
        objective=test_case['TestCaseObjective'],
        steps=test_case['HowToDo'],
        expected=test_case['ExpectedOutcome'],
        context=context
    )
    try:
        res = llm.invoke(prompt)
        return json.loads(res.content.strip().replace("```json", "").replace("```", ""))[0]
    except Exception as e:
        print(f"⚠️ Skipping refinement due to error: {e}")
        return test_case

# === Step 5: Main Generation Loop ===
for fdd_key in PDF_FILES:
    print(f"\n🚀 Generating test cases for: {fdd_key}")
    all_cases = []

    query = f"Generate test cases from {fdd_key} FDD using Rules."
    docs = retrieve_context(qdrant, query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = base_prompt.format(context=context)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().replace("```json", "").replace("```", "")
        base_cases = json.loads(raw)

        print(f"\n✅ Generated {len(base_cases)} base test cases. Refining...")
        for case in base_cases:
            refined = refine_with_member_book(qdrant, case)
            all_cases.append(refined)

        df = pd.DataFrame(all_cases)
        output_file = f"TestCases_{fdd_key}_MPH.xlsx"
        df.to_excel(output_file, index=False)
        print(f"📦 Saved: {output_file}")

    except Exception as e:
        print(f"❌ Error generating test cases for {fdd_key}: {e}")
















import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Excel files ---
check_df = pd.read_excel("check.xlsx")
ground_truth_df = pd.read_excel("ground_truth.xlsx")

# --- Use appropriate column names ---
check_scenarios = check_df["TestScenario"].fillna("").astype(str).tolist()
ground_truth_scenarios = ground_truth_df["TestScenario"].fillna("").astype(str).tolist()

# --- Vectorize all scenarios using TF-IDF ---
vectorizer = TfidfVectorizer().fit(check_scenarios + ground_truth_scenarios)
check_vectors = vectorizer.transform(check_scenarios)
ground_truth_vectors = vectorizer.transform(ground_truth_scenarios)

# --- Calculate cosine similarity matrix ---
similarity_matrix = cosine_similarity(check_vectors, ground_truth_vectors)

# --- Match each check scenario to the most similar ground truth ---
matches = []
for i, sim_vector in enumerate(similarity_matrix):
    best_match_index = sim_vector.argmax()
    best_score = sim_vector[best_match_index]
    match_info = {
        "CheckTestScenario": check_scenarios[i],
        "MatchedGroundTruthScenario": ground_truth_scenarios[best_match_index],
        "SimilarityScore": best_score
    }
    matches.append(match_info)

# --- Convert result to DataFrame and save to Excel ---
result_df = pd.DataFrame(matches)
result_df.to_excel("testcase_matches.xlsx", index=False)

print("✅ Matching completed. Output saved to 'testcase_matches.xlsx'")













import pandas as pd
import json
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# === CONFIG ===
GROQ_API_KEY = "your-groq-api-key"  # 🔁 Replace this
MODEL_NAME = "mixtral-8x7b-32768"
INPUT_FILE = "seed_testcases.xlsx"
OUTPUT_FILE = "Expanded_TestCases.xlsx"
SHEETS = ["Fraud", "Abuse", "Wastage"]

# === STEP 1: SETUP LLM ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

prompt_template = PromptTemplate.from_template("""
You are a QA tester for MetroPlus Health Plan.

Given the following test scenario:
"{scenario}"

Generate as many detailed test cases as possible.
Each test case must include:
1. TestCaseObjective
2. HowToDo
3. ExpectedOutcome

Respond only in JSON format like:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }},
  ...
]
""")

# === STEP 2: PROCESS EACH SHEET ===
grouped_outputs = {}

for sheet in SHEETS:
    print(f"\n📄 Processing sheet: {sheet}")
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)

    # Clean + Filter rows
    df['Status'] = df['Status'].astype(str).str.strip().str.upper()
    df['AlreadyCovered'] = df['AlreadyCovered'].astype(str).str.strip().str.upper()
    df = df[(df['Status'] == 'Y') & (df['AlreadyCovered'] != 'Y')]

    output_rows = []

    for _, row in df.iterrows():
        scenario = row.get("ScenarioDescription", "").strip()
        if not scenario:
            continue

        prompt = prompt_template.format(scenario=scenario).to_string()
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            cases = json.loads(raw)

            if isinstance(cases, list):
                for case in cases:
                    case["ScenarioDescription"] = scenario
                    output_rows.append(case)
                print(f"✅ Scenario: '{scenario[:40]}...' → {len(cases)} test cases")
            else:
                print(f"⚠️ Skipped (not list): {scenario[:40]}")

        except Exception as e:
            print(f"❌ Error on: {scenario[:40]} → {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"\n\nSheet: {sheet}\nScenario: {scenario}\nError: {e}\nResponse:\n{raw if 'raw' in locals() else 'N/A'}")

    grouped_outputs[sheet] = output_rows

# === STEP 3: WRITE OUTPUT TO MULTI-SHEET EXCEL ===
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for sheet, data in grouped_outputs.items():
        if data:
            df_out = pd.DataFrame(data)[["ScenarioDescription", "TestCaseObjective", "HowToDo", "ExpectedOutcome"]]
            df_out.to_excel(writer, sheet_name=sheet, index=False)

print(f"\n✅ All test cases saved to: {OUTPUT_FILE}")


















import pandas as pd
import json
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# === CONFIG ===
GROQ_API_KEY = "your-groq-api-key"  # replace this
MODEL_NAME = "mixtral-8x7b-32768"
INPUT_FILE = "seed_testcases.xlsx"
OUTPUT_FILE = "Expanded_TestCases.xlsx"

# === STEP 1: LOAD EXCEL & FILTER ===
df = pd.read_excel(INPUT_FILE)
df = df[df['Status'] == 'Y']  # Only keep those to expand

# === STEP 2: PREP LLM ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

prompt_template = PromptTemplate.from_template("""
You are an expert QA tester.

Given the following test scenario:
"{scenario}"

Generate as many detailed test cases as possible.
Each test case should have:
1. TestCaseObjective (what to test)
2. HowToDo (steps or setup)
3. ExpectedOutcome (expected behavior)

Respond only in JSON format like:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }},
  ...
]
""")

# === STEP 3: GROUP AND EXPAND ===
grouped_results = {"Fraud": [], "Abuse": [], "Wastage": []}

for _, row in df.iterrows():
    scenario = row["ScenarioDescription"]
    category = row["Category"].strip().capitalize()

    prompt = prompt_template.format(scenario=scenario).to_string()
    response = llm.invoke(prompt)
    raw = response.content.strip().replace("```json", "").replace("```", "")

    try:
        cases = json.loads(raw)
        for case in cases:
            case["ScenarioDescription"] = scenario
            grouped_results[category].append(case)
        print(f"✅ Expanded: {scenario} -> {len(cases)} test cases")
    except Exception as e:
        print(f"❌ Failed on: {scenario} ({e})")
        with open("error_output.txt", "a") as f:
            f.write(f"\n\nScenario: {scenario}\nRaw:\n{raw}\n")

# === STEP 4: WRITE TO EXCEL WITH MULTIPLE SHEETS ===
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for category, data in grouped_results.items():
        if data:
            pd.DataFrame(data)[["ScenarioDescription", "TestCaseObjective", "HowToDo", "ExpectedOutcome"]].to_excel(writer, sheet_name=category, index=False)

print(f"\n📁 Test cases written to: {OUTPUT_FILE}")






















import os
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# === CONFIGURATION ===
GROQ_API_KEY = "your-groq-api-key"  # 🔁 Replace this
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fdd-testcases"
PDF_FILES = {
    "claims": "fdd_claims_mph.pdf",
    "providers": "fdd_providers_mph.pdf",
    "guidingcare": "fdd_guidingcare_mph.pdf",
}
RULES_PDF = "rules.pdf"
CHUNKS_PER_QUERY = 10
MAX_CHUNKS = 30  # loop 3 times (10 + 10 + 10)

# === STEP 1: LOAD & CHUNK DOCUMENTS ===
print("📄 Loading and chunking documents...")

loaders = [PyPDFLoader(RULES_PDF)]
for name, path in PDF_FILES.items():
    loaders.append(PyPDFLoader(path))

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

print(f"✅ Loaded {len(documents)} documents, split into {len(chunks)} chunks.")

# === STEP 2: STORE IN QDRANT ===
print("📦 Storing chunks in Qdrant...")
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    location=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

# === STEP 3: SETUP GROQ & PROMPT ===
print("🤖 Setting up Groq LLM...")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

prompt_template = PromptTemplate.from_template("""
You are a test case generation engine for MetroPlus Health Plan QA team.

Given the following FDD and rules content:
{context}

Extract as many **non-redundant** test scenarios as possible. Each test case should contain:

1. TestCaseObjective — What is being validated?
2. HowToDo — How to test it (what steps or input)?
3. ExpectedOutcome — What should happen?

Return only JSON in this format:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }},
  ...
]

No markdown, no explanation.
""")

# === STEP 4: GENERATE TEST CASES PER FDD IN BATCHES ===
for fdd_key in PDF_FILES.keys():
    print(f"\n🚀 Generating test cases for FDD: {fdd_key}")
    testcases = []

    for i in range(0, MAX_CHUNKS, CHUNKS_PER_QUERY):
        print(f"🔍 Batch {i//CHUNKS_PER_QUERY + 1} (chunks {i+1}-{i+CHUNKS_PER_QUERY})")

        docs = qdrant.similarity_search(
            f"Generate test cases from {fdd_key} FDD in MetroPlus",
            k=CHUNKS_PER_QUERY,
            search_kwargs={"offset": i}
        )
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_text = prompt_template.format(context=context).to_string()

        try:
            response = llm.invoke(prompt_text)
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            part = json.loads(raw)

            if isinstance(part, list):
                testcases.extend(part)
                print(f"✅ Added {len(part)} test cases from batch {i//CHUNKS_PER_QUERY + 1}")
            else:
                print(f"⚠️ Skipping batch {i//CHUNKS_PER_QUERY + 1}: Unexpected format")

        except Exception as e:
            print(f"❌ Error in batch {i//CHUNKS_PER_QUERY + 1}: {type(e).__name__} - {e}")
            with open(f"response_{fdd_key}_batch_{i//CHUNKS_PER_QUERY + 1}.txt", "w") as f:
                f.write(raw if 'raw' in locals() else "NO RESPONSE")
            continue

    # === STEP 5: SAVE TO EXCEL ===
    if testcases:
        df = pd.DataFrame(testcases)
        filename = f"TestCases_{fdd_key}_MPH.xlsx"
        df.to_excel(filename, index=False)
        print(f"📁 Saved {len(df)} test cases to {filename}")
    else:
        print(f"⚠️ No test cases generated for {fdd_key}")















import os
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# === CONFIGURATION ===
GROQ_API_KEY = "your-groq-api-key"  # Replace this
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fdd-testcases"
PDF_FILES = {
    "claims": "fdd_claims_mph.pdf",
    "providers": "fdd_providers_mph.pdf",
    "guidingcare": "fdd_guidingcare_mph.pdf",
}
RULES_PDF = "rules.pdf"

# === STEP 1: LOAD & CHUNK ===
print("📄 Loading documents...")
loaders = [PyPDFLoader(RULES_PDF)]
for name, path in PDF_FILES.items():
    loaders.append(PyPDFLoader(path))

documents = []
for loader in loaders:
    documents.extend(loader.load())

print(f"✅ Total pages loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"🔪 Total chunks created: {len(chunks)}")

# === STEP 2: EMBED & STORE TO QDRANT ===
print("📦 Storing embeddings to Qdrant...")
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    location=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

# === STEP 3: SETUP LLM + RAG CHAIN ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

prompt_template = PromptTemplate.from_template("""
You are a test case generation engine for MetroPlus Health Plan QA team.

Given the following FDD and rules content:
{context}

Extract as many **non-redundant** test scenarios as possible. Each test case should contain:

1. TestCaseObjective — What is being validated?
2. HowToDo — How to test it (what steps or input)?
3. ExpectedOutcome — What should happen?

Return only JSON in this format:
[
  {{
    "TestCaseObjective": "...",
    "HowToDo": "...",
    "ExpectedOutcome": "..."
  }},
  ...
]

No markdown, no explanation.
""")

retriever = qdrant.as_retriever(search_kwargs={"k": 10})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# === STEP 4: GENERATE TEST CASES FOR EACH FDD ===
for fdd_key in PDF_FILES.keys():
    print(f"\n🚀 Generating test cases for: {fdd_key}")

    query = f"Generate test cases from {fdd_key} FDD in MetroPlus"
    response = qa_chain.run(query)

    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean.replace("```json", "").replace("```", "").strip()

    try:
        testcases = json.loads(clean)
    except Exception as e:
        print(f"❌ JSON parsing failed for {fdd_key}: {e}")
        with open(f"response_{fdd_key}.txt", "w") as f:
            f.write(clean)
        continue

    df = pd.DataFrame(testcases)
    filename = f"TestCases_{fdd_key}_MPH.xlsx"
    df.to_excel(filename, index=False)
    print(f"✅ Saved {len(df)} test cases to: {filename}")


















import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import zip_longest

# Used to store nested list target tag outputs
nested_list_outputs = defaultdict(list)

def flatten_xml(elem, result, tag_order, prefix="", claim_id=None):
    for child in elem:
        tag = child.tag.split("}")[-1]
        tag = elem.tag + '_' + tag  # include parent
        key = f"{tag}".lower().replace(" ", "")
        if key not in tag_order:
            tag_order.append(key)

        # Store text
        if child.text and child.text.strip():
            result[key].append(child.text.strip())

        flatten_xml(child, result, tag_order, key)

def extract_claim_and_nested(file_path, target_tags):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all ConvertedSupplierInvoice blocks
    pattern = re.compile(r"<ConvertedSupplierInvoice\b.*?>.*?</ConvertedSupplierInvoice>", re.DOTALL)
    matches = pattern.findall(content)

    parsed_rows = []
    
    for fragment in matches:
        try:
            elem = ET.fromstring(fragment)
            result = defaultdict(list)
            tag_order = []
            claim_id = None

            # Find hccClaimNumber
            for node in elem.iter():
                if node.tag.lower().endswith("hccclaimnumber") and node.text:
                    claim_id = node.text.strip()
                    break

            # Flatten full block
            flatten_xml(elem, result, tag_order)
            parsed_rows.append((result, tag_order))

            # Now extract target nested list tags
            for target_tag in target_tags:
                for target_list in elem.iter():
                    if target_list.tag.lower().endswith(target_tag.lower()):
                        for item in target_list:
                            row = [claim_id]  # Always prepend claim_id
                            for child in item:
                                val = child.text.strip() if child.text else ""
                                row.append(val)
                            nested_list_outputs[target_tag.lower()].append(row)

        except ET.ParseError as e:
            print(f"Skipped malformed block: {e}")
            with open('failed_xml.txt', 'a', encoding='utf-8') as err:
                err.write(fragment.replace("\n", "").strip() + "\n")

    return parsed_rows

def format_cell(val):
    val = val.strip()
    if val == "":
        return ""
    elif val.isdigit() and val.startswith("0"):
        return f"'{val}"
    elif "-" in val or "/" in val:
        return f"'{val}"
    else:
        return val

# Main execution
file_path = "data/claimone.xml"
target_tags = ["DiagnosisCodeList", "ProcedureList"]  # Add more tags if needed

parsed_rows = extract_claim_and_nested(file_path, target_tags)

# Process and write flattened XML output (as before)
all_columns = []
all_data_rows = []

for result, tag_order in parsed_rows:
    new_cols = [col for col in tag_order if col not in all_columns]
    all_columns.extend(new_cols)

    max_len = max((len(v) for v in result.values()), default=1)
    broadcasted = {}

    for col in all_columns:
        values = result.get(col, [])
        if not values:
            broadcasted[col] = [""] * max_len
            continue

        first_val = next((val for val in values if val.strip()), "")
        filled = [val if val.strip() else first_val for val in values]
        if len(filled) < max_len:
            filled += [first_val] * (max_len - len(filled))
        filled = [format_cell(val) for val in filled]
        broadcasted[col] = filled

    rows = list(zip(*[broadcasted[k] for k in all_columns]))

    for row in rows:
        extended_row = list(row) + [""] * (len(all_columns) - len(row))
        all_data_rows.append(extended_row)

# Write full flattened data
with open("output_claimone.txt", "w", encoding="utf-8") as file:
    file.write("|".join(all_columns) + "\n")
    for row in all_data_rows:
        file.write("|".join(row) + "\n")

# Write each nested list tag to separate file
for tag, rows in nested_list_outputs.items():
    with open(f"{tag}.txt", "w", encoding="utf-8") as f:
        for row in rows:
            f.write("|".join(row) + "\n")













re.compile(
    fr"<hccClaimNumber\b[^>]*>.*?</hccClaimNumber>.*?<({target_tag})\b[^>]*>.*?</\1>",
    re.DOTALL
)








def merge_nested_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both nested lists must have the same number of rows.")

    merged = []
    for sub1, sub2 in zip(list1, list2):
        merged.append(sub1 + sub2)
    
    return merged

# Example input
list1 = [['a1', 'a2'], ['b1', 'b2']]
list2 = [['x1', 'x2'], ['y1', 'y2']]

# Call function
merged_output = merge_nested_lists(list1, list2)

# Print result
for row in merged_output:
    print(row)








import re
import xml.etree.ElementTree as ET

def extract_claim_id(block):
    try:
        root = ET.fromstring(block)
        for elem in root.iter():
            if elem.tag.lower().endswith("hccclaimnumber") and elem.text:
                return elem.text.strip()
    except ET.ParseError:
        pass
    return ""

def extract_target_fields(fragment, target_tag):
    try:
        elem = ET.fromstring(fragment)
        rows = []
        for sub in elem:
            row = []
            for child in sub:
                row.append(child.text.strip() if child.text else "")
            rows.append(row)
        return rows
    except ET.ParseError:
        return []

def process_target_blocks(file_path, outer_tag, target_tags):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    outer_pattern = re.compile(fr"<{outer_tag}\b.*?>.*?</{outer_tag}>", re.DOTALL | re.IGNORECASE)
    outer_blocks = outer_pattern.findall(content)

    for target_tag in target_tags:
        output_lines = []

        for block in outer_blocks:
            claim_id = extract_claim_id(block)

            inner_pattern = re.compile(fr"<{target_tag}\b.*?>.*?</{target_tag}>", re.DOTALL | re.IGNORECASE)
            target_blocks = inner_pattern.findall(block)

            for target in target_blocks:
                records = extract_target_fields(target, target_tag)
                for record in records:
                    output_lines.append("|".join([claim_id] + record))

        if output_lines:
            with open(f"{target_tag.lower()}_output.txt", "w", encoding="utf-8") as out_file:
                for line in output_lines:
                    out_file.write(line + "\n")
            print(f"✅ {target_tag.lower()}_output.txt created with {len(output_lines)} rows.")
        else:
            print(f"⚠️ No records found for tag <{target_tag}>")

# === Example Usage ===
file_path = "data/claimone.xml"
outer_tag = "ConvertedSupplierInvoice"
target_tags = ["diagnosisCode", "procedureCode", "serviceCode"]  # replace as needed

process_target_blocks(file_path, outer_tag, target_tags)








import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import zip_longest
import os

nested_list_data = defaultdict(list)

def flatten_xml(elem, result, tag_order, prefix="", claim_id=None):
    for child in elem:
        tag = child.tag.split("}")[-1]
        tag = elem.tag + '_' + tag  # include parent
        key = f"{tag}".lower().replace(" ", "")
        if key not in tag_order:
            tag_order.append(key)

        if tag.lower().endswith("hccclaimnumber") and child.text and child.text.strip():
            claim_id = child.text.strip()

        if child.text and child.text.strip() and len(child) == 0:
            result[key].append(child.text.strip())

        if len(child) > 0 and child.tag.lower().endswith("list"):
            list_tag = child.tag.split("}")[-1].lower()
            for sub_item in child:
                row = []
                for sub_child in sub_item:
                    val = sub_child.text.strip() if sub_child.text else ""
                    row.append(val)
                if claim_id:
                    row.insert(0, claim_id)
                nested_list_data[list_tag].append(row)

        flatten_xml(child, result, tag_order, key, claim_id)

def parse_multiple_xml_etree(file_path, outer_tag, target_tag):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    outer_pattern = re.compile(fr"<{outer_tag}\b.*?>.*?</{outer_tag}>", re.DOTALL | re.IGNORECASE)
    matches = outer_pattern.findall(content)

    all_rows = []

    for xml_fragment in matches:
        if not re.search(fr"<{target_tag}\b.*?>.*?</{target_tag}>", xml_fragment, re.IGNORECASE | re.DOTALL):
            continue
        try:
            elem = ET.fromstring(xml_fragment)
            result = defaultdict(list)
            tag_order = []
            flatten_xml(elem, result, tag_order)

            max_len = max((len(v) for v in result.values()), default=0)
            for k in result:
                while len(result[k]) < max_len:
                    result[k].append("")

            all_rows.append((result, tag_order.copy()))
        except ET.ParseError as e:
            with open('failed_xml.txt', 'a', encoding='utf-8') as err_file:
                err_file.write(xml_fragment.replace("\n", "").strip() + "\n")
            continue

    return all_rows

def format_cell(val):
    val = val.strip()
    if val == "":
        return ""
    elif val.isdigit() and val.startswith("0"):
        return f"'{val}"
    elif "-" in val or "/" in val:
        return f"'{val}"
    else:
        return val

def run_for_target_tags(file_path, outer_tag, target_tags):
    for tag in target_tags:
        nested_list_data.clear()
        parsed_rows = parse_multiple_xml_etree(file_path, outer_tag, tag)
        print(f"✅ Parsed {len(parsed_rows)} blocks for tag <{tag}>")

        all_columns = []
        all_data_rows = []

        for result, tag_order in parsed_rows:
            new_cols = [col for col in tag_order if col not in all_columns]
            all_columns.extend(new_cols)

            max_len = max((len(v) for v in result.values()), default=1)
            broadcasted = {}

            for col in all_columns:
                values = result.get(col, [])
                if not values:
                    broadcasted[col] = [""] * max_len
                    continue
                first_val = next((val for val in values if val.strip()), "")
                filled = [val if val.strip() else first_val for val in values]
                if len(filled) < max_len:
                    filled += [first_val] * (max_len - len(filled))
                filled = [format_cell(val) for val in filled]
                broadcasted[col] = filled

            rows = list(zip(*[broadcasted[k] for k in all_columns]))
            for row in rows:
                extended_row = list(row) + [""] * (len(all_columns) - len(row))
                all_data_rows.append(extended_row)

        output_txt = f"{tag.lower()}_output.txt"
        with open(output_txt, "w", encoding="utf-8") as file:
            file.write("|".join(all_columns) + "\n")
            for row in all_data_rows:
                file.write("|".join(row) + "\n")

        for nested_tag, rows in nested_list_data.items():
            with open(f"{tag.lower()}_{nested_tag}.txt", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write("|".join(row) + "\n")

    print("✅ All tag-specific files generated.")


# EXAMPLE USAGE
file_path = "data/claimone.xml"
outer_tag = "ConvertedSupplierInvoice"
target_tags = ["diagnosisCode", "procedureCode", "serviceCode"]  # pass your actual tags here

run_for_target_tags(file_path, outer_tag, target_tags)













import re
import xml.etree.ElementTree as ET

# Your target tag (case-insensitive)
TARGET_TAG = "diagnosisCode"

# Output file
OUTPUT_FILE = f"{TARGET_TAG.lower()}.txt"

def extract_target_blocks(file_path, target_tag):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all <ConvertedSupplierInvoice> blocks
    pattern = re.compile(r"<ConvertedSupplierInvoice\b.*?>.*?</ConvertedSupplierInvoice>", re.DOTALL)
    matches = pattern.findall(content)

    rows = []

    for xml_fragment in matches:
        try:
            root = ET.fromstring(xml_fragment)
            # Get claim ID
            claim_id = root.findtext(".//hccClaimNumber")
            if not claim_id:
                continue

            # Find all tags matching the target (case-insensitive)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]  # Remove namespace
                if tag.lower() == target_tag.lower() and elem.text and elem.text.strip():
                    rows.append(f"{claim_id}|{elem.text.strip()}")
        except ET.ParseError as e:
            print(f"Skipped malformed XML block: {e}")
            continue

    return rows

# Input file
file_path = "data/claimone.xml"

# Extract and write
extracted_rows = extract_target_blocks(file_path, TARGET_TAG)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in extracted_rows:
        f.write(row + "\n")

print(f"✅ Extracted {len(extracted_rows)} rows to {OUTPUT_FILE}")











import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import zip_longest

# Global dictionary to collect list-type nested data
nested_list_data = defaultdict(list)

def flatten_xml(elem, result, tag_order, prefix="", claim_id=None):
    for child in elem:
        tag = child.tag.split("}")[-1]
        tag = elem.tag + '_' + tag  # include parent as prefix
        key = f"{tag}".lower().replace(" ", "")

        if key not in tag_order:
            tag_order.append(key)

        # Capture claim ID if found
        if tag.lower().endswith("hccclaimnumber") and child.text and child.text.strip():
            claim_id = child.text.strip()

        # Store basic text fields
        if child.text and child.text.strip() and len(child) == 0:
            result[key].append(child.text.strip())

        # Handle list-type nodes (like diagnosisList, serviceDeliveryList, etc.)
        if len(child) > 0 and child.tag.lower().endswith("list"):
            list_tag = child.tag.split("}")[-1].lower()  # tag name without namespace
            for sub_item in child:
                row = []
                for sub_child in sub_item:
                    val = sub_child.text.strip() if sub_child.text else ""
                    row.append(val)
                if claim_id:
                    row.insert(0, claim_id)  # prepend claim ID
                nested_list_data[list_tag].append(row)

        flatten_xml(child, result, tag_order, key, claim_id)

def parse_multiple_xml_etree(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract each complete invoice block
    pattern = re.compile(r"<ConvertedSupplierInvoice\b.*?>.*?</ConvertedSupplierInvoice>", re.DOTALL)
    matches = pattern.findall(content)

    all_rows = []

    for xml_fragment in matches:
        try:
            elem = ET.fromstring(xml_fragment)
            result = defaultdict(list)
            tag_order = []
            flatten_xml(elem, result, tag_order)

            # Normalize lengths
            max_len = max((len(v) for v in result.values()), default=0)
            for k in result:
                while len(result[k]) < max_len:
                    result[k].append("")

            all_rows.append((result, tag_order.copy()))
        except ET.ParseError as e:
            print(f"Skipped malformed XML block: {e}")
            with open('failed_xml.txt', 'a', encoding='utf-8') as err_file:
                err_file.write(xml_fragment.replace("\n", "").strip() + "\n")
            continue

    return all_rows

def format_cell(val):
    val = val.strip()
    if val == "":
        return ""
    elif val.isdigit() and val.startswith("0"):
        return f"'{val}"
    elif "-" in val or "/" in val:
        return f"'{val}"
    else:
        return val

# === MAIN EXECUTION STARTS HERE ===
file_path = "data/claimone.xml"
parsed_rows = parse_multiple_xml_etree(file_path)
print("Parsed blocks:", len(parsed_rows))

all_columns = []
all_data_rows = []

for result, tag_order in parsed_rows:
    new_cols = [col for col in tag_order if col not in all_columns]
    all_columns.extend(new_cols)

    # Align each column
    max_len = max((len(v) for v in result.values()), default=1)
    broadcasted = {}

    for col in all_columns:
        values = result.get(col, [])
        if not values:
            broadcasted[col] = [""] * max_len
            continue

        first_val = next((val for val in values if val.strip()), "")
        filled = [val if val.strip() else first_val for val in values]
        if len(filled) < max_len:
            filled += [first_val] * (max_len - len(filled))

        filled = [format_cell(val) for val in filled]
        broadcasted[col] = filled

    rows = list(zip(*[broadcasted[k] for k in all_columns]))

    for row in rows:
        extended_row = list(row) + [""] * (len(all_columns) - len(row))
        all_data_rows.append(extended_row)

# === WRITE FLATTENED DATA TO OUTPUT FILE ===
with open("output_claimone.txt", "w", encoding="utf-8") as file:
    file.write("|".join(all_columns) + "\n")
    for row in all_data_rows:
        file.write("|".join(row) + "\n")

# === WRITE EACH LIST ELEMENT TO INDIVIDUAL TEXT FILES ===
for tag, rows in nested_list_data.items():
    with open(f"{tag}.txt", "w", encoding="utf-8") as f:
        for row in rows:
            f.write("|".join(row) + "\n")

print("✅ Done: Flattened and nested list data written.")

















prompt_template = PromptTemplate.from_template("""
You are generating test scenarios for MetroPlus Health Plan.

Given the following document content:
{context}

Generate as many test case scenarios as possible under:
- fraud
- abuse
- wastage

Output only raw JSON with keys 'fraud', 'abuse', and 'wastage'. No markdown. No explanation.

Example:
{{
  "fraud": [{{"description": "..."}}, ...],
  "abuse": [{{"description": "..."}}, ...],
  "wastage": [{{"description": "..."}}, ...]
}}
""")








from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Step 1: Define LLM
llm = ChatGroq(
    groq_api_key="your-groq-api-key",  # Replace with your actual key
    model_name="mixtral-8x7b-32768"
)

# Step 2: Create the prompt
prompt_template = PromptTemplate.from_template("""
You are generating test scenarios for MetroPlus Health Plan.

Analyze this text:
{context}

Return JSON:
{
  "fraud": [ { "description": "..." }, ... ],
  "abuse": [ { "description": "..." }, ... ],
  "wastage": [ { "description": "..." }, ... ]
}

No markdown or explanation.
""")

# Step 3: Use fake test context
context = "MetroPlus receives duplicate claims from the same provider. Also, tests ordered unnecessarily for healthy patients."

prompt_text = prompt_template.format(context=context).to_string()

# Step 4: Call the model
response = llm.invoke(prompt_text)

# Step 5: Show raw output
print("🧾 Raw response:\n", response.content if hasattr(response, "content") else str(response))












# Use first 3 docs for now
context = "\n\n".join(doc.page_content for doc in docs[:3])

# Format the prompt into string
prompt_text = prompt_template.format(context=context).to_string()

# Send to Groq
response = llm.invoke(prompt_text)

# Get plain response text
raw_output = response.content if hasattr(response, "content") else str(response)
print("🧾 LLM raw response (start):\n", raw_output[:500])



import json

# Clean markdown formatting if any
clean_output = raw_output.strip()
if clean_output.startswith("```json"):
    clean_output = clean_output.lstrip("```json").rstrip("```").strip()
elif clean_output.startswith("```"):
    clean_output = clean_output.lstrip("```").rstrip("```").strip()

# Try parsing
try:
    parsed = json.loads(clean_output)
    print("✅ Parsed JSON keys:", list(parsed.keys()))
except Exception as e:
    print("❌ JSON parse error:", e)
    print("🔎 Cleaned text:\n", clean_output[:500])











from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Init Groq
llm = ChatGroq(
    groq_api_key="your-groq-api-key",  # 🔁 Replace with your key or use os.getenv
    model_name="mixtral-8x7b-32768"
)

# Create prompt template
prompt_template = PromptTemplate.from_template("""
You are generating test scenarios for MetroPlus Health Plan.

Analyze this text:
{context}

Give as many unique test case scenarios as possible under:
- fraud
- abuse
- wastage

Return valid JSON with this format only (even if some are empty):

{
  "fraud": [ { "description": "..." }, ... ],
  "abuse": [ { "description": "..." }, ... ],
  "wastage": [ { "description": "..." }, ... ]
}

No markdown, no explanation.
""")















from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Qdrant setup
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fraud-docs"

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# Load vector store
vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)

# Fetch documents
docs = vectorstore.similarity_search("fraud abuse wastage in MetroPlus", k=10)

# Print sample document
print("📄 First doc sample:\n", docs[0].page_content[:300])



















Generate as many realistic, **non-redundant** test case scenarios as the content supports under the following three categories:
- Fraud
- Abuse
- Wastage

For each:
- Only return:
  - TestCase: A short title (3–6 words)
  - Description: 1–2 line explanation

❗ Only output raw JSON. Do not use markdown, explanations, or extra formatting.

JSON format:
{
  "fraud": [ { "description": "..." }, ... ],
  "abuse": [ { "description": "..." }, ... ],
  "wastage": [ { "description": "..." }, ... ]
}
""")



import os
import json
import hashlib
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ------------------ Config ------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fraud-docs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key"
MODEL_NAME = "mixtral-8x7b-32768"  # or "llama3-8b-8192"

# ------------------ Initialize Clients ------------------
client = QdrantClient(url=QDRANT_URL)
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# ------------------ Prompt Template ------------------
prompt_template = PromptTemplate.from_template("""
You are generating realistic test scenarios for MetroPlus Health Plan's internal testing.

Given the following document content:
{context}

Generate as many realistic, **non-redundant** test case scenarios as the content supports under the following three categories:
- Fraud
- Abuse
- Wastage

For each:
- Only return:
  - TestCase: A short title (3–6 words)
  - Description: 1–2 line explanation

❗ Only output raw JSON. Do not use markdown, explanations, or extra formatting.

JSON format:
{
  "fraud": [ { "description": "..." }, ... ],
  "abuse": [ { "description": "..." }, ... ],
  "wastage": [ { "description": "..." }, ... ]
}
""")

# ------------------ Helpers ------------------
def hash_description(desc):
    return hashlib.sha256(desc.strip().lower().encode()).hexdigest()

def deduplicate(scenarios):
    seen, unique = set(), []
    for s in scenarios:
        key = hash_description(s["description"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

def to_dataframe(scenario_list):
    return pd.DataFrame([
        {
            "TestCase": s["description"].split(".")[0][:60].strip(),
            "Description": s["description"].strip()
        }
        for s in scenario_list if "description" in s
    ])

# ------------------ Scenario Generation ------------------
fraud_all, abuse_all, wastage_all = [], [], []

docs = vectorstore.similarity_search("fraud abuse wastage healthcare claims metroplus", k=50)

for i in tqdm(range(0, len(docs), 5)):
    chunk = docs[i:i+5]
    context = "\n\n".join(doc.page_content for doc in chunk)
    prompt = prompt_template.format(context=context)

    try:
        response = llm.invoke(prompt)
        raw_output = response.content if hasattr(response, "content") else response
        raw_text = raw_output.strip()

        # Clean markdown wrapper if present
        if raw_text.startswith("```json"):
            raw_text = raw_text.lstrip("```json").rstrip("```").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.lstrip("```").rstrip("```").strip()

        # Attempt JSON parsing
        output = json.loads(raw_text)
        if not isinstance(output, dict):
            raise ValueError("Returned object is not a valid JSON dictionary")

        fraud_all.extend(output.get("fraud", []))
        abuse_all.extend(output.get("abuse", []))
        wastage_all.extend(output.get("wastage", []))

    except Exception as e:
        print(f"\n⚠️ Error in chunk {i}: {e}\nRaw response:\n{raw_text[:500]}\n")
        continue

# ------------------ Deduplicate ------------------
fraud_all = deduplicate(fraud_all)
abuse_all = deduplicate(abuse_all)
wastage_all = deduplicate(wastage_all)

# ------------------ Save to Excel ------------------
output_file = "MetroPlus_TestScenarios.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    to_dataframe(fraud_all).to_excel(writer, sheet_name="Fraud", index=False)
    to_dataframe(abuse_all).to_excel(writer, sheet_name="Abuse", index=False)
    to_dataframe(wastage_all).to_excel(writer, sheet_name="Wastage", index=False)

print("\n✅ Excel file saved as:", output_file)
print(f"📊 Fraud: {len(fraud_all)} | Abuse: {len(abuse_all)} | Wastage: {len(wastage_all)}")













import os
import json
import hashlib
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ------------------ Config ------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fraud-docs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key"
MODEL_NAME = "mixtral-8x7b-32768"  # Alternatives: "llama3-8b-8192"

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)

# Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# ------------------ Prompt Template ------------------
prompt_template = PromptTemplate.from_template("""
You are generating realistic test scenarios for MetroPlus Health Plan's internal system validation.

Analyze the following document content:

{context}

Based on this, generate as many unique and realistic test case scenarios as you can. Focus on fraud, abuse, and wastage scenarios relevant to:
- Claims processing
- Provider behavior
- Pharmacy or diagnostic misuse
- MetroPlus Health Plan workflows

For each test case:
- Only include:
  - TestCase: short 3–6 word title
  - Description: 1–2 line explanation

Return JSON in the format:
{
  "fraud": [ { "description": "..." }, ... ],
  "abuse": [...],
  "wastage": [...]
}
""")

# ------------------ Helpers ------------------
def hash_description(desc):
    return hashlib.sha256(desc.strip().lower().encode()).hexdigest()

def deduplicate(scenarios):
    seen, unique = set(), []
    for s in scenarios:
        key = hash_description(s["description"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

def to_dataframe(scenario_list):
    return pd.DataFrame([
        {
            "TestCase": s["description"].split(".")[0][:60].strip(),
            "Description": s["description"].strip()
        }
        for s in scenario_list
    ])

# ------------------ Scenario Generation Loop ------------------
fraud_all, abuse_all, wastage_all = [], [], []

# Retrieve documents
docs = vectorstore.similarity_search("fraud abuse wastage healthcare claims metroplus", k=50)

# Chunk docs in groups of 5
for i in tqdm(range(0, len(docs), 5)):
    chunk = docs[i:i+5]
    context = "\n\n".join(doc.page_content for doc in chunk)
    prompt = prompt_template.format(context=context)

    try:
        response = llm.invoke(prompt)
        output = json.loads(response) if isinstance(response, str) else json.loads(response.content)

        fraud_all.extend(output.get("fraud", []))
        abuse_all.extend(output.get("abuse", []))
        wastage_all.extend(output.get("wastage", []))

    except Exception as e:
        print(f"⚠️ Error in chunk {i}: {e}")
        continue

# Deduplicate
fraud_all = deduplicate(fraud_all)
abuse_all = deduplicate(abuse_all)
wastage_all = deduplicate(wastage_all)

# ------------------ Save to Excel ------------------
with pd.ExcelWriter("MetroPlus_TestScenarios.xlsx", engine="openpyxl") as writer:
    to_dataframe(fraud_all).to_excel(writer, sheet_name="Fraud", index=False)
    to_dataframe(abuse_all).to_excel(writer, sheet_name="Abuse", index=False)
    to_dataframe(wastage_all).to_excel(writer, sheet_name="Wastage", index=False)

print("✅ Excel file saved as: MetroPlus_TestScenarios.xlsx")
print(f"📊 Fraud: {len(fraud_all)} | Abuse: {len(abuse_all)} | Wastage: {len(wastage_all)}")






























import pandas as pd
import plotly.express as px
from datetime import datetime

# Convert ServDate to datetime
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract date (YYYY-MM-DD) for daily trend
merged_df['Date'] = merged_df['ServDate'].dt.date

# Group by Date and ClaimType
daily_trend = merged_df.groupby(['Date', 'ClaimType']).size().reset_index(name='ClaimCount')

# Sort the trend data by date
daily_trend = daily_trend.sort_values('Date')

# Plot the daily trend
fig = px.line(
    daily_trend,
    x='Date',
    y='ClaimCount',
    color='ClaimType',
    markers=True,
    line_shape='spline',
    title='Daily Claim Trend Analysis'
)

fig.update_traces(line=dict(width=2))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Number of Claims',
    yaxis=dict(type='log', showgrid=False),
    font_color="#303030",
    height=800,
    template='plotly_white'
)

# Show plot
fig.show()

# Save to Excel with current date in filename
today_str = datetime.today().strftime('%Y-%m-%d')
output_filename = f"Daily_Claim_Trend_{today_str}.xlsx"
daily_trend.to_excel(output_filename, index=False)

print(f"Daily claim trend data saved to: {output_filename}")
















# compare_excel.py

import pandas as pd
import pyodbc
import sys
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Compare Manual Excel to DB")
parser.add_argument("--manual_file", required=True, help="Manual Excel file path")
parser.add_argument("--sheet_name", default=0, help="Sheet name or index")
parser.add_argument("--server", required=True, help="SQL Server address, e.g., 172.28.196.4,1433")
parser.add_argument("--database", required=True, help="Database name")
args = parser.parse_args()

# --- Column Mapping (hardcoded for now) ---
column_mapping = {
    "CLAIM HCC ID": "CLAIM_HCC_ID",
    "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATED",
    "IS_ADJUSTI": "IS_ADJUSTED",
    "CLAIM_STATI": "CLAIM_STATUS",
    "F": "CLAIM_TYPE_NAME"
}

# --- Normalize helper ---
def normalize(val):
    if pd.isnull(val):
        return ""
    return str(val).strip().strip("'").lower()

# --- Connect to DB ---
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={args.server};DATABASE={args.database};Trusted_Connection=yes;'
conn = pyodbc.connect(conn_str)

query = """
SELECT CLAIM_HCC_ID, IS_FIRST_PASS_AUTO_ADJUDICATED, IS_ADJUSTED, CLAIM_STATUS, CLAIM_TYPE_NAME
FROM (
    SELECT CLAIM_HCC_ID, IS_FIRST_PASS_AUTO_ADJUDICATED, IS_ADJUSTED, CLAIM_STATUS, CLAIM_TYPE_NAME,
           RANK() OVER (PARTITION BY CLAIM_HCC_ID ORDER BY MOST_RECENT_PROCESS_TIME DESC) AS latestrecord
    FROM dbo.ALL_CLAIM_FACT
) a
WHERE latestrecord = 1
"""

df_db = pd.read_sql(query, conn)

# --- Load manual file ---
df_manual = pd.read_excel(args.manual_file, sheet_name=args.sheet_name)

# --- Compare Logic ---
mismatches = []

for idx, manual_row in df_manual.iterrows():
    claim_id_manual_col = "CLAIM HCC ID"
    claim_id_db_col = column_mapping[claim_id_manual_col]
    manual_claim_id = normalize(manual_row[claim_id_manual_col])

    candidates = df_db[
        df_db[claim_id_db_col]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.lower() == manual_claim_id
    ]

    if candidates.empty:
        mismatches.append({
            "RowIndex": idx,
            "Status": "❌ Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        db_row = candidates.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            m_val = normalize(manual_row.get(m_col))
            d_val = normalize(db_row.get(d_col))
            if m_val != d_val:
                mismatch_found = True
                break

        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "⚠️ Value Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })

# --- Save Mismatch Report ---
output_file = "Mismatch_Report.xlsx"
records = []
for item in mismatches:
    row = {
        "RowIndex": item["RowIndex"],
        "Status": item["Status"]
    }
    for k, v in (item["Manual_Row"] or {}).items():
        row[f"Manual_{k}"] = v
    for k, v in (item["Expected_DB_Row"] or {}).items():
        row[f"Expected_{k}"] = v
    records.append(row)

if records:
    result_df = pd.DataFrame(records)
    result_df.to_excel(output_file, index=False)
    print(f"✅ Mismatches written to '{output_file}' ({len(result_df)} mismatches).")
else:
    print("✅ All rows matched successfully.")





#!/bin/bash

# Usage check
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <manual_file.xlsx> <sheet_name> <server_address> <database>"
  exit 1
fi

# Read arguments
MANUAL_FILE=$1
SHEET_NAME=$2
SERVER=$3
DATABASE=$4

# Run Python script
python compare_excel.py \
  --manual_file "$MANUAL_FILE" \
  --sheet_name "$SHEET_NAME" \
  --server "$SERVER" \
  --database "$DATABASE"
















import pandas as pd
from io import BytesIO

# --- Configurable Inputs ---
manual_file = "manual.xlsx"
db_file = "db.xlsx"
output_file = "Mismatch_Report.xlsx"

# Mapping: manual column → db column
column_mapping = {
    "CLAIM HCC ID": "CLAIM_ID",
    "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATE",
    "IS_ADJUSTI": "IS_ADJUSTI",
    "CLAIM_STATI": "CLAIM_STATI",
    "F": "F",
    "G": "G",
    "Actual Error Message": "Actual Error Message"
}

# --- Helper function to normalize values ---
def normalize(val):
    if pd.isnull(val):
        return ""
    return str(val).strip().strip("'").lower()

# --- Load data ---
df_manual = pd.read_excel(manual_file)
df_db = pd.read_excel(db_file)

# --- Compare Logic ---
mismatches = []

for idx, manual_row in df_manual.iterrows():
    # Normalize and lookup by CLAIM HCC ID
    claim_id_manual_col = "CLAIM HCC ID"
    claim_id_db_col = column_mapping[claim_id_manual_col]
    manual_claim_id = normalize(manual_row[claim_id_manual_col])

    candidates = df_db[
        df_db[claim_id_db_col]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.lower() == manual_claim_id
    ]

    if candidates.empty:
        mismatches.append({
            "RowIndex": idx,
            "Status": "❌ Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        db_row = candidates.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            m_val = normalize(manual_row.get(m_col))
            d_val = normalize(db_row.get(d_col))
            if m_val != d_val:
                mismatch_found = True
                break

        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "⚠️ Value Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })

# --- Convert mismatches to dataframe ---
records = []
for item in mismatches:
    row = {
        "RowIndex": item["RowIndex"],
        "Status": item["Status"]
    }
    for k, v in (item["Manual_Row"] or {}).items():
        row[f"Manual_{k}"] = v
    for k, v in (item["Expected_DB_Row"] or {}).items():
        row[f"Expected_{k}"] = v
    records.append(row)

# --- Save to Excel ---
if records:
    result_df = pd.DataFrame(records)
    result_df.to_excel(output_file, index=False)
    print(f"✅ Mismatches written to '{output_file}' ({len(result_df)} mismatches).")
else:
    print("✅ All rows matched successfully.")










ChatGPT
Saved memory full
You said:
I have two excels with few fields I will give column mapping dictionary needs to compare those fields the data is correct or not if not store it in seperate excel file with entire row expected and actual. In two files one is comes from db it contains every thing one more is manually that needs to compare with db one iterate through each row and checks in db file
ChatGPT said:
You can accomplish this task in Python using pandas. Here’s a step-by-step guide:

✅ Problem Breakdown
You have:

A DB Excel file (db_file.xlsx) — ground truth.

A Manual Excel file (manual_file.xlsx) — to be compared.

A column mapping dictionary — mapping manual column names to DB column names.

You want:

To compare rows from the manual file against the DB file using the mapped columns.

Store mismatched rows in an output Excel file, showing both expected (from DB) and actual (from manual).

🛠️ Code Template
import pandas as pd

# Input files and config
db_file = "db_file.xlsx"
manual_file = "manual_file.xlsx"
output_file = "comparison_result.xlsx"

# Sheet names (optional if you want default sheet)
db_sheet = "Sheet1"
manual_sheet = "Sheet1"

# Mapping: manual_col -> db_col
column_mapping = {
    "Manual_Name": "DB_Name",
    "Manual_ID": "DB_ID",
    "Manual_Date": "DB_Date"
}

# Load files
df_db = pd.read_excel(db_file, sheet_name=db_sheet)
df_manual = pd.read_excel(manual_file, sheet_name=manual_sheet)

# Convert mapping into two lists
manual_cols = list(column_mapping.keys())
db_cols = [column_mapping[col] for col in manual_cols]

# Store mismatches
mismatches = []

# Iterate each row in manual file
for idx, manual_row in df_manual.iterrows():
    # Try to find matching row in DB based on unique identifier or all fields
    matches = df_db.copy()
    
    for m_col, d_col in column_mapping.items():
        matches = matches[matches[d_col] == manual_row[m_col]]

    if matches.empty:
        # No match found
        mismatches.append({
            "RowIndex": idx,
            "Status": "Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        # Found potential match, compare field by field
        db_row = matches.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            if pd.isnull(manual_row[m_col]) and pd.isnull(db_row[d_col]):
                continue
            elif manual_row[m_col] != db_row[d_col]:
                mismatch_found = True
                break
        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })

# Create a flat dataframe from mismatches
records = []
for item in mismatches:
    row = {
        "RowIndex": item["RowIndex"],
        "Status": item["Status"]
    }
    # Flatten the fields
    for k, v in (item["Manual_Row"] or {}).items():
        row[f"Manual_{k}"] = v
    for k, v in (item["Expected_DB_Row"] or {}).items():
        row[f"Expected_{k}"] = v
    records.append(row)

# Save to Excel
output_df = pd.DataFrame(records)
output_df.to_excel(output_file, index=False)
print(f"Comparison completed. Mismatches saved to: {output_file}")
📌 Notes:
This assumes you want exact match. If fuzzy matching or tolerance (e.g., for floats or dates) is required, it can be extended.

If no unique identifier is available, the matching is done row-by-row which may be inefficient for large files.

You can customize the Status column for better logging: Not Found, Mismatch, Match.

🔄 Example Column Mapping
column_mapping = {
    "Employee_ID": "EmpID",
    "Joining_Date": "StartDate",
    "Name": "FullName"
}
Would you like this as a ready-to-run script with a GUI (e.g., Streamlit), or is CLI-based okay? Also, do the manual and DB sheets have unique keys (like ID) to match on more precisely?

You said:
Give me streamlit
ChatGPT said:
Here’s a Streamlit app that:

Accepts two Excel files: one from the Database and one Manual entry.

Accepts a column mapping JSON (from Manual → DB).

Compares values row-by-row using the mapping.

Exports mismatches with full rows (manual + expected from DB) to a downloadable Excel file.

✅ Features:
File upload UI

JSON input for column mapping

Mismatch summary

Download link for mismatches

📦 Requirements
Install required packages if you haven’t:

pip install streamlit pandas openpyxl
🖥️ app.py
import streamlit as st
import pandas as pd
import json
from io import BytesIO

st.title("Excel Data Validator: Manual vs DB")

# Upload Excel files
manual_file = st.file_uploader("Upload Manual Excel File", type=["xlsx"])
db_file = st.file_uploader("Upload DB Excel File", type=["xlsx"])

# Column Mapping Input
col_map_input = st.text_area("Enter Column Mapping (Manual → DB) as JSON", height=200,
                             placeholder='e.g. {\n  "Manual_ID": "DB_ID",\n  "Manual_Name": "DB_Name"\n}')

if st.button("Compare Data"):
    if not manual_file or not db_file or not col_map_input:
        st.error("Please upload both Excel files and provide column mapping.")
    else:
        try:
            col_mapping = json.loads(col_map_input)
            manual_cols = list(col_mapping.keys())
            db_cols = [col_mapping[k] for k in manual_cols]

            df_manual = pd.read_excel(manual_file)
            df_db = pd.read_excel(db_file)

            mismatches = []

            for idx, manual_row in df_manual.iterrows():
                matches = df_db.copy()
                for m_col, d_col in col_mapping.items():
                    matches = matches[matches[d_col] == manual_row[m_col]]

                if matches.empty:
                    mismatches.append({
                        "RowIndex": idx,
                        "Status": "Not Found in DB",
                        "Manual_Row": manual_row.to_dict(),
                        "Expected_DB_Row": None
                    })
                else:
                    db_row = matches.iloc[0]
                    mismatch_found = False
                    for m_col, d_col in col_mapping.items():
                        if pd.isnull(manual_row[m_col]) and pd.isnull(db_row[d_col]):
                            continue
                        elif manual_row[m_col] != db_row[d_col]:
                            mismatch_found = True
                            break
                    if mismatch_found:
                        mismatches.append({
                            "RowIndex": idx,
                            "Status": "Mismatch",
                            "Manual_Row": manual_row.to_dict(),
                            "Expected_DB_Row": db_row.to_dict()
                        })

            if mismatches:
                records = []
                for item in mismatches:
                    row = {
                        "RowIndex": item["RowIndex"],
                        "Status": item["Status"]
                    }
                    for k, v in (item["Manual_Row"] or {}).items():
                        row[f"Manual_{k}"] = v
                    for k, v in (item["Expected_DB_Row"] or {}).items():
                        row[f"Expected_{k}"] = v
                    records.append(row)

                result_df = pd.DataFrame(records)

                st.warning(f"{len(result_df)} mismatches found.")

                @st.cache_data
                def to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='MismatchReport')
                    return output.getvalue()

                excel_bytes = to_excel(result_df)
                st.download_button(
                    label="📥 Download Mismatch Report",
                    data=excel_bytes,
                    file_name="Mismatch_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.dataframe(result_df.head(10))

            else:
                st.success("✅ All rows matched successfully!")

        except Exception as e:
            st.exception(e)
🧪 Example Column Mapping to Paste in Textbox
{
  "Manual_ID": "DB_ID",
  "Manual_Name": "DB_Name",
  "Manual_JoiningDate": "DB_JoiningDate"
}
🚀 Run the App
streamlit run app.py
Let me know if:

You want to match on a unique key only (e.g., ID first, then check others)?

Need fuzzy matching (e.g., ignore case or small date format mismatches)?

You’d like to allow CSV inputs too?

You said:
No matching found I am getting
for m_col, d_col in column_mapping.items():
        matches = matches[matches[d_col] == manual_row[m_col]]
ChatGPT said:
Thanks — the issue is likely due to strict equality checks, which fail when:

Data types differ (int vs str, datetime vs str, etc.).

Whitespace, case, or formatting differ (e.g., "123 " vs "123", or "abc" vs "ABC").

One of the values is NaN.

✅ Fix: Normalize Before Comparison
You can cast both sides to string, strip spaces, and lowercase if needed before filtering.

Here’s how to fix that specific line:

🔧 Replace This:
for m_col, d_col in column_mapping.items():
    matches = matches[matches[d_col] == manual_row[m_col]]
✅ With This (safer, normalized comparison):
for m_col, d_col in column_mapping.items():
    manual_val = str(manual_row[m_col]).strip().lower() if pd.notnull(manual_row[m_col]) else ""
    matches = matches[
        matches[d_col].astype(str).str.strip().str.lower() == manual_val
    ]
🧠 Why This Works:
Converts both manual and DB values to strings.

Strips whitespace.

Converts to lowercase for case-insensitive match.

Handles NaN gracefully.

📝 Optional: Apply to Specific Fields Only
If some fields (e.g., dates or numbers) need stricter checks, let me know — we can handle those separately.

Would you like me to patch this into the full Streamlit code from earlier?

You said:
1 CLAIM HCC ID

IS_FIRST_PASS_AUTO_ADJUDICATE

IS_ADJUSTI

CLAIM_STATI

F

G

Actual Error Message

2 '2025169001147

Y

-Claim Type-

N

3 '2025169001148

Y

N

Denied

Final

1. Please Use Appropriate Medicare Code

Professional

Professional

This is my data
ChatGPT said:
Thank you — that sample helps a lot. Let’s break this down and solve the issue you're facing with no match found, using your data structure and the goal.

🧾 Sample Manual Data You Gave (Tabular View)
CLAIM HCC ID	IS_FIRST_PASS_AUTO_ADJUDICATE	IS_ADJUSTI	CLAIM_STATI	F	G	Actual Error Message
'2025169001147	Y	-Claim Type-	N			
'2025169001148	Y	N	Denied	Final	Professional	Please Use Appropriate Medicare Code
🧩 Assumption
Manual data columns like "CLAIM HCC ID" map to DB columns (e.g., "CLAIM_ID").

The apostrophe (') in IDs is common when exported from Excel — must be stripped.

Matching should ignore case, whitespace, and treat strings consistently.

✅ Fix: Robust Normalized Comparison
Update your Streamlit logic to normalize values before comparison:

🔄 Replace Comparison Block (Full Update for Streamlit)
for m_col, d_col in col_mapping.items():
    manual_val = manual_row.get(m_col)

    if pd.isnull(manual_val):
        manual_val = ""
    else:
        manual_val = str(manual_val).strip().strip("'").lower()  # strip apostrophe too

    # Normalize DB column as string
    matches = matches[
        matches[d_col]
        .astype(str)
        .str.strip()
        .str.strip("'")  # strip leading apostrophes
        .str.lower()
        == manual_val
    ]
🧠 What This Handles:
'2025169001147 → 2025169001147 (apostrophe removed)

" Y " → "y" (strip + lowercase)

"Claim Type" vs "Professional" case handled

Avoids .loc[] or type mismatch errors

✅ Also Suggested:
If the CLAIM HCC ID is unique, match only on that, and then compare other columns one by one.

# Match by CLAIM ID only
manual_claim_id = str(manual_row["CLAIM HCC ID"]).strip().strip("'").lower()
matches = df_db[
    df_db["CLAIM_ID"].astype(str).str.strip().str.strip("'").str.lower() == manual_claim_id
]
Then compare other fields (IS_ADJUSTI, etc.) after match.

🧪 Want me to give you the updated full Streamlit code with this fixed and sample data columns added?
Let me know, and I’ll plug in the field names directly.

You said:
Give me full code properly
ChatGPT said:
Here is the complete Streamlit app that compares a manual Excel sheet against a DB Excel file, using a column mapping. It applies robust matching logic (handling apostrophes, whitespace, case mismatches) and exports mismatches into a downloadable Excel file.

✅ Streamlit App: compare_excel_app.py
import streamlit as st
import pandas as pd
import json
from io import BytesIO

st.title("📊 Excel Comparator: Manual vs DB")

# Upload Excel files
manual_file = st.file_uploader("Upload Manual Excel File", type=["xlsx"])
db_file = st.file_uploader("Upload DB Excel File", type=["xlsx"])

# Sample default mapping
default_mapping = """{
  "CLAIM HCC ID": "CLAIM_ID",
  "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATE",
  "IS_ADJUSTI": "IS_ADJUSTI",
  "CLAIM_STATI": "CLAIM_STATI",
  "F": "F",
  "G": "G",
  "Actual Error Message": "Actual Error Message"
}"""

# Column Mapping Input
col_map_input = st.text_area("Enter Column Mapping (Manual → DB) as JSON", value=default_mapping, height=250)

if st.button("🔍 Compare"):
    if not manual_file or not db_file or not col_map_input:
        st.error("Please upload both Excel files and provide column mapping.")
    else:
        try:
            # Load column mapping
            col_mapping = json.loads(col_map_input)
            manual_cols = list(col_mapping.keys())
            db_cols = list(col_mapping.values())

            # Load data
            df_manual = pd.read_excel(manual_file)
            df_db = pd.read_excel(db_file)

            # Ensure string format and strip unwanted characters
            def normalize(val):
                if pd.isnull(val):
                    return ""
                return str(val).strip().strip("'").lower()

            mismatches = []

            for idx, manual_row in df_manual.iterrows():
                # Match using CLAIM HCC ID (assuming it's unique)
                claim_id_col_manual = "CLAIM HCC ID"
                claim_id_col_db = col_mapping[claim_id_col_manual]

                manual_claim_id = normalize(manual_row[claim_id_col_manual])
                candidate_matches = df_db[
                    df_db[claim_id_col_db]
                    .astype(str)
                    .str.strip()
                    .str.strip("'")
                    .str.lower()
                    == manual_claim_id
                ]

                if candidate_matches.empty:
                    mismatches.append({
                        "RowIndex": idx,
                        "Status": "❌ Not Found in DB",
                        "Manual_Row": manual_row.to_dict(),
                        "Expected_DB_Row": None
                    })
                else:
                    db_row = candidate_matches.iloc[0]
                    mismatch_found = False
                    for m_col, d_col in col_mapping.items():
                        m_val = normalize(manual_row.get(m_col))
                        d_val = normalize(db_row.get(d_col))
                        if m_val != d_val:
                            mismatch_found = True
                            break

                    if mismatch_found:
                        mismatches.append({
                            "RowIndex": idx,
                            "Status": "⚠️ Value Mismatch",
                            "Manual_Row": manual_row.to_dict(),
                            "Expected_DB_Row": db_row.to_dict()
                        })

            if mismatches:
                records = []
                for item in mismatches:
                    row = {
                        "RowIndex": item["RowIndex"],
                        "Status": item["Status"]
                    }
                    for k, v in (item["Manual_Row"] or {}).items():
                        row[f"Manual_{k}"] = v
                    for k, v in (item["Expected_DB_Row"] or {}).items():
                        row[f"Expected_{k}"] = v
                    records.append(row)

                result_df = pd.DataFrame(records)

                st.warning(f"⚠️ {len(result_df)} mismatch(es) found.")

                # Export to Excel
                def to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name="Mismatches")
                    return output.getvalue()

                excel_bytes = to_excel(result_df)
                st.download_button(
                    label="📥 Download Mismatch Report",
                    data=excel_bytes,
                    file_name="Mismatch_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.dataframe(result_df.head(10), use_container_width=True)
            else:
                st.success("✅ All rows matched successfully!")

        except Exception as e:
            st.error("An error occurred during comparison.")
            st.exception(e)
🚀 To Run This:
Save the code as compare_excel_app.py

Install requirements if needed:

pip install streamlit pandas openpyxl
Run:

streamlit run compare_excel_app.py
🧪 Example Excel Inputs
manual.xlsx:

Columns like "CLAIM HCC ID", "IS_ADJUSTI", etc.

db.xlsx:

Columns like "CLAIM_ID", "IS_ADJUSTI", etc.

Just make sure the column names in the mapping match exactly (case-sensitive, unless you modify it).

Would you like to extend this to support:

CSV files too?

Multi-match handling (same ID appears multiple times)?

Custom match column (instead of hardcoded CLAIM HCC ID)?

You said:
No streamlit just give direct code
ChatGPT said:
Great — here’s the pure Python script (no Streamlit) that:

Reads two Excel files: one manual, one from the DB.

Uses a column mapping (manual_col -> db_col).

Compares each row in the manual file to the DB file (based on CLAIM HCC ID).

Writes mismatches into an output Excel file with full rows (actual vs expected).

✅ Python Script: compare_excels.py
import pandas as pd
from io import BytesIO

# --- Configurable Inputs ---
manual_file = "manual.xlsx"
db_file = "db.xlsx"
output_file = "Mismatch_Report.xlsx"

# Mapping: manual column → db column
column_mapping = {
    "CLAIM HCC ID": "CLAIM_ID",
    "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATE",
    "IS_ADJUSTI": "IS_ADJUSTI",
    "CLAIM_STATI": "CLAIM_STATI",
    "F": "F",
    "G": "G",
    "Actual Error Message": "Actual Error Message"
}

# --- Helper function to normalize values ---
def normalize(val):
    if pd.isnull(val):
        return ""
    return str(val).strip().strip("'").lower()

# --- Load data ---
df_manual = pd.read_excel(manual_file)
df_db = pd.read_excel(db_file)

# --- Compare Logic ---
mismatches = []

for idx, manual_row in df_manual.iterrows():
    # Normalize and lookup by CLAIM HCC ID
    claim_id_manual_col = "CLAIM HCC ID"
    claim_id_db_col = column_mapping[claim_id_manual_col]
    manual_claim_id = normalize(manual_row[claim_id_manual_col])

    candidates = df_db[
        df_db[claim_id_db_col]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.lower() == manual_claim_id
    ]

    if candidates.empty:
        mismatches.append({
            "RowIndex": idx,
            "Status": "❌ Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        db_row = candidates.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            m_val = normalize(manual_row.get(m_col))
            d_val = normalize(db_row.get(d_col))
            if m_val != d_val:
                mismatch_found = True
                break

        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "⚠️ Value Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })












ChatGPT
Saved memory full
You said:
I have two excels with few fields I will give column mapping dictionary needs to compare those fields the data is correct or not if not store it in seperate excel file with entire row expected and actual. In two files one is comes from db it contains every thing one more is manually that needs to compare with db one iterate through each row and checks in db file
ChatGPT said:
You can accomplish this task in Python using pandas. Here’s a step-by-step guide:

✅ Problem Breakdown
You have:

A DB Excel file (db_file.xlsx) — ground truth.

A Manual Excel file (manual_file.xlsx) — to be compared.

A column mapping dictionary — mapping manual column names to DB column names.

You want:

To compare rows from the manual file against the DB file using the mapped columns.

Store mismatched rows in an output Excel file, showing both expected (from DB) and actual (from manual).

🛠️ Code Template
import pandas as pd

# Input files and config
db_file = "db_file.xlsx"
manual_file = "manual_file.xlsx"
output_file = "comparison_result.xlsx"

# Sheet names (optional if you want default sheet)
db_sheet = "Sheet1"
manual_sheet = "Sheet1"

# Mapping: manual_col -> db_col
column_mapping = {
    "Manual_Name": "DB_Name",
    "Manual_ID": "DB_ID",
    "Manual_Date": "DB_Date"
}

# Load files
df_db = pd.read_excel(db_file, sheet_name=db_sheet)
df_manual = pd.read_excel(manual_file, sheet_name=manual_sheet)

# Convert mapping into two lists
manual_cols = list(column_mapping.keys())
db_cols = [column_mapping[col] for col in manual_cols]

# Store mismatches
mismatches = []

# Iterate each row in manual file
for idx, manual_row in df_manual.iterrows():
    # Try to find matching row in DB based on unique identifier or all fields
    matches = df_db.copy()
    
    for m_col, d_col in column_mapping.items():
        matches = matches[matches[d_col] == manual_row[m_col]]

    if matches.empty:
        # No match found
        mismatches.append({
            "RowIndex": idx,
            "Status": "Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        # Found potential match, compare field by field
        db_row = matches.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            if pd.isnull(manual_row[m_col]) and pd.isnull(db_row[d_col]):
                continue
            elif manual_row[m_col] != db_row[d_col]:
                mismatch_found = True
                break
        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })

# Create a flat dataframe from mismatches
records = []
for item in mismatches:
    row = {
        "RowIndex": item["RowIndex"],
        "Status": item["Status"]
    }
    # Flatten the fields
    for k, v in (item["Manual_Row"] or {}).items():
        row[f"Manual_{k}"] = v
    for k, v in (item["Expected_DB_Row"] or {}).items():
        row[f"Expected_{k}"] = v
    records.append(row)

# Save to Excel
output_df = pd.DataFrame(records)
output_df.to_excel(output_file, index=False)
print(f"Comparison completed. Mismatches saved to: {output_file}")
📌 Notes:
This assumes you want exact match. If fuzzy matching or tolerance (e.g., for floats or dates) is required, it can be extended.

If no unique identifier is available, the matching is done row-by-row which may be inefficient for large files.

You can customize the Status column for better logging: Not Found, Mismatch, Match.

🔄 Example Column Mapping
column_mapping = {
    "Employee_ID": "EmpID",
    "Joining_Date": "StartDate",
    "Name": "FullName"
}
Would you like this as a ready-to-run script with a GUI (e.g., Streamlit), or is CLI-based okay? Also, do the manual and DB sheets have unique keys (like ID) to match on more precisely?

You said:
Give me streamlit
ChatGPT said:
Here’s a Streamlit app that:

Accepts two Excel files: one from the Database and one Manual entry.

Accepts a column mapping JSON (from Manual → DB).

Compares values row-by-row using the mapping.

Exports mismatches with full rows (manual + expected from DB) to a downloadable Excel file.

✅ Features:
File upload UI

JSON input for column mapping

Mismatch summary

Download link for mismatches

📦 Requirements
Install required packages if you haven’t:

pip install streamlit pandas openpyxl
🖥️ app.py
import streamlit as st
import pandas as pd
import json
from io import BytesIO

st.title("Excel Data Validator: Manual vs DB")

# Upload Excel files
manual_file = st.file_uploader("Upload Manual Excel File", type=["xlsx"])
db_file = st.file_uploader("Upload DB Excel File", type=["xlsx"])

# Column Mapping Input
col_map_input = st.text_area("Enter Column Mapping (Manual → DB) as JSON", height=200,
                             placeholder='e.g. {\n  "Manual_ID": "DB_ID",\n  "Manual_Name": "DB_Name"\n}')

if st.button("Compare Data"):
    if not manual_file or not db_file or not col_map_input:
        st.error("Please upload both Excel files and provide column mapping.")
    else:
        try:
            col_mapping = json.loads(col_map_input)
            manual_cols = list(col_mapping.keys())
            db_cols = [col_mapping[k] for k in manual_cols]

            df_manual = pd.read_excel(manual_file)
            df_db = pd.read_excel(db_file)

            mismatches = []

            for idx, manual_row in df_manual.iterrows():
                matches = df_db.copy()
                for m_col, d_col in col_mapping.items():
                    matches = matches[matches[d_col] == manual_row[m_col]]

                if matches.empty:
                    mismatches.append({
                        "RowIndex": idx,
                        "Status": "Not Found in DB",
                        "Manual_Row": manual_row.to_dict(),
                        "Expected_DB_Row": None
                    })
                else:
                    db_row = matches.iloc[0]
                    mismatch_found = False
                    for m_col, d_col in col_mapping.items():
                        if pd.isnull(manual_row[m_col]) and pd.isnull(db_row[d_col]):
                            continue
                        elif manual_row[m_col] != db_row[d_col]:
                            mismatch_found = True
                            break
                    if mismatch_found:
                        mismatches.append({
                            "RowIndex": idx,
                            "Status": "Mismatch",
                            "Manual_Row": manual_row.to_dict(),
                            "Expected_DB_Row": db_row.to_dict()
                        })

            if mismatches:
                records = []
                for item in mismatches:
                    row = {
                        "RowIndex": item["RowIndex"],
                        "Status": item["Status"]
                    }
                    for k, v in (item["Manual_Row"] or {}).items():
                        row[f"Manual_{k}"] = v
                    for k, v in (item["Expected_DB_Row"] or {}).items():
                        row[f"Expected_{k}"] = v
                    records.append(row)

                result_df = pd.DataFrame(records)

                st.warning(f"{len(result_df)} mismatches found.")

                @st.cache_data
                def to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='MismatchReport')
                    return output.getvalue()

                excel_bytes = to_excel(result_df)
                st.download_button(
                    label="📥 Download Mismatch Report",
                    data=excel_bytes,
                    file_name="Mismatch_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.dataframe(result_df.head(10))

            else:
                st.success("✅ All rows matched successfully!")

        except Exception as e:
            st.exception(e)
🧪 Example Column Mapping to Paste in Textbox
{
  "Manual_ID": "DB_ID",
  "Manual_Name": "DB_Name",
  "Manual_JoiningDate": "DB_JoiningDate"
}
🚀 Run the App
streamlit run app.py
Let me know if:

You want to match on a unique key only (e.g., ID first, then check others)?

Need fuzzy matching (e.g., ignore case or small date format mismatches)?

You’d like to allow CSV inputs too?

You said:
No matching found I am getting
for m_col, d_col in column_mapping.items():
        matches = matches[matches[d_col] == manual_row[m_col]]
ChatGPT said:
Thanks — the issue is likely due to strict equality checks, which fail when:

Data types differ (int vs str, datetime vs str, etc.).

Whitespace, case, or formatting differ (e.g., "123 " vs "123", or "abc" vs "ABC").

One of the values is NaN.

✅ Fix: Normalize Before Comparison
You can cast both sides to string, strip spaces, and lowercase if needed before filtering.

Here’s how to fix that specific line:

🔧 Replace This:
for m_col, d_col in column_mapping.items():
    matches = matches[matches[d_col] == manual_row[m_col]]
✅ With This (safer, normalized comparison):
for m_col, d_col in column_mapping.items():
    manual_val = str(manual_row[m_col]).strip().lower() if pd.notnull(manual_row[m_col]) else ""
    matches = matches[
        matches[d_col].astype(str).str.strip().str.lower() == manual_val
    ]
🧠 Why This Works:
Converts both manual and DB values to strings.

Strips whitespace.

Converts to lowercase for case-insensitive match.

Handles NaN gracefully.

📝 Optional: Apply to Specific Fields Only
If some fields (e.g., dates or numbers) need stricter checks, let me know — we can handle those separately.

Would you like me to patch this into the full Streamlit code from earlier?

You said:
1 CLAIM HCC ID

IS_FIRST_PASS_AUTO_ADJUDICATE

IS_ADJUSTI

CLAIM_STATI

F

G

Actual Error Message

2 '2025169001147

Y

-Claim Type-

N

3 '2025169001148

Y

N

Denied

Final

1. Please Use Appropriate Medicare Code

Professional

Professional

This is my data
ChatGPT said:
Thank you — that sample helps a lot. Let’s break this down and solve the issue you're facing with no match found, using your data structure and the goal.

🧾 Sample Manual Data You Gave (Tabular View)
CLAIM HCC ID	IS_FIRST_PASS_AUTO_ADJUDICATE	IS_ADJUSTI	CLAIM_STATI	F	G	Actual Error Message
'2025169001147	Y	-Claim Type-	N			
'2025169001148	Y	N	Denied	Final	Professional	Please Use Appropriate Medicare Code
🧩 Assumption
Manual data columns like "CLAIM HCC ID" map to DB columns (e.g., "CLAIM_ID").

The apostrophe (') in IDs is common when exported from Excel — must be stripped.

Matching should ignore case, whitespace, and treat strings consistently.

✅ Fix: Robust Normalized Comparison
Update your Streamlit logic to normalize values before comparison:

🔄 Replace Comparison Block (Full Update for Streamlit)
for m_col, d_col in col_mapping.items():
    manual_val = manual_row.get(m_col)

    if pd.isnull(manual_val):
        manual_val = ""
    else:
        manual_val = str(manual_val).strip().strip("'").lower()  # strip apostrophe too

    # Normalize DB column as string
    matches = matches[
        matches[d_col]
        .astype(str)
        .str.strip()
        .str.strip("'")  # strip leading apostrophes
        .str.lower()
        == manual_val
    ]
🧠 What This Handles:
'2025169001147 → 2025169001147 (apostrophe removed)

" Y " → "y" (strip + lowercase)

"Claim Type" vs "Professional" case handled

Avoids .loc[] or type mismatch errors

✅ Also Suggested:
If the CLAIM HCC ID is unique, match only on that, and then compare other columns one by one.

# Match by CLAIM ID only
manual_claim_id = str(manual_row["CLAIM HCC ID"]).strip().strip("'").lower()
matches = df_db[
    df_db["CLAIM_ID"].astype(str).str.strip().str.strip("'").str.lower() == manual_claim_id
]
Then compare other fields (IS_ADJUSTI, etc.) after match.

🧪 Want me to give you the updated full Streamlit code with this fixed and sample data columns added?
Let me know, and I’ll plug in the field names directly.

You said:
Give me full code properly
ChatGPT said:
Here is the complete Streamlit app that compares a manual Excel sheet against a DB Excel file, using a column mapping. It applies robust matching logic (handling apostrophes, whitespace, case mismatches) and exports mismatches into a downloadable Excel file.

✅ Streamlit App: compare_excel_app.py
import streamlit as st
import pandas as pd
import json
from io import BytesIO

st.title("📊 Excel Comparator: Manual vs DB")

# Upload Excel files
manual_file = st.file_uploader("Upload Manual Excel File", type=["xlsx"])
db_file = st.file_uploader("Upload DB Excel File", type=["xlsx"])

# Sample default mapping
default_mapping = """{
  "CLAIM HCC ID": "CLAIM_ID",
  "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATE",
  "IS_ADJUSTI": "IS_ADJUSTI",
  "CLAIM_STATI": "CLAIM_STATI",
  "F": "F",
  "G": "G",
  "Actual Error Message": "Actual Error Message"
}"""

# Column Mapping Input
col_map_input = st.text_area("Enter Column Mapping (Manual → DB) as JSON", value=default_mapping, height=250)

if st.button("🔍 Compare"):
    if not manual_file or not db_file or not col_map_input:
        st.error("Please upload both Excel files and provide column mapping.")
    else:
        try:
            # Load column mapping
            col_mapping = json.loads(col_map_input)
            manual_cols = list(col_mapping.keys())
            db_cols = list(col_mapping.values())

            # Load data
            df_manual = pd.read_excel(manual_file)
            df_db = pd.read_excel(db_file)

            # Ensure string format and strip unwanted characters
            def normalize(val):
                if pd.isnull(val):
                    return ""
                return str(val).strip().strip("'").lower()

            mismatches = []

            for idx, manual_row in df_manual.iterrows():
                # Match using CLAIM HCC ID (assuming it's unique)
                claim_id_col_manual = "CLAIM HCC ID"
                claim_id_col_db = col_mapping[claim_id_col_manual]

                manual_claim_id = normalize(manual_row[claim_id_col_manual])
                candidate_matches = df_db[
                    df_db[claim_id_col_db]
                    .astype(str)
                    .str.strip()
                    .str.strip("'")
                    .str.lower()
                    == manual_claim_id
                ]

                if candidate_matches.empty:
                    mismatches.append({
                        "RowIndex": idx,
                        "Status": "❌ Not Found in DB",
                        "Manual_Row": manual_row.to_dict(),
                        "Expected_DB_Row": None
                    })
                else:
                    db_row = candidate_matches.iloc[0]
                    mismatch_found = False
                    for m_col, d_col in col_mapping.items():
                        m_val = normalize(manual_row.get(m_col))
                        d_val = normalize(db_row.get(d_col))
                        if m_val != d_val:
                            mismatch_found = True
                            break

                    if mismatch_found:
                        mismatches.append({
                            "RowIndex": idx,
                            "Status": "⚠️ Value Mismatch",
                            "Manual_Row": manual_row.to_dict(),
                            "Expected_DB_Row": db_row.to_dict()
                        })

            if mismatches:
                records = []
                for item in mismatches:
                    row = {
                        "RowIndex": item["RowIndex"],
                        "Status": item["Status"]
                    }
                    for k, v in (item["Manual_Row"] or {}).items():
                        row[f"Manual_{k}"] = v
                    for k, v in (item["Expected_DB_Row"] or {}).items():
                        row[f"Expected_{k}"] = v
                    records.append(row)

                result_df = pd.DataFrame(records)

                st.warning(f"⚠️ {len(result_df)} mismatch(es) found.")

                # Export to Excel
                def to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name="Mismatches")
                    return output.getvalue()

                excel_bytes = to_excel(result_df)
                st.download_button(
                    label="📥 Download Mismatch Report",
                    data=excel_bytes,
                    file_name="Mismatch_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.dataframe(result_df.head(10), use_container_width=True)
            else:
                st.success("✅ All rows matched successfully!")

        except Exception as e:
            st.error("An error occurred during comparison.")
            st.exception(e)
🚀 To Run This:
Save the code as compare_excel_app.py

Install requirements if needed:

pip install streamlit pandas openpyxl
Run:

streamlit run compare_excel_app.py
🧪 Example Excel Inputs
manual.xlsx:

Columns like "CLAIM HCC ID", "IS_ADJUSTI", etc.

db.xlsx:

Columns like "CLAIM_ID", "IS_ADJUSTI", etc.

Just make sure the column names in the mapping match exactly (case-sensitive, unless you modify it).

Would you like to extend this to support:

CSV files too?

Multi-match handling (same ID appears multiple times)?

Custom match column (instead of hardcoded CLAIM HCC ID)?

You said:
No streamlit just give direct code
ChatGPT said:
Great — here’s the pure Python script (no Streamlit) that:

Reads two Excel files: one manual, one from the DB.

Uses a column mapping (manual_col -> db_col).

Compares each row in the manual file to the DB file (based on CLAIM HCC ID).

Writes mismatches into an output Excel file with full rows (actual vs expected).

✅ Python Script: compare_excels.py
import pandas as pd
from io import BytesIO

# --- Configurable Inputs ---
manual_file = "manual.xlsx"
db_file = "db.xlsx"
output_file = "Mismatch_Report.xlsx"

# Mapping: manual column → db column
column_mapping = {
    "CLAIM HCC ID": "CLAIM_ID",
    "IS_FIRST_PASS_AUTO_ADJUDICATE": "IS_FIRST_PASS_AUTO_ADJUDICATE",
    "IS_ADJUSTI": "IS_ADJUSTI",
    "CLAIM_STATI": "CLAIM_STATI",
    "F": "F",
    "G": "G",
    "Actual Error Message": "Actual Error Message"
}

# --- Helper function to normalize values ---
def normalize(val):
    if pd.isnull(val):
        return ""
    return str(val).strip().strip("'").lower()

# --- Load data ---
df_manual = pd.read_excel(manual_file)
df_db = pd.read_excel(db_file)

# --- Compare Logic ---
mismatches = []

for idx, manual_row in df_manual.iterrows():
    # Normalize and lookup by CLAIM HCC ID
    claim_id_manual_col = "CLAIM HCC ID"
    claim_id_db_col = column_mapping[claim_id_manual_col]
    manual_claim_id = normalize(manual_row[claim_id_manual_col])

    candidates = df_db[
        df_db[claim_id_db_col]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.lower() == manual_claim_id
    ]

    if candidates.empty:
        mismatches.append({
            "RowIndex": idx,
            "Status": "❌ Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        db_row = candidates.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            m_val = normalize(manual_row.get(m_col))
            d_val = normalize(db_row.get(d_col))
            if m_val != d_val:
                mismatch_found = True
                break

        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "⚠️ Value Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })















import pandas as pd

# Input files and config
db_file = "db_file.xlsx"
manual_file = "manual_file.xlsx"
output_file = "comparison_result.xlsx"

# Sheet names (optional if you want default sheet)
db_sheet = "Sheet1"
manual_sheet = "Sheet1"

# Mapping: manual_col -> db_col
column_mapping = {
    "Manual_Name": "DB_Name",
    "Manual_ID": "DB_ID",
    "Manual_Date": "DB_Date"
}

# Load files
df_db = pd.read_excel(db_file, sheet_name=db_sheet)
df_manual = pd.read_excel(manual_file, sheet_name=manual_sheet)

# Convert mapping into two lists
manual_cols = list(column_mapping.keys())
db_cols = [column_mapping[col] for col in manual_cols]

# Store mismatches
mismatches = []

# Iterate each row in manual file
for idx, manual_row in df_manual.iterrows():
    # Try to find matching row in DB based on unique identifier or all fields
    matches = df_db.copy()
    
    for m_col, d_col in column_mapping.items():
        matches = matches[matches[d_col] == manual_row[m_col]]

    if matches.empty:
        # No match found
        mismatches.append({
            "RowIndex": idx,
            "Status": "Not Found in DB",
            "Manual_Row": manual_row.to_dict(),
            "Expected_DB_Row": None
        })
    else:
        # Found potential match, compare field by field
        db_row = matches.iloc[0]
        mismatch_found = False
        for m_col, d_col in column_mapping.items():
            if pd.isnull(manual_row[m_col]) and pd.isnull(db_row[d_col]):
                continue
            elif manual_row[m_col] != db_row[d_col]:
                mismatch_found = True
                break
        if mismatch_found:
            mismatches.append({
                "RowIndex": idx,
                "Status": "Mismatch",
                "Manual_Row": manual_row.to_dict(),
                "Expected_DB_Row": db_row.to_dict()
            })

# Create a flat dataframe from mismatches
records = []
for item in mismatches:
    row = {
        "RowIndex": item["RowIndex"],
        "Status": item["Status"]
    }
    # Flatten the fields
    for k, v in (item["Manual_Row"] or {}).items():
        row[f"Manual_{k}"] = v
    for k, v in (item["Expected_DB_Row"] or {}).items():
        row[f"Expected_{k}"] = v
    records.append(row)

# Save to Excel
output_df = pd.DataFrame(records)
output_df.to_excel(output_file, index=False)
print(f"Comparison completed. Mismatches saved to: {output_file}")














with open("output.txt", "w") as file:
    s = "|".join(col for col in all_columns)
    file.write(s + "\n")  # Write header with newline

    for row in all_data_rows:
        s = "|".join(row_val for row_val in row)
        file.write(s + "\n")  # Write each row with newline


def format_cell(val):
    val = val.strip()
    if val == "":
        return ""
    # Wrap leading zero numbers as string
    if val.isdigit() and val.startswith("0"):
        return f"'{val}"
    # Wrap date-like formats as string (basic check)
    if "-" in val or "/" in val:
        return f"'{val}"
    return val

filled = [format_cell(val) for val in filled]












import os
from glob import glob
from itertools import zip_longest

# ✅ Replace with your actual parsing function
def parse_multiple_xml_etree(file_path):
    """
    Expected to return a list of (result_dict, tag_order_list) for each file.
    Each result_dict: column → list of values
    """
    # Mock example:
    return [
        ({
            'ClaimID': ['001', '002'],
            'PatientName': ['John Doe', 'Jane Smith']
        }, ['ClaimID', 'PatientName'])
    ]

# ✅ Folder with XML files
folder_path = r"C:\path\to\xml\folder"  # <-- Change this
output_file = "merged_output.txt"

all_columns = []
all_data_rows = []

def format_cell(val):
    val = val.strip() if isinstance(val, str) else val
    return str(val) if val is not None else ""

# 🔁 Loop through all XML files in folder
for file_path in glob(os.path.join(folder_path, "*.xml")):
    print(f"📄 Processing: {os.path.basename(file_path)}")
    parsed_rows = parse_multiple_xml_etree(file_path)

    for result, tag_order in parsed_rows:
        # Track column order
        new_cols = [col for col in tag_order if col not in all_columns]
        all_columns.extend(new_cols)

        max_len = max((len(v) for v in result.values()), default=1)
        broadcasted = {}

        for col in all_columns:
            values = result.get(col, [])
            if not values:
                broadcasted[col] = [""] * max_len
                continue

            first_val = next((val for val in values if val.strip()), "")
            filled = [val if val.strip() else first_val for val in values]
            if len(filled) < max_len:
                filled += [first_val] * (max_len - len(filled))

            # Preserve leading zero numbers
            filled = [f"'{val}" if isinstance(val, str) and val.isdigit() and val.startswith("0") else val for val in filled]
            broadcasted[col] = [format_cell(val) for val in filled]

        rows = list(zip(*[broadcasted[k] for k in all_columns]))
        for row in rows:
            extended_row = list(row) + [""] * (len(all_columns) - len(row))
            all_data_rows.append(extended_row)

# 📝 Write merged TXT file (tab-separated)
with open(output_file, "w", encoding="utf-8") as file:
    file.write('\t'.join(all_columns) + "\n")
    for row in all_data_rows:
        file.write('\t'.join(str(val) for val in row) + "\n")

print(f"\n✅ Done! Output saved to: {output_file}")








import os
from glob import glob
from itertools import zip_longest

# ✅ Replace with your actual parsing function
def parse_multiple_xml_etree(file_path):
    """
    Expected to return a list of (result_dict, tag_order_list) for each file.
    Each result_dict: column → list of values
    """
    # Mock example:
    return [
        ({
            'ClaimID': ['001', '002'],
            'PatientName': ['John Doe', 'Jane Smith']
        }, ['ClaimID', 'PatientName'])
    ]

# ✅ Folder with XML files
folder_path = r"C:\path\to\xml\folder"  # <-- Change this
output_file = "merged_output.txt"

all_columns = []
all_data_rows = []

def format_cell(val):
    val = val.strip() if isinstance(val, str) else val
    return str(val) if val is not None else ""

# 🔁 Loop through all XML files in folder
for file_path in glob(os.path.join(folder_path, "*.xml")):
    print(f"📄 Processing: {os.path.basename(file_path)}")
    parsed_rows = parse_multiple_xml_etree(file_path)

    for result, tag_order in parsed_rows:
        # Track column order
        new_cols = [col for col in tag_order if col not in all_columns]
        all_columns.extend(new_cols)

        max_len = max((len(v) for v in result.values()), default=1)
        broadcasted = {}

        for col in all_columns:
            values = result.get(col, [])
            if not values:
                broadcasted[col] = [""] * max_len
                continue

            first_val = next((val for val in values if val.strip()), "")
            filled = [val if val.strip() else first_val for val in values]
            if len(filled) < max_len:
                filled += [first_val] * (max_len - len(filled))

            # Preserve leading zero numbers
            filled = [f"'{val}" if isinstance(val, str) and val.isdigit() and val.startswith("0") else val for val in filled]
            broadcasted[col] = [format_cell(val) for val in filled]

        rows = list(zip(*[broadcasted[k] for k in all_columns]))
        for row in rows:
            extended_row = list(row) + [""] * (len(all_columns) - len(row))
            all_data_rows.append(extended_row)

# 📝 Write merged TXT file (tab-separated)
with open(output_file, "w", encoding="utf-8") as file:
    file.write('\t'.join(all_columns) + "\n")
    for row in all_data_rows:
        file.write('\t'.join(str(val) for val in row) + "\n")

print(f"\n✅ Done! Output saved to: {output_file}")

















import os
from glob import glob
from itertools import zip_longest

# 🔁 Replace with your actual parser function
def parse_multiple_xml_etree(file_path):
    # This is just a placeholder
    return [
        ({
            'ClaimID': ['001'], 
            'PatientName': ['John Doe']
        }, ['ClaimID', 'PatientName'])
    ]

# 📁 Folder containing XML files
folder_path = r'C:\path\to\xml\folder'  # <-- change this
output_file = 'merged_output.txt'

all_data_rows = []
header_written = False
final_columns = []

# 🔁 Process each XML file
for file_path in glob(os.path.join(folder_path, "*.xml")):
    print(f"📄 Parsing: {os.path.basename(file_path)}")
    
    parsed_rows = parse_multiple_xml_etree(file_path)
    
    for result, tag_order in parsed_rows:
        if not header_written:
            final_columns = tag_order
            header_written = True
        
        # Prepare rows (aligned by columns)
        max_len = max((len(v) for v in result.values()), default=1)
        row_data = {col: result.get(col, [""] * max_len) for col in final_columns}
        rows = list(zip_longest(*[row_data[col] for col in final_columns], fillvalue=""))
        all_data_rows.extend(rows)

# 📝 Write to text file (tab-separated)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\t'.join(final_columns) + '\n')
    for row in all_data_rows:
        f.write('\t'.join(str(val) for val in row) + '\n')

print(f"\n✅ All XML data merged and written to: {output_file}")











import pyodbc
import pandas as pd

# Define connection
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=172.28.196.3,1433;'
    'DATABASE=PowerSTEPP;'
    'Trusted_Connection=yes;'
)

# Table to export
table_name = 'dbo.Employees'  # Replace with your actual table name

# Read entire table into DataFrame
query = f"SELECT * FROM {table_name}"
df = pd.read_sql(query, conn)

# Export to Excel
output_path = 'employees_data.xlsx'  # Change file name/path as needed
df.to_excel(output_path, index=False)

print(f"✅ Data exported to Excel: {output_path}")

conn.close()








import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=172.28.196.3,1433;'
    'DATABASE=PowerSTEPP;'
    'Trusted_Connection=yes;'
)

cursor = conn.cursor()

# Get all table names
cursor.execute("""
    SELECT TABLE_SCHEMA, TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE'
""")

print("📋 Tables in PowerSTEPP database:")
for row in cursor.fetchall():
    print(f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}")

cursor.close()
conn.close()








import pyodbc

# Define connection string
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=MHP01HEALTHEDG;'
    'DATABASE=your_database_name;'
    'UID=your_username;'
    'PWD=your_password'
)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sys.databases")

for row in cursor.fetchall():
    print("📁", row[0])

conn.close()










ChatGPT
Saved memory full
You said:
This is my resume i need to answer about me in 900 characters

● 4 years of experience in Deep learning, Computer vision and Image processing software design 
and development for desktop and embedded environments
▶ Experience in structural and object-oriented programming.
▶ Experience in Supervised and Un-Supervised Learning.
▶ Experience in working with ARMNN, Snapdragon, Advantech, Intel NUC, neural compute stick,
Thermal sensor with Raspberry-pi.
▶ Experience in RAG with LLM’s and using opensource models.
▶ Experience with AWS, Azure and few other popular API based LLM invoke Mechanisms.
▶ Having knowledge on Generative AI, Active Learning, Knowledge Distillation, Pruning, Hierarchal
learning, NLP and Curriculum Learning.
▶ Experience in porting the Deep learning models to light weight platforms. (Micro
Processors/Micro Controllers)
▶ Experience in Data Clustering Techniques.
▶ Experience with Lidar data for object detection and classification along with lane segmentation.
▶ Experience with Medical Imaging like cell classification, x-ray enhancement, cell segmentation.
▶ Having Knowledge on DETR where Transformers can be used for detecting objects in an image.
Also other architectures like HDRNET, R2CNN.
▶ Experience in developing image processing algorithms in python, Matlab(beginner).
▶ Experience in developing computer vision solutions using OpenCV,Scikit-image, Matplotlib.
▶ Design and develop deep learning algorithms using Keras, Tensorflow, TF-Lite, Pytorch,OPENVINO
▶ Experience in developing deep learning application for object detection using YOLO, RESNET,
Mobilenet, Squeezenet, Peleenet and custom designed DL Model and so on.
▶ Experience in using Segmentation architectures like U-Net,E-net,ResUnet.
▶ Experience in using LLAMA, GPT and other LLM models through fastchat and RAG enabling
seamless communication with PDF and other documentations.
▶ Experience in Object Tracking and Monitoring and worked on Homograph for multiple cameras
feed.
▶ Experience with FFMPEG, Gstreamer, Yocto Linux platforms.
▶ Experience in developing deep learning application for person reidentification using siamese
neural networks.
▶ Experience in developing deep learning application for single shot object recognition using AI
model trained using triplet loss.
▶ Experience in developing deep learning application for sequential data using LSTM(RNNS),
Transformers.
▶ Experience in using Auto encoders for identification of defects in the image.
▶ Knowledge of using GANs for semantic segmentation and also had experiencing in rewriting GAN
for novelty features.
▶ Experience in using Multi task loss and Active learning for improving the accuracy of Deep
ChatGPT said:
Here is a concise and impactful "About Me" response within 900 characters, based on your resume:

I’m a Deep Learning and Computer Vision Engineer with 4 years of experience building and deploying AI solutions across embedded, desktop, and cloud platforms. I specialize in model optimization and porting DL models to edge devices like ARMNN, Intel NUC, and Raspberry Pi. I’ve worked extensively with YOLO, U-Net, ResNet, and transformers like DETR for object detection, segmentation, and classification—including Lidar and medical imaging use cases. I have hands-on experience with LLMs and RAG pipelines using open-source models, FastChat, AWS, and Azure. My strengths include Generative AI, Active Learning, and real-time video analytics using FFMPEG/GStreamer. I bring a strong foundation in both traditional image processing and deep learning, with a focus on efficient, scalable AI applications.














import pandas as pd
from openpyxl import load_workbook

# Step 1: Load the source Excel
file_path = 'your_file.xlsx'
sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')

# Step 2: Copy column structure for the new sheet
prd_sample_df = pd.DataFrame(columns=sheet1_df.columns)

# Step 3: Apply transformations to a temporary copy
temp_df = sheet1_df.copy()

# Format Correction: Fix ClaimType
temp_df['ClaimType'] = temp_df['ClaimType'].replace({
    'Institutional Inpatient': 'Institutional',
    'Institutional Outpatient': 'Institutional'
})

# Format Correction: Fix POS (prefix 00)
temp_df['POS'] = temp_df['POS'].apply(lambda x: f"{int(x):04}" if pd.notnull(x) else x)

# Step 4: Define your column mapping (from Sheet1 → prd_sample)
column_mapping = {
    'ClaimType': 'ClaimType',
    'POS': 'POS'
}

# Step 5: Populate values into prd_sample_df using the mapping
for src_col, target_col in column_mapping.items():
    if src_col in temp_df.columns and target_col in prd_sample_df.columns:
        prd_sample_df[target_col] = temp_df[src_col]

# Step 6: Write both sheets into a new Excel file
with pd.ExcelWriter('updated_output.xlsx', engine='openpyxl', mode='w') as writer:
    sheet1_df.to_excel(writer, sheet_name='Sheet1', index=False)
    prd_sample_df.to_excel(writer, sheet_name='prd_sample', index=False)

print("✅ Sheet 'prd_sample' created and updated successfully.")









import pandas as pd

# Step 1: Load the Excel file
file_path = 'your_file.xlsx'

# Read both sheets
sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')
sheet2_df = pd.read_excel(file_path, sheet_name='Sheet2')

# Step 2: Define column mapping from Sheet1 to Sheet2
column_mapping = {
    'ClaimType': 'ClaimTypeMapped',  # e.g., Sheet1["ClaimType"] → Sheet2["ClaimTypeMapped"]
    'POS': 'POSMapped'
}

# Step 3: Apply transformations before mapping

# Fix ClaimType: Institutional Inpatient → Institutional
sheet1_df['ClaimType'] = sheet1_df['ClaimType'].replace({
    'Institutional Inpatient': 'Institutional',
    'Institutional Outpatient': 'Institutional'  # Optional: group both
})

# Fix POS: if value is numeric 31, make it string and prefix 00 → "0031"
sheet1_df['POS'] = sheet1_df['POS'].apply(lambda x: f"{int(x):04}" if pd.notnull(x) else x)

# Step 4: Copy mapped columns from Sheet1 to Sheet2
for src_col, target_col in column_mapping.items():
    if src_col in sheet1_df.columns and target_col in sheet2_df.columns:
        sheet2_df[target_col] = sheet1_df[src_col]
    else:
        print(f"⚠️ Column not found: {src_col} or {target_col}")

# Step 5: Save the updated Excel with both sheets
with pd.ExcelWriter('updated_output.xlsx', engine='openpyxl', mode='w') as writer:
    sheet1_df.to_excel(writer, sheet_name='Sheet1', index=False)
    sheet2_df.to_excel(writer, sheet_name='Sheet2', index=False)

print("✅ Excel updated with mapped columns and corrected values.")












import pandas as pd

# Step 0: Define relevant claim types
claim_types = ['Institutional Inpatient', 'Institutional Outpatient', 'Professional']
selected_rows = []

# Step 1: Loop through each claim type
for claim_type in claim_types:
    ct_df = merged_df[merged_df['ClaimType'] == claim_type]
    
    # Step 2: Get top 10 diagnosis codes for this claim type
    top_10_diag = ct_df['DiagCode'].value_counts().nlargest(10).index
    
    for diag in top_10_diag:
        diag_subset = ct_df[ct_df['DiagCode'] == diag]
        
        # Top 3 service codes and providers for this diag within claim type
        top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
        top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index
        
        # Get top 3 rows per service code
        for serv in top_3_serv:
            temp = diag_subset[diag_subset['ServCode'] == serv].head(3)
            selected_rows.append(temp)
        
        # Get top 3 rows per provider
        for prov in top_3_prov:
            temp = diag_subset[diag_subset['Provider'] == prov].head(3)
            selected_rows.append(temp)

    # Step 3: Add top 3 high-dollar claims from this claim type
    top_amount_claims = ct_df.sort_values('AmountPaid', ascending=False).head(3)
    selected_rows.append(top_amount_claims)

# Step 4: Combine everything
final_df = pd.concat(selected_rows).drop_duplicates().reset_index(drop=True)

# Step 5: Export or print
final_df.to_excel('ClaimTypeWise_TopClaims.xlsx', index=False)

print("✅ Final Data Preview:")
print(final_df.head(10))







import pandas as pd

# Step 0: Define valid ClaimTypes and filter
valid_claim_types = ['Institutional Inpatient', 'Institutional Outpatient', 'Professional']
filtered_df = merged_df[merged_df['ClaimType'].isin(valid_claim_types)]

# Step 1: Get top 10 diagnosis codes from filtered data
top_10_diag = filtered_df['DiagCode'].value_counts().nlargest(10).index

# Container for selected rows
selected_rows = []

# Step 2: Loop through each top diagnosis code
for diag in top_10_diag:
    diag_subset = filtered_df[filtered_df['DiagCode'] == diag]
    
    # Top 3 service codes and top 3 providers for this diagnosis
    top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index

    # Get top 3 rows for each ServCode
    for serv in top_3_serv:
        temp = diag_subset[diag_subset['ServCode'] == serv].head(3)
        selected_rows.append(temp)
    
    # Get top 3 rows for each Provider
    for prov in top_3_prov:
        temp = diag_subset[diag_subset['Provider'] == prov].head(3)
        selected_rows.append(temp)

# Step 3: Add Top 3 high-dollar claims from each ClaimType
top_amount_claims = (
    filtered_df.sort_values('AmountPaid', ascending=False)
    .groupby('ClaimType')
    .head(3)
)
selected_rows.append(top_amount_claims)

# Step 4: Add Top 3 claims by summary stats (optional: based on highest AmountPaid)
top_stats_claims = (
    filtered_df.groupby('ClaimType', group_keys=False)
    .apply(lambda x: x.sort_values('AmountPaid', ascending=False).head(3))
)
selected_rows.append(top_stats_claims)

# Step 5: Combine all collected rows
final_df = pd.concat(selected_rows).drop_duplicates().reset_index(drop=True)

# Step 6: Export or preview
final_df.to_excel('Filtered_TopClaims_By_Diag_Service_Provider_Pay.xlsx', index=False)

print("✅ Final Data Preview:")
print(final_df.head(10))










import pandas as pd

# Step 1: Get top 10 diagnosis codes
top_10_diag = merged_df['DiagCode'].value_counts().nlargest(10).index

# Final result container
all_selected_rows = []

# Step 2: For each diagnosis code
for diag in top_10_diag:
    diag_subset = merged_df[merged_df['DiagCode'] == diag]
    
    # Top 3 service codes and providers for that diagnosis
    top_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_prov = diag_subset['Provider'].value_counts().nlargest(3).index

    # For each top service code → get top 3 rows
    for serv in top_serv:
        top_rows_serv = diag_subset[diag_subset['ServCode'] == serv].head(3)
        all_selected_rows.append(top_rows_serv)

    # For each top provider → get top 3 rows
    for prov in top_prov:
        top_rows_prov = diag_subset[diag_subset['Provider'] == prov].head(3)
        all_selected_rows.append(top_rows_prov)

# Step 3: Combine and remove duplicates
final_df = pd.concat(all_selected_rows).drop_duplicates().reset_index(drop=True)

# Step 4: Optional — Export to Excel
final_df.to_excel('TopDiag_ServiceProvider_LimitedRows.xlsx', index=False)

# Preview
print(final_df.head())








import pandas as pd

# Step 1: Get Top 10 DiagCodes
top_10_diag = merged_df['DiagCode'].value_counts().nlargest(10).index

# Step 2: Initialize final results list
all_matches = []

# Step 3: Loop through each top diagnosis
for diag in top_10_diag:
    diag_subset = merged_df[merged_df['DiagCode'] == diag]
    
    # Get top 3 service codes and top 3 providers for this diagnosis
    top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index
    
    # Filter full rows for either top 3 service codes or top 3 providers
    serv_matches = diag_subset[diag_subset['ServCode'].isin(top_3_serv)]
    prov_matches = diag_subset[diag_subset['Provider'].isin(top_3_prov)]
    
    # Combine and deduplicate
    combined_matches = pd.concat([serv_matches, prov_matches]).drop_duplicates()
    
    # Append to final list
    all_matches.append(combined_matches)

# Step 4: Combine all into a single DataFrame
final_df = pd.concat(all_matches).drop_duplicates().reset_index(drop=True)

# Step 5: Optional — Export to Excel with all fields
final_df.to_excel('TopDiag_FullDetails.xlsx', index=False)

# Preview first few rows
print(final_df.head())












{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2062, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 2089, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 2114, 67, 68, 12357, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 4181, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 2160, 113, 114, 115, 112, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 86152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 165, 167, 20648, 169, 170, 171, 172, 174, 175, 176, 177, 178, 30899, 179, 181, 182, 183, 184, 185, 187, 188, 190, 191, 193, 2242, 8387, 196, 197, 200, 201, 202, 2251, 204, 207, 41, 211, 212, 213, 215, 217, 2266, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 234, 235, 238, 2287, 240, 241, 242, 4339, 244, 4341, 243, 247, 248, 249, 250, 251, 253, 255, 256, 258, 260, 262, 264, 2315, 267, 269, 4368, 272, 274, 276, 277, 279, 280, 281, 283, 284, 288, 290, 291, 292, 294, 298, 301, 302, 303,
304, 305, 6450, 307, 306, 309, 310, 312, 313, 315, 316, 14653, 320, 10561, 322, 321, 326, 327, 330, 4428, 333, 332, 66, 336, 338, 4436, 342, 344, 347, 348, 349, 69, 4447, 355, 14, 357, 358, 359, 360, 362, 367, 10608, 6515, 375, 378, 381, 384, 2433, 387, 388, 389, 391, 392, 393, 397, 400, 401, 404, 405, 413, 417, 418, 422, 425, 426, 85, 431, 433, 435, 436, 440, 2489, 448, 454, 459, 466, 467, 471, 477, 478, 480, 483, 486, 489, 490, 491, 492, 493, 503, 505, 507, 509, 511, 514, 522, 532, 534, 536, 538, 539, 542, 543, 544, 545, 562, 567, 568, 580, 582, 583, 2632, 588, 6734, 591, 590, 2642, 594, 599, 602, 606, 2662, 621, 4721, 10867, 627, 8821, 648, 651, 653, 654, 655, 664, 671, 672, 680, 682, 683, 136, 687, 688, 689, 692, 693, 694, 697, 710, 715, 21198, 719, 722, 724, 725, 727, 731, 733, 735, 739, 2788, 2789, 2790, 744, 749, 751, 758, 762, 763, 774, 781, 795, 799, 800, 804, 808, 815, 822, 829, 830, 9026, 858, 870, 872, 873, 2935, 894, 895, 896, 2952, 911, 913, 2977, 5030, 954, 957, 960, 965, 25542, 9167, 980, 989, 1003, 7148, 3054, 1018, 1025, 13326, 1041, 1049, 1051, 7199, 1059, 1063, 1077, 1081, 1088, 3157, 1123, 1126, 1136, 5240, 1148, 5246, 3199, 1158, 1160, 1162, 1168, 1172, 7319, 1189, 1200, 1208, 19641, 1213, 3262, 3263, 1215, 1229, 1232, 1238, 1241, 1248, 1252, 1258, 1259, 1265, 1270, 1271, 1274, 1276, 3325, 1280, 1281, 1293, 1300, 7454, 15659, 1325, 1334, 1338, 1355, 1359, 1366, 1367, 1385, 9579, 1414, 1436, 1442, 1463, 1467, 1477, 26066, 13782, 5592, 1506, 1512, 1564, 1584, 30261, 1610, 1612, 3690, 7788, 1654, 5764, 1690, 3744, 11953, 1724, 1729, 1730, 1740, 1751, 1752, 1754, 14058, 1782, 1793, 1802, 1817, 1823, 14135, 1851, 1865, 1874, 1908, 1910, 12156, 1942, 1976, 1983, 6084, 10191, 2021, 2022, 28648, 2029, 2032, 4086)






unique_claims = claim_trend['ClaimType'].unique()
palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Plotly

color_map = {claim: palette[i % len(palette)] for i, claim in enumerate(unique_claims)}









claim_types = claim_trend['ClaimType'].unique()
color_map = {ct: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, ct in enumerate(claim_types)}

# Plot with consistent color mapping
fig = px.bar(
    claim_trend,
    x="YearMonth",
    y="ClaimCount",
    color="ClaimType",
    color_discrete_map=color_map,
    barmode="group",
    title="Monthly Claim Trends"
)








# Ensure your merged_df has the 'YearMonth' column properly formatted
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%Y-%B')

# Step 1: Sort and calculate percentage change (already assumed to be done)
# claim_trend has columns: ['YearMonth', 'ClaimType', 'ClaimCount', 'PercentChange']

# Step 2: Filter for percentage drop below -20%
significant_drop = claim_trend[claim_trend['PercentChange'] < -20]

# Step 3: Get all claims from current and previous month for that ClaimType
all_records = []

for _, row in significant_drop.iterrows():
    curr_ym = row['YearMonth']
    claim_type = row['ClaimType']
    
    # Get index of current month in order list
    idx = month_order.index(curr_ym)
    
    if idx == 0:
        continue  # No previous month to compare
    
    prev_ym = month_order[idx - 1]

    # Filter merged_df for both months and claim type
    relevant_claims = merged_df[
        (merged_df['ClaimType'] == claim_type) &
        (merged_df['YearMonth'].isin([prev_ym, curr_ym]))
    ][['YearMonth', 'ClaimType', 'ClaimID', 'ServCode', 'RevenueCode', 'DiagCode', 'Provider']]
    
    all_records.append(relevant_claims)

# Step 4: Concatenate all results into one DataFrame
final_claims_df = pd.concat(all_records, ignore_index=True)

# Display or export final_claims_df
print(final_claims_df)










import pandas as pd

# Ensure date is in datetime format
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract Year-Month in the format 'YYYY-MMM' (e.g., '2022-Mar')
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%Y-%B')

# Create monthly claim count summary
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Custom month ordering
month_order = ['2022-March', '2022-July', '2022-August', '2022-September', '2022-October', '2022-November', '2022-December',
               '2023-January', '2023-February', '2023-March', '2023-April', '2023-May', '2023-June', '2023-July', '2023-August', 
               '2023-September', '2024-January', '2024-February', '2024-March', '2024-April', '2024-May', '2024-June', 
               '2024-July', '2024-August', '2024-September', '2025-January', '2025-February', '2025-March', '2025-April', 
               '2025-May', '2025-June']

# Set correct order for plotting/sorting
claim_trend['YearMonth'] = pd.Categorical(claim_trend['YearMonth'], categories=month_order, ordered=True)

# Sort by YearMonth and ClaimType for accurate diff calculation
claim_trend = claim_trend.sort_values(['ClaimType', 'YearMonth'])

# Calculate percentage change
claim_trend['PercentChange'] = claim_trend.groupby('ClaimType')['ClaimCount'].pct_change() * 100

# Optional: Round the percentage change
claim_trend['PercentChange'] = claim_trend['PercentChange'].round(2)

# Display the DataFrame
print(claim_trend.head(15))














import plotly.express as px
import pandas as pd

# Filter for selected claim types
selected_claims = ['Professional', 'Institutional Inpatient', 'Institutional Outpatient']
service_df = merged_df[merged_df['ClaimType'].isin(selected_claims)]

# Group and count
top_service_codes = (
    service_df.groupby(['ClaimType', 'RevCode'])
    .size()
    .reset_index(name="Count")
)

# Ensure RevCode is string
top_service_codes['RevCode'] = top_service_codes['RevCode'].astype(str)

# Get top 1 RevCode per ClaimType
top_service_codes = (
    top_service_codes
    .sort_values('Count', ascending=False)
    .groupby('ClaimType')
    .head(1)
)

# Create figure
fig_find6 = px.bar(
    top_service_codes,
    x='RevCode',
    y='Count',
    color='ClaimType',
    barmode='overlay',
    height=500,
    opacity=0.75,
    text='Count',
    title='Top Revenue Code per Claim Type'
)

# Update trace text
fig_find6.update_traces(textposition='auto')

# Update layout
fig_find6.update_layout(
    xaxis_title='Revenue Code',
    yaxis=dict(title='Count', type='log', showgrid=False, zerolinecolor='#060608'),
    bargap=0.1
)

# 🚨 Identify missing claim types
present_claim_types = top_service_codes['ClaimType'].unique().tolist()
missing_claim_types = [ct for ct in selected_claims if ct not in present_claim_types]

# Add annotation for each missing claim type
for i, missing in enumerate(missing_claim_types):
    fig_find6.add_annotation(
        text=f"No data for {missing}",
        xref="paper", yref="paper",
        x=1.05, y=1 - i * 0.05,  # adjust spacing between notes
        showarrow=False,
        font=dict(color="crimson", size=12)
    )

fig_find6.show()










from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd

# Just to be safe, ensure DOB is in datetime format
merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')

# Define today's date once
today = datetime.today()

# Define a function to calculate age
def calculate_age(dob):
    if pd.isnull(dob):
        return None
    # Convert pandas Timestamp to Python datetime
    dob = dob.to_pydatetime()
    return relativedelta(today, dob).years

# Apply the function to calculate Age
merged_df['Age'] = merged_df['DOB'].apply(calculate_age)

# View result
print(merged_df[['DOB', 'Age']].head())








import pandas as pd
from datetime import datetime

# Assuming your DataFrame is called merged_df and it has a 'DOB' column

# Make sure 'DOB' column is in datetime format
merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')

# Calculate age
today = pd.to_datetime('today')
merged_df['Age'] = (today - merged_df['DOB']).astype('<m8[Y]').astype(int)

# View result
print(merged_df[['DOB', 'Age']].head())









if not common_cols:
    st.error("🚫 No common columns found between Member and Claim data. Cannot proceed with merging.")
else:



import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Functions ----------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(" ", "")
    return df

def clean_values(df):
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df

def find_member_differences(df, key_col='MemberID'):
    dupes = df[df.duplicated(subset=key_col, keep=False)]
    differences = []
    for member, group in dupes.groupby(key_col):
        differing_cols = []
        for col in group.columns:
            if group[col].nunique() > 1:
                differing_cols.append(col)
        if differing_cols:
            row = {key_col: member}
            for col in differing_cols:
                row[col] = group[col].unique().tolist()
            differences.append(row)
    return pd.DataFrame(differences)

def generate_claim_report(merged_df):
    claim_count = merged_df['ClaimType'].value_counts()
    claim_percentages = (claim_count / len(merged_df)) * 100
    claim_percent_df = claim_percentages.reset_index()
    claim_percent_df.columns = ['ClaimType', 'Percentage']
    return claim_percent_df

def render_bar_chart(claim_percent_df):
    fig = px.bar(
        claim_percent_df,
        x='ClaimType',
        y='Percentage',
        title='Percentage Distribution of Claim Types',
        text='Percentage',
        color='ClaimType'
    )
    fig.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside',
        marker=dict(line=dict(width=1))
    )
    fig.update_layout(
        bargap=0.1,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis=dict(title='Percentage', showgrid=False, zerolinecolor='gray')
    )
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📊 Custom Data Visualization Tool")

with st.sidebar:
    st.header("Upload Excel Files")
    member_file = st.file_uploader("Upload Member File (.xlsx)", type=['xlsx'])
    claim_file = st.file_uploader("Upload Claim File (.xlsx)", type=['xlsx'])

    member_sheet = st.text_input("Enter Member Sheet Name", value="MemberPIC_20250609")
    claim_sheet = st.text_input("Enter Claim Sheet Name", value="Delimited")

    load_data = st.button("Load Data")

if load_data and member_file and claim_file:
    # Read files
    df_Member = pd.read_excel(member_file, sheet_name=member_sheet)
    df_Claim = pd.read_excel(claim_file, sheet_name=claim_sheet)

    st.subheader("🔍 Data Overview")
    st.write("✅ Member Data: ", df_Member.shape)
    st.write(df_Member.columns.tolist())

    st.write("✅ Claim Data: ", df_Claim.shape)
    st.write(df_Claim.columns.tolist())

    # Clean column names
    df_Member = clean_columns(df_Member)
    df_Claim = clean_columns(df_Claim)

    # Multiselect for common columns
    common_cols = list(set(df_Member.columns).intersection(set(df_Claim.columns)))
    selected_cols = st.multiselect("Select Common Column(s) to Merge On", common_cols, default=["MemberID"])

    if selected_cols:
        df_Member = clean_values(df_Member)
        df_Claim = clean_values(df_Claim)

        # Find differences in Member table (optional)
        diff_df = find_member_differences(df_Member, key_col=selected_cols[0])
        if not diff_df.empty:
            st.write("⚠️ Duplicate Member Entries with Differences")
            st.dataframe(diff_df)

        d1_deduped = df_Member.drop_duplicates(subset=selected_cols, keep='first')
        merged_df = pd.merge(df_Claim, d1_deduped, on=selected_cols, how='left')

        st.subheader("🧾 Merged DataFrame Info")
        st.write("Shape:", merged_df.shape)
        st.write("Columns:", merged_df.columns.tolist())
        st.dataframe(merged_df.head())

        if st.button("📈 Generate Claim Type Distribution Report"):
            try:
                claim_percent_df = generate_claim_report(merged_df)
                st.dataframe(claim_percent_df)
                st.plotly_chart(render_bar_chart(claim_percent_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")













_________________________
def find_column_in_dfs(dfs, target_column):
    found_in = []

    for idx, df in enumerate(dfs):
        if target_column in df.columns:
            found_in.append((idx, df.columns.tolist()))

    if found_in:
        print(f"\n✅ Column '{target_column}' found in {len(found_in)} DataFrame(s):\n")
        for idx, columns in found_in:
            print(f"➡️  DataFrame #{idx} contains it. Columns: {columns}")
    else:
        print(f"\n❌ Column '{target_column}' not found in any DataFrame.")

    return found_in











import os
import re
import pandas as pd

def collect_usecase_sheets(root_folder):
    dfs = []
    count = 0
    ARCHIVE = 'ARCHIVE'  # Define the ARCHIVE keyword or folder name

    # Regex pattern to match all variants of "Use Cases" sheet names
    use_case_pattern = re.compile(
        r'(?:\d+(?:\.\d+)?\s*\.*\s*)?(?:SKN_)?(?:Non-)?Aliessa_?Use_?Cases|Use[\s_]?Cases(?:_[A-Za-z0-9]+)?|Use_?Cases[\sA-Za-z0-9_]*'
    )

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.xlsm') and file.startswith('OMSAP'):
                file_path = os.path.join(dirpath, file)

                if 'ARCHIVE' in file_path.upper() or 'MAILS' in file_path.upper():
                    continue

                count += 1
                try:
                    xls = pd.ExcelFile(file_path)
                    sheet_names = xls.sheet_names

                    matched_sheet = next(
                        (s for s in sheet_names if use_case_pattern.fullmatch(s.strip())), None
                    )

                    if matched_sheet:
                        df = pd.read_excel(file_path, sheet_name=matched_sheet)
                        dfs.append(df)
                    else:
                        folder = file_path.split(os.sep)[2] if len(file_path.split(os.sep)) > 2 else "Unknown"
                        print(f"No 'Use Cases' sheet found in: {file_path} (Folder: {folder})")

                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

    print(f"Total matching files: {count}")
    return dfs












(?:\d+(?:\.\d+)?\s*\.*\s*)?(?:SKN_)?(?:Non-)?Aliessa_?Use_?Cases|Use[\s_]?Cases(?:_[A-Za-z0-9]+)?|Use_?Cases[\sA-Za-z0-9_]*











B20 (HIV Disease)
• Highest diagnosis consistently across all months
• Peaked in Oct 2024, slight drop by Mar 2025

E11.9 (Type 2 Diabetes without complications)
• Steady increase till Jan 2025
• Sharp drop observed in Mar 2025


F11.20 (Opioid dependence, uncomplicated)
• Rose steadily till Oct 2024
• Declined significantly after Feb 2025

I10 (Essential Hypertension)
• Gradual upward trend throughout
• Minor dip noted after Feb 2025

N18.6 (End stage renal disease)
• Very stable over the entire period
• Minimal fluctuations month-to-month

R68.89 (Other general symptoms and signs)
• Growth till Oct 2024, then leveled off
• Slight drop by Mar 2025

R69 (Unknown causes of morbidity/mortality)
• Minor rise around Dec 2024
• Dropped again by Mar 2025

Z00.00 (General medical exam without abnormal findings)
• Spike seen during Oct–Dec 2024
• Sharp decline by Mar 2025

Z21 (Asymptomatic HIV infection status)
• Fairly stable till Jan 2025
• Sudden decrease in the final month

Z76.89 (Other specified encounters)
• Consistently low volume throughout
• Nearly flat by Mar 2025


import pandas as pd
import plotly.express as px

# Step 1: Ensure 'ServDate' is datetime
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'], errors='coerce')

# Step 2: Create Year-Month column
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Step 3: Get top 10 diagnosis codes
top_10_diags = (
    merged_df['Diag']
    .value_counts()
    .head(10)
    .index
)

# Step 4: Filter for only top 10
diag_trend_df = merged_df[merged_df['Diag'].isin(top_10_diags)]

# Step 5: Group by YearMonth and Diag
trend_counts = (
    diag_trend_df.groupby(['YearMonth', 'Diag'])
    .size()
    .reset_index(name='Count')
)

# Step 6: Sort YearMonth properly
trend_counts['YearMonth'] = pd.to_datetime(trend_counts['YearMonth'])
trend_counts = trend_counts.sort_values('YearMonth')

# Step 7: Plot
fig = px.line(
    trend_counts,
    x='YearMonth',
    y='Count',
    color='Diag',
    markers=True,
    title='Trend Analysis of Top 10 Diagnosis Codes',
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Claim Count',
    xaxis=dict(
        tickformat="%b %Y",  # Format like 'Jan 2024'
        dtick="M1"
    ),
    plot_bgcolor='white',
    font_color="#303030",
    legend_title='Diagnosis Code'
)

fig.show()










1,,,,,

import pandas as pd
import plotly.express as px

# Replace this with your actual data
merged_df = pd.read_csv("your_data.csv")

# Top 5 providers by total bill
top_billers = (
    merged_df.groupby('Provider')['BillAmount']
    .sum()
    .reset_index()
    .sort_values(by='BillAmount', ascending=False)
    .head(5)
)

fig1 = px.bar(
    top_billers,
    x='Provider',
    y='BillAmount',
    text='BillAmount',
    title='Top 5 Providers by Total Billed Amount',
    color_discrete_sequence=['#a0d8ef']  # light blue
)
fig1.update_traces(textposition='outside')
fig1.update_layout(yaxis_title='Total Bill Amount', plot_bgcolor='white')
fig1.show()
2,,,,,,
top_submitters = (
    merged_df.groupby('Provider')
    .size()
    .reset_index(name='ClaimCount')
    .sort_values(by='ClaimCount', ascending=False)
    .head(5)
)

fig2 = px.bar(
    top_submitters,
    x='Provider',
    y='ClaimCount',
    text='ClaimCount',
    title='Top 5 Providers by Number of Claims Submitted',
    color_discrete_sequence=['#c6e2ff']  # soft blue
)
fig2.update_traces(textposition='outside')
fig2.update_layout(yaxis_title='Claim Count', plot_bgcolor='white')
fig2.show()
3,,,,,,

denied_df = merged_df[merged_df['PayStat'] == 'D']

top_denied = (
    denied_df.groupby('Provider')
    .size()
    .reset_index(name='DeniedCount')
    .sort_values(by='DeniedCount', ascending=False)
    .head(5)
)

fig3 = px.bar(
    top_denied,
    x='Provider',
    y='DeniedCount',
    text='DeniedCount',
    title='Top 5 Providers by Number of Denied Claims',
    color_discrete_sequence=['#ffcccc']  # light red
)
fig3.update_traces(textposition='outside')
fig3.update_layout(yaxis_title='Denied Claims', plot_bgcolor='white')
fig3.show()









import pandas as pd
import plotly.express as px

# Convert 'ServDate' to datetime if it's not already
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract Year-Month in desired format (e.g., Jun-2024)
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%b-%Y')

# Ensure proper month order for plotting
month_order = ['Jun-2024', 'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024',
               'Nov-2024', 'Dec-2024', 'Jan-2025', 'Feb-2025', 'Mar-2025']
merged_df['YearMonth'] = pd.Categorical(merged_df['YearMonth'], categories=month_order, ordered=True)

# Filter only denied claims
denied_df = merged_df[merged_df['PayStat'] == 'D']

# Group by month and claim type to get denial counts
denial_trend = denied_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='DenialCount')

# Plot line chart
fig = px.line(
    denial_trend,
    x='YearMonth',
    y='DenialCount',
    color='ClaimType',
    markers=True,
    title='Monthly Denial Trend by Claim Type'
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Denials',
    font_color="#303030",
    plot_bgcolor='white',
    legend_title_text='Claim Type'
)

fig.update_traces(line=dict(width=3), marker=dict(size=6))

fig.show()








import pandas as pd
import plotly.express as px

# Assume 'merged_df' is your dataframe and has columns 'ServDate' and 'ClaimType'
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract 'Year-Month' format from service date
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Group by YearMonth and ClaimType to get the count of claims per month
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Create ordered list of expected months (June 2024 to March 2025)
month_order = ['2024-06', '2024-07', '2024-08', '2024-09', '2024-10',
               '2024-11', '2024-12', '2025-01', '2025-02', '2025-03']

# Ensure all months show in the plot, even if counts are 0
claim_trend['YearMonth'] = pd.Categorical(claim_trend['YearMonth'], categories=month_order, ordered=True)
claim_trend = claim_trend.sort_values('YearMonth')

# Plot line graph
fig = px.line(
    claim_trend,
    x='YearMonth',
    y='ClaimCount',
    color='ClaimType',
    markers=True,
    line_shape='spline',
    title='Monthly Trend of Each Claim Type (Jun 2024 - Mar 2025)',
    color_discrete_sequence=px.colors.qualitative.Safe
)

fig.update_traces(line=dict(width=2))

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Claims',
    font_color="#303030",
    height=500,
    template='plotly_white',
    legend_title_text='Claim Type',
    xaxis=dict(
        tickangle=45,
        type='category',
        categoryorder='array',
        categoryarray=month_order
    )
)

fig.show()









import pandas as pd
import plotly.express as px

# Filter for relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional Inpatient', 'Institutional Outpatient'])]

# Map PayStat codes to readable labels
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({'P': 'Paid', 'D': 'Denied'})

# Drop rows with missing PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group by ClaimType and PayStat
claim_status_count = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Calculate percentage
total_counts = claim_status_count.groupby('ClaimType')['Count'].transform('sum')
claim_status_count['Percentage'] = (claim_status_count['Count'] / total_counts * 100).round(2)

# Label for display
claim_status_count['label'] = claim_status_count['Count'].astype(str) + ' (' + claim_status_count['Percentage'].astype(str) + '%)'

# Plot grouped bar chart
fig_find4 = px.bar(
    claim_status_count,
    x='ClaimType',
    y='Count',
    color='PayStat',
    barmode='group',
    text='label',
    title='Paid vs Denied Claim Counts by Claim Type',
    height=500,
    color_discrete_map={'Paid': '#90ee90', 'Denied': '#ff9999'}
)

# Style update
fig_find4.update_traces(textposition='outside')
fig_find4.update_layout(
    font_color="#303030",
    xaxis_title='Claim Type',
    yaxis=dict(title='Claim Count', type='log', showgrid=False, zerolinecolor='#DBDBDB'),
    legend_title='Claim Status',
    bargap=0.3
)







import pandas as pd
import plotly.express as px

# Convert ServDate to datetime
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract year-month
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Group by YearMonth and ClaimType
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Sort by YearMonth for proper line flow
claim_trend = claim_trend.sort_values('YearMonth')

# Line plot for each claim type
fig = px.line(
    claim_trend,
    x='YearMonth',
    y='ClaimCount',
    color='ClaimType',
    markers=True,
    line_shape='spline',
    title='Monthly Trend of Each Claim Type (Jun 2024 - Mar 2025)',
    color_discrete_sequence=px.colors.qualitative.Safe  # light + distinguishable
)

# Update layout
fig.update_traces(line=dict(width=2))
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Claims',
    font_color="#303030",
    height=500,
    template='plotly_white',
    legend_title_text='Claim Type',
    xaxis=dict(tickangle=45)
)

fig.show()












import plotly.express as px

# Bar chart with thicker bars
fig = px.bar(claim_percent_df,
             x='ClaimType',
             y='Percentage',
             title='Percentage Distribution of Claim Types',
             text='Percentage',
             color='ClaimType')

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside',
    marker=dict(line=dict(width=0)),  # No border, solid bars
)

fig.update_layout(
    width=800,
    height=500,
    bargap=0.1,  # Reduce space between bars
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    yaxis_title='Percentage'
)

fig.show()









Hi Ishita, Supreeth,

I've added two more findings to the deck. As requested, I’ve included interactive graphs by converting them into HTML outputs and linking them to the graph tiles in the presentation. Please download the ZIP file, open the presentation in slideshow mode, and click on any graph tile to view the corresponding interactive graph.

Thanks!







claims with bill amount over 10K, there are no professional claims; only institutional claims are present, with B20 being the most common diagnosis code in outpatient (49,955 claims), followed by Z21, Z00.00, R69, and F11.20.





Institutional - Outpatient has T1019 as the most used service code with 33,840 claims.

Institutional - Inpatient sees 99232 as the top service code with 1,679 claims.

Professional has 99309 as the most frequent service code with 105 claims.










import pandas as pd
import plotly.express as px

# Categorize Age Function
def categorize_age(age):
    if age == 0:
        return 'New Born'
    elif age <= 18:
        return '1 to 18 years'
    elif age <= 35:
        return '19 to 35 years'
    elif age <= 59:
        return '36 to 59 years'
    else:
        return '60 and over'

# Apply age category
plot_df['AgeGroup'] = plot_df['Age'].apply(categorize_age)

# Group by AgeGroup and ClaimType
plot_df = plot_df.groupby(['AgeGroup', 'ClaimType'])['ClaimType'].count().reset_index(name='ClaimTypeCount')

# Order age groups
age_order = ['New Born', '1 to 18 years', '19 to 35 years', '36 to 59 years', '60 and over']
plot_df['AgeGroup'] = pd.Categorical(plot_df['AgeGroup'], categories=age_order, ordered=True)
plot_df = plot_df.sort_values(['AgeGroup', 'ClaimType'])

# Create bar chart
fig_find1 = px.bar(
    plot_df,
    x='AgeGroup',
    y='ClaimTypeCount',
    color='ClaimType',
    text='ClaimTypeCount',
    opacity=0.75,
    barmode='group',
    title='Total Number of Claims by Age Group and Claim Type'
)

# Update chart aesthetics
fig_find1.update_traces(
    texttemplate='%{text:.0f}',
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig_find1.update_layout(
    font_color="#383038",
    bargroupgap=0.05,
    bargap=0.2,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(
        title='Claim Count',
        showgrid=False,
        zerolinecolor="#DBDBDB",
        showline=True,
        linecolor="#080808",
        linewidth=2,
        type='log'
    )
)
















fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Total Number of Claims by Age Group and Claim Type",
        "Top 5 Frequent Diagnoses by Age Group",
        "Top Claim Type by Number of Providers",
        "Paid vs Denied Claim Counts by Claim Type"
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "pie"}, {"type": "bar"}]]
)

# Add each figure to subplot
fig_dashboard.add_trace(fig_find1, row=1, col=1)
fig_dashboard.add_trace(fig_find1b, row=1, col=1)

fig_dashboard.add_trace(fig_find2, row=1, col=2)
fig_dashboard.add_trace(fig_find2b, row=1, col=2)























import pandas as pd
import plotly.express as px

# Step 1: Filter Institutional and Professional
claim_df = merged_df[merged_df['ClaimType'].isin(['Institutional', 'Professional'])]

# Step 2: Count total claims per ClaimType
claim_counts = claim_df.groupby('ClaimType').size().reset_index(name='TotalCount')

# Step 3: Filter only those ClaimTypes with count > 10,000
eligible_types = claim_counts[claim_counts['TotalCount'] > 10000]['ClaimType']
filtered_df = claim_df[claim_df['ClaimType'].isin(eligible_types)]

# Step 4: Count DiagnosisCode occurrences
top_diag = (
    filtered_df.groupby(['ClaimType', 'DiagnosisCode'])
    .size()
    .reset_index(name='Count')
    .sort_values(['ClaimType', 'Count'], ascending=[True, False])
    .groupby('ClaimType')
    .head(5)
)

# Step 5: Visualization
fig = px.bar(
    top_diag,
    x='DiagnosisCode',
    y='Count',
    color='ClaimType',
    barmode='group',
    text='Count',
    title='Top 5 Diagnosis Codes by Claim Type (Count > 10K)'
)

fig.update_traces(
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    xaxis_title='Diagnosis Code',
    yaxis_title='Claim Count',
    xaxis_tickangle=45,
    height=500
)

fig.show()








Institutional – Outpatient claims account for the majority of provider activity, representing 70.5% of all claim submissions. This indicates a strong emphasis on outpatient care across the provider network, likely driven by cost efficiency, accessibility, and broader coverage models.

 denial and payment rates are approximately 57.1% denied and 42.9% paid for Institutional Inpatient, 19.4% denied and 80.6% paid for Institutional Outpatient, and 22.8% denied and 77.2% paid for Professional services.





Diagnosis code B20 (HIV disease) consistently ranks as the top diagnosis for patients aged 19 and above, with the highest concentration seen in the 36–59 age group




“With Institutional claims comprising 86.16%—dominated by top provider JACOBI MEDICAL CENTER and top service code 99232—and Professional claims at just 0.123%—led by IVEY GRACE and service code 99309—the data reveals a sharp concentration of volume and impact around a few key players.”








import plotly.express as px

# Filter for selected claim types
benefit_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional Inpatient'])]

# Group and count
benefit_dist = benefit_df.groupby(['ClaimType', 'BenPkg']).size().reset_index(name='Count')

# Top 5 benefit codes per claim type
benefit_dist = benefit_dist.sort_values('Count', ascending=False).groupby('ClaimType').head(5)

# Bar chart
fig5 = px.bar(
    benefit_dist,
    x='BenPkg',
    y='Count',
    color='ClaimType',
    barmode='group',
    text='Count',
    title='Top 5 Benefit Codes by Claim Type',
    height=500
)

fig5.update_traces(textposition='outside')
fig5.update_layout(
    xaxis_title='Benefit Code',
    yaxis_title='Count',
    bargap=0.3,
    font_color="#303030",
    legend_title='Claim Type'
)

fig5.show()










import plotly.express as px

# Filter for relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin([
    'Professional', 'Institutional Inpatient', 'Institutional Outpatient'
])]

# Map PayStat codes
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({
    'P': 'Paid',
    'D': 'Denied'
})

# Drop rows with NaN PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group and count
claim_status_count = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Plot as grouped bar
fig = px.bar(
    claim_status_count,
    x='ClaimType',
    y='Count',
    color='PayStat',
    barmode='group',
    text='Count',
    title='Paid vs Denied Claim Counts by Claim Type',
    height=500
)

fig.update_traces(textposition='outside')
fig.update_layout(
    font_color="#303030",
    xaxis_title='Claim Type',
    yaxis_title='Claim Count',
    legend_title='Claim Status',
    bargap=0.3
)

fig.show()






import plotly.express as px

# Filter relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin([
    'Professional', 'Institutional Inpatient', 'Institutional Outpatient'
])]

# Map PayStat codes
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({
    'P': 'Paid',
    'D': 'Denied'
})

# Drop rows with NaN PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group and count
claim_status_pie = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Sunburst chart
fig2 = px.sunburst(
    claim_status_pie,
    path=['ClaimType', 'PayStat'],
    values='Count',
    title='Percentage of Paid vs Denied Claims by Claim Type'
)

fig2.update_traces(textinfo="label+percent entry")

fig2.show()







1 

import plotly.express as px
import pandas as pd

# Count of providers per claim type
provider_claim_counts = merged_df.groupby(['ClaimType'])['Provider'].nunique().reset_index(name='ProviderCount')

# Get top 1
top_provider_claim = provider_claim_counts.sort_values('ProviderCount', ascending=False).head(1)

fig1 = px.pie(top_provider_claim, names='ClaimType', values='ProviderCount',
              title='Top Claim Type by Number of Providers', hole=0.3)

fig1.show()


2

claim_status_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

claim_status_pie = claim_status_df.groupby(['ClaimType', 'ClaimStatus']).size().reset_index(name='Count')

fig2 = px.sunburst(claim_status_pie, path=['ClaimType', 'ClaimStatus'], values='Count',
                   title='Denied vs Paid Claim Distribution - Professional & Institutional')

fig2.show()

3

service_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

top_service_codes = service_df.groupby(['ClaimType', 'ServiceCode']).size().reset_index(name='Count')
top_service_codes = top_service_codes.sort_values('Count', ascending=False).groupby('ClaimType').head(10)

fig3 = px.bar(top_service_codes, x='ServiceCode', y='Count', color='ClaimType',
              barmode='group', text='Count', title='Top Service Codes by Claim Type')

fig3.update_traces(textposition='outside')
fig3.update_layout(xaxis_title='Service Code', yaxis_title='Count')

fig3.show()

4
diagnosis_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

top_diag_codes = diagnosis_df.groupby(['ClaimType', 'DiagnosisCode']).size().reset_index(name='Count')
top_diag_codes = top_diag_codes.sort_values('Count', ascending=False).groupby('ClaimType').head(10)

fig4 = px.bar(top_diag_codes, x='DiagnosisCode', y='Count', color='ClaimType',
              barmode='group', text='Count', title='Top Diagnosis Codes by Claim Type')

fig4.update_traces(textposition='outside')
fig4.update_layout(xaxis_title='Diagnosis Code', yaxis_title='Count')

fig4.show()

5


benefit_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

benefit_dist = benefit_df.groupby(['ClaimType', 'BenefitCode']).size().reset_index(name='Count')
benefit_dist = benefit_dist.sort_values('Count', ascending=False).groupby('ClaimType').head(5)

fig5 = px.sunburst(benefit_dist, path=['ClaimType', 'BenefitCode'], values='Count',
                   title='Benefit Code Distribution Across Claim Types')

fig5.show()








import pandas as pd
import plotly.express as px
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
px.defaults.template = "plotly_white"

# Create a copy
plot_df = merged_df.copy()

# Categorize Age groups
def categorize_age(age):
    if age == 0:
        return 'New Born'
    elif age <= 18:
        return '1 to 18 years'
    elif age <= 35:
        return '19 to 35 years'
    elif age <= 59:
        return '36 to 59 years'
    else:
        return '60 and over'

plot_df["AgeGroup"] = plot_df['Age'].apply(categorize_age)

# Group by AgeGroup and ClaimType and count
plot_df = plot_df.groupby(['AgeGroup', 'ClaimType'])['ClaimType'].count().reset_index(name='ClaimTypeCount')

# Sort by AgeGroup for better visual order
age_order = ['New Born', '1 to 18 years', '19 to 35 years', '36 to 59 years', '60 and over']
plot_df['AgeGroup'] = pd.Categorical(plot_df['AgeGroup'], categories=age_order, ordered=True)
plot_df = plot_df.sort_values(['AgeGroup', 'ClaimType'])

# Optional: Save to Excel
# plot_df.to_excel("Average_Bill_Amount_by_Age_and_ClaimType.xlsx", index=False)

# Create bar chart
fig = px.bar(
    plot_df,
    x='AgeGroup',
    y='ClaimTypeCount',
    color='ClaimType',
    text='ClaimTypeCount',
    opacity=0.75,
    barmode='group',
    height=500,
    title="Total Number of Claims by Age Group and Claim Type"
)

# Update styling
fig.update_traces(
    texttemplate='%{text:.0f}',
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargroupgap=0.05,
    bargap=0.3,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(title='Claim Count', showgrid=False, zerolinecolor='#DBDBDB', showline=True, linecolor='#DBDBDB', linewidth=2)
)

fig.show()









import xml.etree.ElementTree as ET
from collections import defaultdict
import re

def flatten_xml(elem, result, tag_order, prefix=""):
    for child in elem:
        tag = child.tag.split("}")[-1]
        key = f"{prefix}{tag}".lower()
        if key not in tag_order:
            tag_order.append(key)
        if child.text and child.text.strip():
            result[key].append(child.text.strip())
        flatten_xml(child, result, tag_order, key + "_")

def parse_multiple_xml_etree(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to match all <ConvertedSupplierInvoice>...</ConvertedSupplierInvoice> blocks
    pattern = re.compile(r"<ConvertedSupplierInvoice\b.*?>.*?</ConvertedSupplierInvoice>", re.DOTALL)
    matches = pattern.findall(content)

    all_rows = []

    for xml_fragment in matches:
        try:
            # Ensure it's valid XML before processing
            elem = ET.fromstring(xml_fragment)
            result = defaultdict(list)
            tag_order = []
            flatten_xml(elem, result, tag_order)

            max_len = max((len(v) for v in result.values()), default=0)
            for k in result:
                while len(result[k]) < max_len:
                    result[k].append("")

            all_rows.append((result, tag_order.copy()))
        except ET.ParseError as e:
            print(f"Skipped malformed XML block: {e}")
            continue

    return all_rows








# scrap

# Strip currency symbols, commas, etc.
df['BillAmt'] = df['BillAmt'].astype(str).str.replace(r'[^\d.]', '', regex=True)

# Convert to numeric
df['BillAmt'] = pd.to_numeric(df['BillAmt'], errors='coerce')

# Drop rows where BillAmt couldn't be converted
df = df.dropna(subset=['BillAmt'])






import pandas as pd
import plotly.express as px

# Bin BillAmt into ranges
bins = [475, 5000, 20000, 50000, 100000, 250000, 500000, 813439]
labels = ['<5K', '5K–20K', '20K–50K', '50K–100K', '100K–250K', '250K–500K', '>500K']
df['BillAmtBin'] = pd.cut(df['BillAmt'], bins=bins, labels=labels, include_lowest=True)

# Create AgeGroup
df['AgeGroup'] = df['Age'].apply(
    lambda x: '18-29' if x < 30 else
              '30-44' if x < 45 else
              '45-59' if x < 60 else
              '60+'
)

# Group the data to get counts
grouped_df = df.groupby(['BillAmtBin', 'ClaimType', 'Race', 'AgeGroup']).size().reset_index(name='Count')

# Create bar plot
fig = px.bar(
    grouped_df,
    x='BillAmtBin',
    y='Count',
    color='ClaimType',
    facet_col='Race',
    facet_row='AgeGroup',
    barmode='group',
    text='Count',
    title='Claim Count by Bill Amount Bin, Race, AgeGroup and Claim Type'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    font_color="#303030",
    xaxis_title="Bill Amount Range",
    yaxis_title="Number of Claims",
    height=900,
    bargap=0.3
)

fig.show()
----------------------------------------------------------------

# Group and count based on all requested dimensions
grouped_df = df.groupby(
    ['BillAmtBin', 'ClaimType', 'Race', 'AgeGroup', 'GrpNum', 'BenefitPackage']
).size().reset_index(name='Count')

# Create bar chart with faceting on Race, AgeGroup, GrpNum, BenefitPackage
fig = px.bar(
    grouped_df,
    x='BillAmtBin',
    y='Count',
    color='ClaimType',
    facet_col='BenefitPackage',   # Can also try 'Race' here depending on focus
    facet_row='GrpNum',           # Can also try 'AgeGroup' or 'Race' here
    barmode='group',
    text='Count',
    title='Claim Count by Bill Amount Bin, Claim Type, Benefit Package, and GrpNum'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    font_color="#303030",
    xaxis_title="Bill Amount Range",
    yaxis_title="Number of Claims",
    height=1200,
    bargap=0.3
)

fig.show()














Females lack claims for Group Home and Residential Substance Abuse Treatment, both of which are present for males (82 and 15 claims respectively).

Males have no claims for Mobile Unit services, which have 12 claims under females, indicating possible access or utilization differences.

Other claim types dominate for both genders, but especially for females (116,679), far exceeding all other categories, suggesting broad categorization or unclassified services.

Institutional - Outpatient claims are significantly higher in males (158,542) compared to females (71,160), indicating possible gender-based differences in outpatient service utilization.

Telehealth claims are more frequent in females (418) than males (257), reflecting either greater access, acceptance, or necessity among females for remote care.















Step 1: Removing Duplicate Members and Merging Datasets
To remove duplicate entries of members based on MemberNum in df1 while keeping the first occurrence.
To identify common columns between the cleaned df1 and df2 for potential merging.
Note – Validation of Member-Level Data Consistency
I checked the data with Member num i can see the duplicate values only in the Ethnicity. we can confirm that data is duplicated If my understanding is wrong I will make that change accordingly.

Profiling of Categorical Columns
To better understand the structure and distribution of non-numeric (categorical) data in the merged_d
# Step 1: Get the description of categorical columns
cat_profile = merged_df.select_dtypes(include=['object']).describe()

# Step 2: Store in Excel with sheet name
with pd.ExcelWriter('data_profiling.xlsx', engine='openpyxl') as writer:
    cat_profile.to_excel(writer, sheet_name='Categorical Profiling')


# Group by MemberNum and get number of unique provider names
member_provider_counts = df1.groupby('MemberNum')['ProvName'].nunique()

# Count members who visited more than one provider
multi_provider_members_count = (member_provider_counts > 1).sum()

print(f"Number of members who visited more than one provider: {multi_provider_members_count}")

multi_provider_members = member_provider_counts[member_provider_counts > 1].index.tolist()



The total number of diagnoses increases with age, with the highest counts in the "60 and over" and "45 to 59 years" groups.

One diagnosis category dominates across all age groups, especially in older ones (blue bar segment).

The "18 to 29 years" group has the lowest total diagnosis count, with very small bar segments.

Diagnosis diversity is greater in older age groups, indicated by more and larger colored segments in their bars.





Demographic & Provider Profiling: Analyzed key categorical features (Age, MemberNum, and ProvName) to understand claim distribution and average billing trends across member groups and providers.

Claim Concentration: Identified which age groups and providers had the highest number of claimholders, helping to spot clusters of utilization.

Cost Insights: Computed the average bill amount for each category, providing a view into which segments (e.g., certain providers or age groups) tend to incur higher costs




Institutional - Inpatient claims consistently show the highest average bill amounts across all age groups, with the 18 to 29 years group peaking at $4,239
Telehealth and End-Stage Renal Disease Treatment also show relatively high average bill amounts in older age groups (particularly in the 60 and over and 45 to 59 years categories)
Young adults (18 to 29 years) incur lower costs in most claim types, except for the unusually high Institutional - Inpatient expenses.
Services like Independent Lab, Mobile Unit, Other, and Group Home show consistently lower billing amounts.

ClaimTypes such as Mobile Unit, Group Home, and Residential Substance Abuse Treatment are absent in one or more age groups, suggesting:
These services are either not utilized or not recorded for those specific age ranges.
For example, Group Home and Mobile Unit are not present in the 30 to 44 and 45 to 59 age groups.
Telehealth, End-Stage Renal Disease Treatment, and Ambulance - Land tend to have higher average bills for older groups like 60 and over











from plotly.offline import plot, iplot, init_notebook_mode
import plotly.express as px

init_notebook_mode(connected=True)
px.defaults.template = "plotly_white"

# Copy and create age bins
plot_df = merged_df.copy()
plot_df["AgeGroup"] = plot_df['Age'].apply(
    lambda i: '18 to 29 years' if i < 30 else
              '30 to 44 years' if i < 45 else
              '45 to 59 years' if i < 60 else
              '60 and over'
)

# Count number of diagnoses per AgeGroup
diag_counts = (
    plot_df.groupby(['AgeGroup', 'Diag'])['MemberNum']
    .count()
    .reset_index(name='DiagCount')
)

# Get most frequent diagnosis per AgeGroup
top_diag_per_age = diag_counts.sort_values('DiagCount', ascending=False).groupby('AgeGroup').head(1)

# Plot
fig = px.bar(
    top_diag_per_age,
    x='AgeGroup',
    y='DiagCount',
    color='Diag',
    height=500,
    text='DiagCount',
    title="Most Frequent Diagnosis by Age Group"
)

fig.update_traces(
    texttemplate="%{text:,.0f}",
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargap=0.3,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(
        title='Number of Diagnoses',
        showgrid=False,
        zerolinecolor='#DBDBDB',
        showline=True,
        linecolor='#DBDBDB',
        linewidth=2
    )
)

fig.show()




-----------------------------------------------------

import plotly.express as px

# Group the data by Gender and ClaimType, and count frequencies
claim_counts = (
    df.groupby(['Gender', 'ClaimType'])['MemberNum']
    .count()
    .reset_index(name='ClaimCount')
)

# Plot with Plotly
fig = px.bar(
    claim_counts,
    x='Gender',
    y='ClaimCount',
    color='ClaimType',
    barmode='group',
    text='ClaimCount',
    title="Frequency of ClaimType by Gender"
)

# Styling the chart
fig.update_traces(
    texttemplate="%{text:,}",
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargap=0.3,
    legend_title_text='Claim Type',
    xaxis=dict(title='Gender', showgrid=False),
    yaxis=dict(title='Number of Claims', showgrid=False, zerolinecolor='#DBDBDB')
)

fig.show()

