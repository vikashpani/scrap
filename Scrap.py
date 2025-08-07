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
