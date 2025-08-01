



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
