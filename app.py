import streamlit as st
from streamlit_chat import message
import base64
import requests
import re
from io import BytesIO
from PIL import Image
import fitz
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Taha Ali RAG + Normal ChatBot", layout="wide")
st.title("🤖 TahaAliGPT - Document RAG + Normal Chat")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_URL = "https://api.groq.com/openai/v1/chat/completions"

# -------------------- SESSION STATE --------------------
# Document chat state
if 'doc_messages' not in st.session_state:
    st.session_state.doc_messages = []
if 'doc_vectorstore' not in st.session_state:
    st.session_state.doc_vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Normal chat state (new)
if 'normal_messages' not in st.session_state:
    st.session_state.normal_messages = []

# -------------------- TEXT EXTRACTION (GROQ VISION) - unchanged --------------------
def extract_text_from_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    prompt = "Extract all visible text from this document page. Preserve line breaks."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    try:
        response = requests.post(GROQ_VISION_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return ""
    except Exception as e:
        print(f"Vision error: {e}")
        return ""

def extract_text_from_pdf_with_groq(pdf_bytes: bytes, filename: str) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        st.info(f"📄 Processing {filename} - Page {page_num}...")
        text = extract_text_from_image(img)
        if text:
            pages_data.append({"page": page_num, "text": text})
    doc.close()
    return pages_data

def process_uploaded_files(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue
        filename = uploaded_file.name
        raw_pages = []
        if filename.endswith(".pdf"):
            raw_pages = extract_text_from_pdf_with_groq(uploaded_file.getvalue(), filename)
        else:  # .txt
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            if text.strip():
                raw_pages = [{"page": 1, "text": text}]
        if not raw_pages:
            st.warning(f"No text extracted from {filename}. Skipping.")
            continue
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for page_data in raw_pages:
            page_num = page_data["page"]
            chunks = text_splitter.split_text(page_data["text"])
            for chunk_text in chunks:
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"source": filename, "page": page_num}
                ))
        st.session_state.processed_files.add(filename)
    
    if not all_chunks:
        return None
    
    st.info(f"📚 Total chunks: {len(all_chunks)}. Building embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore

# -------------------- RAG: ANSWER WITHOUT SOURCES (MODIFIED) --------------------
def rag_answer(query: str, retrieved_docs: list) -> str:
    """
    Answer based on retrieved documents, but DO NOT show any source citations.
    """
    if not retrieved_docs:
        return "No relevant document chunks found."
    
    # Build numbered context (kept for LLM understanding, but we won't reveal sources)
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    system_prompt = f"""You are a document assistant. Use ONLY the provided context to answer the user's question.
Answer clearly without mentioning source numbers or page numbers.
Do not include any references in your answer.

Context:
{context}
"""

    messages = [SystemMessage(content=system_prompt)]
    for human, ai in st.session_state.doc_messages[-5:]:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=query))
    
    response = llm(messages).content
    
    # Simply return the answer without any source lines
    # Remove any accidental number references just in case
    answer = re.sub(r'\n?\s*\[(\d+(?:,\s*\d+)*)\]\s*$', '', response).strip()
    return answer

# -------------------- UI WITH TWO TABS --------------------
tab_doc, tab_normal = st.tabs(["📄 Document Chat (RAG with Sources)", "💬 Normal Chat (General AI)"])

# ==================== DOCUMENT CHAT TAB (UNCHANGED) ====================
with tab_doc:
    st.markdown("### 📚 Upload Multiple Documents (PDF/TXT)")
    
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True, key="doc_uploader")
    if uploaded_files:
        if st.button("🔄 Process All", key="process_docs"):
            with st.spinner("Processing documents (Groq Vision + Embeddings)..."):
                vs = process_uploaded_files(uploaded_files)
                if vs:
                    st.session_state.doc_vectorstore = vs
                    st.success(f"✅ {len(st.session_state.processed_files)} document(s) ready!")
                else:
                    st.error("No text could be extracted. Check your files.")
    
    if st.session_state.processed_files:
        st.info(f"📁 Active: {', '.join(st.session_state.processed_files)}")
    
    st.markdown("---")
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for i, (human, ai) in enumerate(st.session_state.doc_messages):
        message(ai, key=f"doc_ai_{i}")
        message(human, is_user=True, key=f"doc_user_{i}")
    
    # Input
    if st.session_state.doc_vectorstore is not None:
        if prompt := st.chat_input("Ask something about your documents..."):
            st.session_state.doc_messages.append((prompt, ""))
            with st.spinner("Retrieving relevant information..."):
                retrieved_docs = st.session_state.doc_vectorstore.similarity_search(prompt, k=5)
                answer = rag_answer(prompt, retrieved_docs)
            st.session_state.doc_messages[-1] = (prompt, answer)
            st.rerun()
    else:
        st.info("📂 Upload and process documents first, then ask questions.")

# ==================== NORMAL CHAT TAB (NEW) ====================
with tab_normal:
    st.markdown("### 💬 Chat with General AI Assistant")
    st.caption("Ask me anything")
    
    # Display normal chat history
    for i, (human, ai) in enumerate(st.session_state.normal_messages):
        message(ai, key=f"norm_ai_{i}")
        message(human, is_user=True, key=f"norm_user_{i}")
    
    # Normal chat input
    if prompt := st.chat_input("Ask a general question..."):
        st.session_state.normal_messages.append((prompt, ""))
        with st.spinner("Thinking..."):
            # Simple system prompt for general assistant
            system_msg = SystemMessage(content="You are a helpful, knowledgeable AI assistant. Answer clearly and concisely. You have broad knowledge about the world, science, technology, history, and everyday topics.")
            messages = [system_msg]
            for human, ai in st.session_state.normal_messages[:-1]:  # exclude the latest empty one
                messages.append(HumanMessage(content=human))
                messages.append(AIMessage(content=ai))
            messages.append(HumanMessage(content=prompt))
            response = llm(messages).content
        st.session_state.normal_messages[-1] = (prompt, response)
        st.rerun()