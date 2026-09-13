import streamlit as st
import os
import collections
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔑 SECURE METADATA ENVIRONMENTAL ROUTING
if "GEMINI_API_KEY" in st.secrets:
    API_KEY_STRING = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY_STRING = ""

# 1. Page Configuration & Setup
st.set_page_config(
    page_title="AI Operations & Supply Chain Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- PROFESSIONAL VISUAL INTERFACE UPGRADE (SAFE FOR ACTIONS) ---
st.markdown("""
    <style>
    /* Global Page Padding Reset */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Top Corporate Header Banner Layout */
    .header-banner {
        background: linear-gradient(135deg, #1A365D 0%, #2A4365 100%);
        padding: 24px; border-radius: 10px; margin-bottom: 28px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-banner h1 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 26px !important; margin: 0 0 6px 0 !important; }
    .header-banner p { color: #E2E8F0 !important; font-size: 14px !important; margin: 0 !important; opacity: 0.95; }
    .team-badge { font-weight: bold; color: #63B3ED !important; }
    
    /* Section Headers Design Layout */
    h2 { color: #2C5282 !important; font-weight: 600 !important; font-size: 19px !important; border-bottom: 2px solid #EDF2F7; padding-bottom: 6px; margin-bottom: 16px; }
    
    /* Grounded Output UX Containers */
    .premium-response { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #2563EB; border-radius: 8px; padding: 20px; margin-top: 14px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); color: #000000 !important; }
    .premium-citation { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 14px; margin-top: 8px; font-size: 13px; color: #475569; }
    </style>
""", unsafe_allow_html=True)

# Render the Executive Banner with your Team Identity Card
st.markdown("""
    <div class="header-banner">
        <h1>🤖 AI Operations & Supply Chain Knowledge Assistant</h1>
        <p>Engineered by Team: <span class="team-badge">AI-Catalysts- Hackathon</span> | Powered by Gemini 3.6 Flash & Local TF-IDF</p>
    </div>
""", unsafe_allow_html=True)

# 2. Initialize Session State Variables
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "keyword_frequencies" not in st.session_state:
    st.session_state.keyword_frequencies = {}

# 3. Sidebar Configuration & Live Frequency Monitor
st.sidebar.markdown("### ⚙️ Core Infrastructure")
st.sidebar.success("⚡ Gemini 3.6 Flash Engine: ACTIVE")
st.sidebar.info("📌 Knowledge Index Bank: LOCAL RAM")

# --- LIVE SIDEBAR METRICS PORTAL ---
if st.session_state.keyword_frequencies:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Document Word Frequencies")
    for doc_name, freq_list in st.session_state.keyword_frequencies.items():
        short_name = doc_name if len(doc_name) < 25 else f"{doc_name[:22]}..."
        with st.sidebar.expander(f"📁 {short_name}", expanded=True):
            for word, count in freq_list:
                st.caption(f"🔹 **{word.upper()}**: repeated {count} times")

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Corporate guardrails are live. Anti-hallucination tracking activated.")

# 4. Graphical Layout Panels
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("📄 Upload Knowledge Documents")
    uploaded_files = st.file_uploader(
        "Upload standard operational PDFs:",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    process_btn = st.button("⚙️ Process & Index Documents")

    if process_btn and uploaded_files:
        all_chunks = []
        all_sources = []
        extracted_freq_dict = {}
        
        STOP_WORDS = set([
            "the", "and", "a", "of", "to", "in", "is", "for", "that", "by", "on", "with", 
            "as", "an", "at", "be", "this", "from", "it", "are", "or", "was", "will", "shall",
            "must", "should", "under", "within", "strict", "exact", "any", "all", "each", "based"
        ])
        
        with st.spinner("Parsing operational files & analyzing key focus terms..."):
            for uploaded_file in uploaded_files:
                try:
                    reader = PdfReader(uploaded_file)
                    file_text = ""
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            file_text += text + "\n"
                    
                    lower_text = file_text.lower()
                    cleaned_text = "".join([c if c.isalnum() or c.isspace() else " " for c in lower_text])
                    
                    filtered_words = []
                    for word in cleaned_text.split():
                        if word not in STOP_WORDS and len(word) > 2 and not word.isdigit():
                            filtered_words.append(word)
                    
                    word_counts = collections.Counter(filtered_words)
                    extracted_freq_dict[uploaded_file.name] = word_counts.most_common(12)
                    
                    chunk_size = 500
                    words = file_text.split()
                    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                    
                    for idx, chunk in enumerate(chunks):
                        if chunk.strip():
                            all_chunks.append(chunk)
                            all_sources.append(f"{uploaded_file.name} (Segment {idx+1})")
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {e}")
            
            if all_chunks:
                st.session_state.chunks = all_chunks
                st.session_state.sources = all_sources
                st.session_state.keyword_frequencies = extracted_freq_dict
                
                vectorizer = TfidfVectorizer(stop_words='english')
                st.session_state.tfidf_matrix = vectorizer.fit_transform(all_chunks)
                st.session_state.vectorizer = vectorizer
                
                st.success(f"Successfully processed {len(uploaded_files)} document(s) into {len(all_chunks)} searchable knowledge nodes!")
                st.rerun()
            else:
                st.warning("No readable text could be retrieved from the uploaded documents.")

with col2:
    st.header("💬 Query Knowledge Base")
    user_query = st.text_input("Ask an operational or policy question:", placeholder="e.g., What is the maximum time cap for container clearance?")
    submit_query = st.button("🔍 Search Engine")
    
    if submit_query and user_query:
        if st.session_state.tfidf_matrix is None:
            st.error("Please upload and index documents on the left before running search queries.")
        else:
            with st.spinner("Scanning indexes and compiling grounded response..."):
                query_vec = st.session_state.vectorizer.transform([user_query])
                similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
                
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                context_str = ""
                matched_sources = []
                for idx in top_indices:
                    if similarities[idx] > 0.05:
                        context_str += f"Source: {st.session_state.sources[idx]}\nContent: {st.session_state.chunks[idx]}\n\n"
                        if st.session_state.sources[idx] not in matched_sources:
                            matched_sources.append(st.session_state.sources[idx])
                
                if not context_str.strip():
                    st.warning("No relevant document references matched your query parameters.")
                    context_str = "No reference text available."
                
                system_prompt = "You are an expert Operations and Supply Chain Knowledge Assistant.\nAnswer user questions accurately based ONLY on the operational text reference provided below.\nIf the answer cannot be confidently verified from the text, state exactly: \n'Information not found in the uploaded operational knowledge base.' Do not make up answers.\n\n--- START REFERENCE TEXT ---\n" + context_str + "\n--- END REFERENCE TEXT ---"
                
                try:
                    if not API_KEY_STRING:
                        st.error("🔒 Security Error: `GEMINI_API_KEY` is completely missing from your Streamlit Secrets console.")
                    else:
                        client = genai.Client(api_key=API_KEY_STRING)
                        config_setup = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0)
                        response = client.models.generate_content(model='gemini-3.6-flash', contents=user_query, config=config_setup)
                        
                        st.markdown("### 📝 Grounded Response")
                        st.markdown(f'<div class="premium-response">{response.text}</div>', unsafe_allow_html=True)
                        
                        if "Information not found" not in response.text and matched_sources:
                            st.write("")
                            st.markdown("#### 📌 Data Source Citations")
