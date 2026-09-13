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

# --- PREMIUM LUXURY BRAND CSS INJECTION MATRIX ---
st.markdown("""
    <style>
    /* Global Page Structure and Typography */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 94%; }
    body { font-family: 'Inter', -apple-system, sans-serif !important; }
    
    /* Obsidian Executive Header Banner */
    .premium-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 30px; border-radius: 16px; margin-bottom: 32px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .premium-banner h1 { color: #FFFFFF !important; font-weight: 800 !important; font-size: 28px !important; margin: 0 0 6px 0 !important; letter-spacing: -0.5px; text-align: center !important; }
    .premium-banner .banner-sub { color: #94A3B8 !important; font-size: 14px !important; margin: 0 0 20px 0 !important; font-weight: 400; text-align: center !important; }
    
    /* Neon Team Badge */
    .team-badge { background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: 0.5px; }
    
    /* Roster Two-Line Layout Container Modules */
    .roster-box { background: rgba(30, 41, 59, 0.5); padding: 16px 20px; border-radius: 10px; border: 1px solid #334155; max-width: 800px; margin: 0 auto; }
    .roster-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center; }
    .roster-row:first-child { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(51, 65, 85, 0.5); }
    .roster-label { color: #94A3B8; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Leader Gold Shield Badge */
    .badge-leader { background: linear-gradient(135deg, #FEF08A 0%, #EAB308 100%); color: #451A03 !important; font-weight: 700; padding: 4px 14px; border-radius: 6px; font-size: 13px; box-shadow: 0 2px 4px rgba(234, 179, 8, 0.2); }
    
    /* Member Emerald Ice Badges */
    .badge-member { background: linear-gradient(135deg, #D1FAE5 0%, #10B981 100%); color: #064E3B !important; font-weight: 600; padding: 4px 12px; border-radius: 6px; font-size: 12px; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.15); display: inline-block; }
    
    /* Interactive Column Layout Headers */
    h2 { color: #1E3A8A !important; font-weight: 700 !important; font-size: 20px !important; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 20px; }
    
    /* Grounded Output Display Frame */
    .premium-response { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #2563EB; border-radius: 8px; padding: 20px; margin-top: 14px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    .premium-citation { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 14px; margin-top: 8px; font-size: 13px; color: #475569; }
    </style>
""", unsafe_allow_html=True)

# Render Centered Obsidian Executive Header Banner with Two-Line Roster
st.markdown("""
    <div class="premium-banner">
        <h1>🤖 AI Operations & Supply Chain Knowledge Assistant</h1>
        <div class="banner-sub">Engineered by Team: <span class="team-badge">AI-Catalysts- Hackathon</span></div>
        <div class="roster-box">
            <div class="roster-row">
                <span class="roster-label">👑 Project Leader:</span>
                <span class="badge-leader">Hafiz Masood Ur Rehman</span>
            </div>
            <div class="roster-row">
                <span class="roster-label">👥 Team Members:</span>
                <span class="badge-member">Fatima Ijaz</span>
                <span class="badge-member">Muhammad Aslam</span>
                <span class="badge-member">Shakeel Ahmed</span>
                <span class="badge-member">Sami Ur Rahman</span>
                <span class="badge-member">Muhammad Haroon Jan</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. Initialize Session State Variables Permanently
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "sources" not in st.session_state:
    st.session_state.sources = None
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

# --- DECOUPLED FLATTENED RAG ROUTING CORE ---
def run_search_pipeline(query_text):
    if st.session_state.tfidf_matrix is None:
        st.error("Please upload and index documents on the left before running search queries.")
        return
    if not API_KEY_STRING:
        st.error("🔒 Security Error: `GEMINI_API_KEY` is completely missing from your Streamlit Secrets vault console.")
        return
        
    with st.spinner("Scanning matrix indexes and compiling response context..."):
        query_vec = st.session_state.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        context_str = ""
        matched_sources = []
        for idx in top_indices:
            if similarities[idx] > 0.05:
                context_str += f"Source: {st.session_state.sources[idx]}\nContent: {st.session_state.chunks[idx]}\n\n"
                current_source = st.session_state.sources[idx]
                if current_source not in matched_sources:
                    matched_sources.append(current_source)
                    
        if not context_str.strip():
            st.warning("No relevant document references matched your query parameters.")
            context_str = "No reference text available."
            
        system_prompt = "You are an expert Operations and Supply Chain Knowledge Assistant.\nAnswer user questions accurately based ONLY on the operational text reference provided below.\nIf the answer cannot be confidently verified from the text, state exactly: \n'Information not found in the uploaded operational knowledge base.' Do not make up answers.\n\n--- START REFERENCE TEXT ---\n" + context_str + "\n--- END REFERENCE TEXT ---"
        
        try:
            client = genai.Client(api_key=API_KEY_STRING)
            config_setup = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0)
            response = client.models.generate_content(model='gemini-3.6-flash', contents=query_text, config=config_setup)
            
            st.markdown("### 📝 Grounded Response")
            st.markdown(f'<div class="premium-response">{response.text}</div>', unsafe_allow_html=True)
            
            if "Information not found" not in response.text and matched_sources:
                st.write("")
                st.markdown("#### 📌 Data Source Citations")
                for source in matched_sources:
                    st.markdown(f'<div class="premium-citation">✔ Verified Reference: <code>{source}</code></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Google Gemini Engine Exception: {e}")

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
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        file_text += page_text + "\n"
                    
