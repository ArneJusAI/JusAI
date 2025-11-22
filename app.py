# app.py – JusAI (med MULTIPPEL DOKUMENTOPPLASTING i chat)
import streamlit as st
import requests
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Importer for dokumenthåndtering
try:
    import PyPDF2
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

# ---------------------------------------------------------
# STREAMLIT INNSTILLINGER
# ---------------------------------------------------------
st.set_page_config(
    page_title="JusAI – Jus For alle",
    page_icon="⚖️",
    layout="wide"  # Endret til wide for bedre plass
)

# Custom CSS for bedre utseende
st.markdown("""
<style>
    .uploaded-file {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .file-list {
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ JusAI")
st.caption("Norsk juridisk assistent med Claude AI og Lovdata-indeks.")

# ---------------------------------------------------------
# API-NØKKEL FRA SECRETS
# ---------------------------------------------------------
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------
# LAST INN LOVDATA-DATABASEN (RAG)
# ---------------------------------------------------------
@st.cache_resource
def load_rag_index():
    """Laster lovdata_index.pkl fra GitHub"""
    url = "https://github.com/ArneJusAI/JusAI/releases/download/V2.0/lovdata_index.pkl"
    local_path = "lovdata_index.pkl"

    try:
        if not os.path.exists(local_path):
            st.info("Laster ned lovdata_index.pkl...")
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)

        with open(local_path, "rb") as f:
            data = pickle.load(f)
        return data

    except Exception as e:
        st.warning(f"⚠️ Kunne ikke laste lovdata_index.pkl: {str(e)[:150]}")
        return []

indexed_data = load_rag_index()

# ---------------------------------------------------------
# EMBEDDING-MODELL
# ---------------------------------------------------------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str) -> np.ndarray:
    model = get_embed_model()
    vec = model.encode(query)
    return np.array(vec, dtype=float)

# ---------------------------------------------------------
# DOKUMENTHÅNDTERING
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file):
    """Henter ut tekst fra opplastede filer"""
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    try:
        # PDF
        if "pdf" in file_type or file_name.endswith('.pdf'):
            if not PDF_AVAILABLE:
                return "⚠️ PDF-støtte er ikke installert."
            
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        # DOCX
        elif "word" in file_type or file_name.endswith('.docx'):
            if not DOCX_AVAILABLE:
                return "⚠️ Word-støtte er ikke installert."
            
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        
        # TXT
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return text.strip()
    
    except Exception as e:
        return f"⚠️ Kunne ikke lese {uploaded_file.name}: {str(e)[:100]}"

# ---------------------------------------------------------
# RAG-FUNKSJONER
# ---------------------------------------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def search_lovdata(query: str, data, k: int = 8) -> list:
    if not data:
        return []

    query_vector = embed_query(query)
    q_lower = query.lower()
    
    results = []
    for chunk_text, chunk_emb, metadata in data:
        chunk_emb = np.array(chunk_emb, dtype=float)
        score = cosine_similarity(query_vector, chunk_emb)
        
        if q_lower in chunk_text.lower():
            score += 0.3
        
        results.append((chunk_text, score, metadata))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

def build_context(query: str, data) -> str:
    if not data:
        return "Ingen lovdata tilgjengelig."
    
    top_results = search_lovdata(query, data, k=8)
    context_parts = []
    
    for i, (chunk, score, meta) in enumerate(top_results):
        context_parts.append(f"--- Kilde {i+1} (relevans: {score:.2f}) ---\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    with st.expander("🔍 Lovdata-kilder brukt"):
        for i, (chunk, score, meta) in enumerate(top_results):
            st.write(f"**Treff {i+1}** – Relevans: {score:.2f}")
            st.caption(f"{meta}")
            st.text(chunk[:300] + "...")
            st.divider()
    
    return context

# ---------------------------------------------------------
# CLAUDE API
# ---------------------------------------------------------
def call_claude(prompt: str) -> str:
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 2500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"].strip()
        else:
            return f"⚠️ Feil fra Claude: {response.status_code}\n{response.text[:200]}"
    
    except Exception as e:
        return f"⚠️ Kunne ikke kontakte Claude: {str(e)[:200]}"

# ---------------------------------------------------------
# HOVEDFUNKSJON
# ---------------------------------------------------------
def get_legal_answer(query: str, documents: list = None):
    """Henter lovdata og spør Claude"""
    
    context = build_context(query, indexed_data)
    
    # Bygg dokumentseksjon hvis det finnes opplastede filer
    doc_section = ""
    if documents and len(documents) > 0:
        doc_section = "\n\n---\nBRUKERENS OPPLASTEDE DOKUMENTER:\n\n"
        for i, (name, content) in enumerate(documents):
            # Begrens lengde per dokument
            content_preview = content[:2500] if len(content) > 2500 else content
            doc_section += f"DOKUMENT {i+1}: {name}\n{content_preview}\n\n"
        doc_section += "---\n"
    
    system_prompt = f"""Du er JusAI – en norsk juridisk AI-assistent.

VIKTIG: Svar på NORSK, bruk "du" til brukeren, vær presis og pedagogisk.

Strukturer ALLTID svaret ditt slik:

## 🎯 Kort svar
[1-2 setninger: Hva er konklusjonen?]

## 📖 Juridisk grunnlag
[Hvilke lover/paragrafer gjelder? Vær konkret, f.eks. "Husleieloven § 10-3"]

## 💡 Forklaring
[Hvorfor er dette riktig? Forklar logikken enkelt.]

## ⚠️ Viktig å vite
[Eventuelle tidsfrister, unntak, eller forbehold]

## 📋 Neste steg
[Hva bør brukeren gjøre nå? Konkrete handlinger.]

---

Her er relevant lovdata:

{context}

{doc_section}

---

Brukerens spørsmål:
{query}

Gi et presist, forståelig svar basert på norsk rett."""
    
    answer = call_claude(system_prompt)
    return answer

# ---------------------------------------------------------
# SESSION STATE INITIALISERING
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Hei! Jeg er JusAI – din juridiske assistent. 👋

Jeg kan hjelpe deg med spørsmål om:
- 🏠 Husleie og depositum
- 👷 Arbeidsrett (oppsigelser, varsel)
- 🏘️ Naboforhold
- 📝 Kontrakter
- ⚖️ Forliksråd

💡 **Tips:** Du kan dra og slippe dokumenter (PDF, Word, TXT) direkte i chat-boksen nedenfor, eller bruke opplastningsknappen. Last opp ALLE dokumenter som er relevante for saken din!

Still gjerne spørsmålet ditt!"""
        }
    ]

# Lagre opplastede dokumenter
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# ---------------------------------------------------------
# LAYOUT: To kolonner
# ---------------------------------------------------------
col_chat, col_docs = st.columns([2, 1])

# ---------------------------------------------------------
# VENSTRE: CHAT
# ---------------------------------------------------------
with col_chat:
    # Vis tidligere meldinger
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Skriv ditt juridiske spørsmål her...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("⚖️ JusAI analyserer..."):
                answer = get_legal_answer(
                    user_input, 
                    documents=st.session_state.uploaded_documents
                )
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------
# HØYRE: DOKUMENTHÅNDTERING
# ---------------------------------------------------------
with col_docs:
    st.subheader("📁 Dokumenter i saken")
    
    # Opplastningsområde
    uploaded_files = st.file_uploader(
        "Dra og slipp filer her, eller klikk for å velge",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Du kan laste opp flere dokumenter samtidig. Alle vil bli brukt som kontekst."
    )
    
    if uploaded_files:
        new_docs_added = False
        
        for uploaded_file in uploaded_files:
            # Sjekk om filen allerede er lastet opp
            existing_names = [name for name, _ in st.session_state.uploaded_documents]
            
            if uploaded_file.name not in existing_names:
                with st.spinner(f"Leser {uploaded_file.name}..."):
                    text = extract_text_from_file(uploaded_file)
                    
                    if not text.startswith("⚠️"):
                        st.session_state.uploaded_documents.append(
                            (uploaded_file.name, text)
                        )
                        new_docs_added = True
                    else:
                        st.error(text)
        
        if new_docs_added:
            st.success("✅ Dokumenter lastet inn!")
            st.rerun()
    
    # Vis liste over opplastede dokumenter
    if st.session_state.uploaded_documents:
        st.markdown("### 📄 Aktive dokumenter:")
        
        for i, (name, content) in enumerate(st.session_state.uploaded_documents):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i+1}.** {name}")
            
            with col2:
                if st.button("👁️", key=f"view_{i}", help="Se innhold"):
                    with st.expander(f"Innhold: {name}", expanded=True):
                        st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
            
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="Fjern"):
                    st.session_state.uploaded_documents.pop(i)
                    st.rerun()
        
        # Mulighet til å analysere alle dokumenter
        if st.button("🔍 Analyser alle dokumenter", type="primary", use_container_width=True):
            with st.spinner("Analyserer dokumentene juridisk..."):
                all_docs_text = "\n\n---\n\n".join([
                    f"DOKUMENT: {name}\n\n{content[:3000]}" 
                    for name, content in st.session_state.uploaded_documents
                ])
                
                analysis_prompt = f"""Du er JusAI. Analyser disse dokumentene samlet som en juridisk sak:

{all_docs_text}

Gi en analyse strukturert slik:

## 📊 Oversikt over saken
[Hva handler dette om totalt sett?]

## 📄 Oppsummering av dokumentene
[Kort om hvert dokument og deres rolle]

## ⚠️ Juridiske risikoer
[Hva er problematisk?]

## ✅ Sterke punkter
[Hva støtter saken din?]

## 🎯 Anbefaling
[Hva bør gjøres nå?]"""
                
                analysis = call_claude(analysis_prompt)
                
                with st.chat_message("assistant"):
                    st.markdown("**📊 Analyse av alle dokumenter:**\n\n" + analysis)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**📊 Analyse av alle dokumenter:**\n\n{analysis}"
                })
        
        # Knapp for å fjerne alle
        if st.button("🗑️ Fjern alle dokumenter", use_container_width=True):
            st.session_state.uploaded_documents = []
            st.rerun()
    
    else:
        st.info("👆 Ingen dokumenter lastet opp ennå.\n\nLast opp alle dokumenter som er relevante for saken din.")
    
    # Tips-boks
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips
    
    **Last opp:**
    - Alle kontrakter
    - Brev/e-poster
    - Krav og varsler
    - Kvitteringer
    - Bilder (som PDF)
    
    **Så spør:**
    - "Analyser saken min"
    - "Hva er mine rettigheter?"
    - "Hva bør jeg gjøre?"
    """)

# ---------------------------------------------------------
# SIDEBAR: INFO
# ---------------------------------------------------------
with st.sidebar:
    st.header("Om JusAI")
    st.write("""
    JusAI bruker:
    - 🤖 **Claude AI** (Anthropic)
    - 📚 **Lovdata-indeks** (74 617 tekster)
    - 🔍 **RAG** (søker i lovverket)
    - 📄 **Multi-dokument analyse**
    
    Alle svar er basert på norsk lov og rettspraksis.
    """)
    
    st.divider()
    
    st.success("""
    ☕ Liker du JusAI?
    [Støtt prosjektet](https://buymeacoffee.com/jusai)
    """)
    
    st.divider()
    
    st.warning("""
    ⚠️ **Viktig:**
    JusAI gir juridisk veiledning, men erstatter ikke advokat ved komplekse saker.
    """)
    
    # Statistikk
    if st.session_state.uploaded_documents:
        st.divider()
        st.metric("📁 Dokumenter lastet", len(st.session_state.uploaded_documents))
        total_chars = sum(len(content) for _, content in st.session_state.uploaded_documents)
        st.metric("📝 Totalt tegn", f"{total_chars:,}")
