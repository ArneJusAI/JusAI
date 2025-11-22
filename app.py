# app.py – JusAI (FORENKLET med Claude + DOKUMENTANALYSE)
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
    layout="centered"
)

st.title("⚖️ JusAI")
st.caption("Norsk juridisk assistent med Claude AI og Lovdata-indeks.")

# ---------------------------------------------------------
# API-NØKKEL FRA SECRETS
# ---------------------------------------------------------
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]

# Claude-modell (dette er den beste for jus)
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
# EMBEDDING-MODELL (for å søke i lovdata)
# ---------------------------------------------------------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str) -> np.ndarray:
    """Lager en vektor av spørsmålet for søk"""
    model = get_embed_model()
    vec = model.encode(query)
    return np.array(vec, dtype=float)

# ---------------------------------------------------------
# DOKUMENTHÅNDTERING (PDF, DOCX, TXT)
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file):
    """Henter ut tekst fra opplastede filer"""
    
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    try:
        # PDF
        if "pdf" in file_type or file_name.endswith('.pdf'):
            if not PDF_AVAILABLE:
                return "⚠️ PDF-støtte er ikke installert. Last opp som TXT i stedet."
            
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        # DOCX (Word)
        elif "word" in file_type or file_name.endswith('.docx'):
            if not DOCX_AVAILABLE:
                return "⚠️ Word-støtte er ikke installert. Last opp som TXT i stedet."
            
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        
        # TXT
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return text.strip()
    
    except Exception as e:
        return f"⚠️ Kunne ikke lese filen: {str(e)[:200]}"

# ---------------------------------------------------------
# SØKEFUNKSJONER (RAG)
# ---------------------------------------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Beregner hvor like to vektorer er (0-1)"""
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def search_lovdata(query: str, data, k: int = 8) -> list:
    """Søker etter de mest relevante lovtekstene"""
    if not data:
        return []

    query_vector = embed_query(query)
    q_lower = query.lower()
    
    results = []
    for chunk_text, chunk_emb, metadata in data:
        chunk_emb = np.array(chunk_emb, dtype=float)
        score = cosine_similarity(query_vector, chunk_emb)
        
        # Gi ekstra poeng hvis søkeordet finnes i teksten
        if q_lower in chunk_text.lower():
            score += 0.3
        
        results.append((chunk_text, score, metadata))
    
    # Sorter etter beste treff
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

def build_context(query: str, data) -> str:
    """Bygger kontekst fra lovdata til Claude"""
    if not data:
        return "Ingen lovdata tilgjengelig - bruk generell juridisk kunnskap."
    
    top_results = search_lovdata(query, data, k=8)
    
    context_parts = []
    for i, (chunk, score, meta) in enumerate(top_results):
        context_parts.append(f"--- Kilde {i+1} (relevans: {score:.2f}) ---\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # Vis hva som ble funnet (for transparens)
    with st.expander("🔍 Lovdata-kilder brukt (klikk for detaljer)"):
        for i, (chunk, score, meta) in enumerate(top_results):
            st.write(f"**Treff {i+1}** – Relevans: {score:.2f}")
            st.caption(f"{meta}")
            st.text(chunk[:300] + "...")
            st.divider()
    
    return context

# ---------------------------------------------------------
# CLAUDE API-KALL
# ---------------------------------------------------------
def call_claude(prompt: str) -> str:
    """Sender spørsmål til Claude og får svar"""
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
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
# DOKUMENTANALYSE
# ---------------------------------------------------------
def analyze_document(document_text: str, doc_name: str = "dokumentet") -> str:
    """Analyserer et juridisk dokument"""
    
    # Begrens teksten (for å ikke overskride token-limit)
    if len(document_text) > 8000:
        document_text = document_text[:8000] + "\n\n[... resten av dokumentet er kuttet for lengde ...]"
    
    analysis_prompt = f"""Du er JusAI – en norsk juridisk AI-assistent som analyserer dokumenter.

Analyser følgende dokument nøye og gi en strukturert vurdering:

---
DOKUMENT ({doc_name}):

{document_text}
---

Gi en analyse strukturert slik:

## 📄 Type dokument
[Hva er dette? Kontrakt, avtale, varsel, krav, leiekontrakt, etc.]

## ⚠️ Identifiserte risikoer
[Finn problematiske klausuler, formuleringer eller mangler. Vær konkret.]

## ✅ Positive elementer
[Hva er juridisk OK eller godt i dokumentet?]

## 📝 Anbefalte endringer
[Konkrete forslag til forbedringer, med begrunnelse]

## ⚖️ Samlet vurdering
[På en skala 1-10, hvor 10 er perfekt juridisk dokument. Gi også en overordnet anbefaling.]

Vær pedagogisk, presis, og referer til relevante norske lover der det passer."""

    return call_claude(analysis_prompt)

# ---------------------------------------------------------
# HOVEDFUNKSJON: SØK I LOVDATA + SPØR CLAUDE
# ---------------------------------------------------------
def get_legal_answer(query: str, document_context: str = None):
    """Henter lovdata og spør Claude om svaret"""
    
    # 1. Finn relevante lovtekster
    context = build_context(query, indexed_data)
    
    # 2. Legg til dokumentkontekst hvis det finnes
    doc_section = ""
    if document_context:
        doc_section = f"""

---
BRUKERENS OPPLASTEDE DOKUMENT:

{document_context[:3000]}
---

Når du svarer, ta hensyn til dette dokumentet der det er relevant.
"""
    
    # 3. Lag en god prompt til Claude
    system_prompt = f"""Du er JusAI – en norsk juridisk AI-assistent.

VIKTIG: Du skal svare på NORSK, bruk "du" til brukeren, vær presis og pedagogisk.

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

Her er relevant lovdata jeg har funnet:

{context}

{doc_section}

---

Brukerens spørsmål:
{query}

Gi et presist, forståelig svar basert på norsk rett."""
    
    # 4. Spør Claude
    answer = call_claude(system_prompt)
    
    return answer

# ---------------------------------------------------------
# CHAT-GRENSESNITT
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

Du kan også **laste opp dokumenter** (PDF, Word, TXT) i sidebaren for analyse!

Still gjerne spørsmålet ditt!"""
        }
    ]

# For å lagre opplastet dokument
if "uploaded_document" not in st.session_state:
    st.session_state.uploaded_document = None
    st.session_state.document_name = None

# Vis tidligere meldinger
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input fra bruker
user_input = st.chat_input("Skriv ditt juridiske spørsmål her...")

if user_input:
    # Legg til brukerens melding
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Få svar fra JusAI (Claude + RAG + dokument hvis relevant)
    with st.chat_message("assistant"):
        with st.spinner("⚖️ JusAI tenker..."):
            answer = get_legal_answer(
                user_input, 
                document_context=st.session_state.uploaded_document
            )
            st.markdown(answer)
    
    # Lagre svaret
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------
# SIDEBAR MED DOKUMENTOPPLASTING
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Last opp dokument")
    
    uploaded_file = st.file_uploader(
        "Last opp PDF, Word eller TXT",
        type=["pdf", "docx", "doc", "txt"],
        help="Dokumentet vil bli analysert og brukt som kontekst i samtalen"
    )
    
    if uploaded_file is not None:
        if st.button("🔍 Analyser dokument", type="primary"):
            with st.spinner("Leser dokumentet..."):
                # Hent ut tekst
                doc_text = extract_text_from_file(uploaded_file)
                
                if doc_text.startswith("⚠️"):
                    st.error(doc_text)
                else:
                    # Lagre i session state
                    st.session_state.uploaded_document = doc_text
                    st.session_state.document_name = uploaded_file.name
                    
                    st.success(f"✅ {uploaded_file.name} er lastet inn!")
                    
                    # Vis preview
                    with st.expander("👀 Se innhold (første 500 tegn)"):
                        st.text(doc_text[:500] + "...")
                    
                    # Kjør automatisk analyse
                    with st.spinner("Analyserer juridisk innhold..."):
                        analysis = analyze_document(doc_text, uploaded_file.name)
                        
                        # Vis analyse
                        st.markdown("### 📊 Analyse")
                        st.markdown(analysis)
                        
                        # Legg analyse i chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**Jeg har analysert {uploaded_file.name}:**\n\n{analysis}"
                        })
    
    # Vis hvilket dokument som er aktivt
    if st.session_state.uploaded_document:
        st.info(f"📎 Aktivt dokument: **{st.session_state.document_name}**")
        if st.button("🗑️ Fjern dokument"):
            st.session_state.uploaded_document = None
            st.session_state.document_name = None
            st.rerun()
    
    st.divider()
    
    st.header("Om JusAI")
    st.write("""
    JusAI bruker:
    - 🤖 **Claude AI** (Anthropic)
    - 📚 **Lovdata-indeks** (74 617 tekster)
    - 🔍 **RAG** (søker i lovverket)
    - 📄 **Dokumentanalyse**
    
    Alle svar er basert på norsk lov og rettspraksis.
    """)
    
    st.divider()
    
    st.info("""
    💡 **Tips:**
    - Vær konkret i spørsmålet
    - Nevn beløp hvis relevant
    - Fortell hva som har skjedd
    - Last opp kontrakter for analyse
    """)
    
    st.divider()
    
    st.warning("""
    ⚠️ **Viktig:**
    JusAI gir juridisk veiledning, men erstatter ikke advokat ved komplekse saker.
    """)
    
    st.divider()
    
    st.success("""
    ☕ Liker du JusAI?
    [Støtt prosjektet](https://buymeacoffee.com/jusai)
    """)
