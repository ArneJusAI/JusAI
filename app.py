# app.py – JusAI  (konsensus + RAG med lokal embedding)
import streamlit as st
import requests
import pickle
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# STREAMLIT SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="JusAI – Jus For alle",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ JusAI")
st.caption("Norsk juridisk assistent med konsensus og RAG (Lovdata-indeks).")

# ---------------------------------------------------------
# API-KEYS FRA SECRETS
# ---------------------------------------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROK_API_KEY   = st.secrets["GROK_API_KEY"]

# Modellnavn – kan endres hvis du vil bruke andre modeller
GEMINI_TEXT_MODEL = "gemini-2.0-flash"     # f.eks. gemini-1.5-flash / gemini-1.5-pro / gemini-2.0-flash
GROK_MODEL        = "grok-2-latest"        # juster til det som står i xAI-dashboardet om nødvendig

# ---------------------------------------------------------
# LAST INN RAG-DATABASE (lovdata_index.pkl) FRA GITHUB RELEASE
# ---------------------------------------------------------
@st.cache_resource
def load_rag_index():
    """
    Laster lovdata_index.pkl.
    Hvis den ikke finnes lokalt, lastes den ned fra GitHub Releases.
    """
    # NB: endre v2.0 her hvis releasen din har en annen tag
    url = "url = "https://github.com/ArneJusAI/JusAI/releases/download/V2.0/lovdata_index.pkl"
"
    local_path = "lovdata_index.pkl"

    try:
        if not os.path.exists(local_path):
            st.info("Laster ned lovdata_index.pkl fra GitHub ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)

        with open(local_path, "rb") as f:
            data = pickle.load(f)

        return data

    except Exception as e:
        st.warning(f"⚠️ Klarte ikke å laste lovdata_index.pkl – bruker kun generell kunnskap. ({str(e)[:150]})")
        return []

indexed_data = load_rag_index()

# ---------------------------------------------------------
# LOKAL EMBEDDING-MODELL (samme som brukt til å bygge indeksen)
# ---------------------------------------------------------
@st.cache_resource
def get_embed_model():
    # Må være samme modell som i build_lovdata_index.py
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str) -> np.ndarray:
    model = get_embed_model()
    vec = model.encode(query)
    return np.array(vec, dtype=float)

# ---------------------------------------------------------
# RAG HJELPEFUNKSJONER
# ---------------------------------------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def retrieve_top_k(query_embedding: np.ndarray, data, k: int = 10, query_text: str = ""):
    """
    Returnerer de k mest relevante tekstbitene fra lovdata_index.
    Liten ekstra boost hvis query-teksten finnes direkte i chunken.
    """
    if not data:
        return []

    similarities = []
    q_lower = query_text.lower() if query_text else ""

    for chunk_text, chunk_emb, metadata in data:
        chunk_emb = np.array(chunk_emb, dtype=float)
        score = cosine_similarity(query_embedding, chunk_emb)

        # Boost eksakte teksttreff (f.eks. ved lovtittel eller paragrafnavn)
        if q_lower and q_lower in chunk_text.lower():
            score += 0.5

        similarities.append((chunk_text, score, metadata))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def hybrid_rag_context(query: str, data) -> str:
    """Bygger en kontekststreng fra RAG + en prompt om generell kunnskap."""
    if data:
        query_emb = embed_query(query)
        top_chunks = retrieve_top_k(query_emb, data, k=10, query_text=query)
        context_parts = [chunk for chunk, _, _ in top_chunks]
        context = "\n\n".join(context_parts)

        # Debug-visning: hva RAG faktisk fant
        with st.expander("🔍 RAG-treff (debug)", expanded=False):
            for i, (chunk, score, meta) in enumerate(top_chunks):
                st.write(f"Treff {i+1} – score {score:.3f}")
                st.write(meta)
                st.write(chunk[:400] + "…")
                st.write("---")
    else:
        context = ""

    context += (
        "\n\n[Generell kunnskap]: Du kan også bruke din treningskunnskap om norsk rett, "
        "Høyesterettspraksis og juridisk teori for å supplere dersom RAG-konteksten ikke dekker alt."
    )
    return context.strip()

# ---------------------------------------------------------
# KALL TIL MODELLENE
# ---------------------------------------------------------
def call_gemini(prompt: str) -> str:
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{GEMINI_TEXT_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        r = requests.post(url, json=payload, timeout=45)
        if r.status_code == 200:
            data = r.json()
            # typisk struktur: candidates[0].content.parts[0].text
            return data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        else:
            return f"[Gemini-feil {r.status_code}] {r.text[:200]}"
    except Exception as e:
        return f"[Gemini-exception] {str(e)[:200]}"

def call_grok(prompt: str) -> str:
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROK_MODEL,
            "messages": [
                {"role": "system", "content": "Du er JusAI, en norsk juridisk assistent."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"[Grok-feil {r.status_code}] {r.text[:200]}"
    except Exception as e:
        return f"[Grok-exception] {str(e)[:200]}"

# ---------------------------------------------------------
# KONSENSUSLOGIKK
# ---------------------------------------------------------
def konsensus_svar(query: str):
    """Kjører RAG, spør Gemini og Grok og lager et felles svar."""
    context = hybrid_rag_context(query, indexed_data)

    hovedprompt = f"""
Du er JusAI – en norsk AI-jurist.

Oppgave:
- Svar kort og klart på norsk.
- Bruk "du" til brukeren.
- Vær presis, og skill mellom lovtekst, forarbeider, praksis og skjønn.
- Hvis relevant: henvis til konkrete paragrafer (f.eks. naboloven § 3, avtaleloven § 36) og Høyesterettsdommer (HR-åååå-xxx).

Kontekst (fra lovdata_index/RAG, kan være ufullstendig):
{context}

Brukerens spørsmål:
{query}

Gi et strukturert svar med:
1. Kort konklusjon først
2. Begrunnelse med henvisninger
3. Eventuelle praktiske råd / neste steg
"""

    svarer = {}

    gemini_svar = call_gemini(hovedprompt)
    svarer["Gemini"] = gemini_svar
    time.sleep(0.5)

    grok_svar = call_grok(hovedprompt)
    svarer["Grok"] = grok_svar

    # Lag et samlet "metasvar" hos Gemini basert på begge
    meta_prompt = f"""
Du er en norsk AI-jurist som skal lage et kort felles svar basert på to andre AI-svar.

Brukerens spørsmål:
{query}

Svar fra Gemini:
{gemini_svar}

Svar fra Grok:
{grok_svar}

Oppgave:
- Lag ett samlet svar til brukeren (ca. 2–4 avsnitt).
- Hent ut det som er felles og juridisk mest presist.
- Hvis det er uenighet, forklar kort, men gi likevel en klar anbefaling.
- Skriv på norsk, med konklusjon først.
"""
    final = call_gemini(meta_prompt)
    konsensus_tekst = "Felles svar generert basert på Gemini og Grok."

    # Kilder – kan senere hentes ekte fra metadata i RAG
    kilder = []
    if indexed_data:
        kilder.append("Utdrag fra lovdata_index.pkl (GitHub Release, lokal RAG)")
    kilder.append("Generell norsk juridisk teori og rettspraksis")

    return final, kilder, konsensus_tekst, svarer

# ---------------------------------------------------------
# CHAT-STATE I STREAMLIT
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "content": "Hei! Still meg et juridisk spørsmål (for eksempel: 'Kan naboen kreve at jeg feller treet mitt?')."
        }
    ]

# Vis tidligere meldinger
for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg["role"] == "ai" else "user"):
        st.markdown(msg["content"])

# Inputboks nederst
prompt = st.chat_input("Skriv spørsmålet ditt her...")

if prompt:
    # Legg til brukerens melding
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Svar fra JusAI
    with st.chat_message("assistant"):
        with st.spinner("JusAI analyserer spørsmålet ditt..."):
            final, kilder, konsensus_tekst, alle_svar = konsensus_svar(prompt)

            st.markdown(final)

            with st.expander("Kilder / bakgrunn"):
                for k in kilder:
                    st.write(f"• {k}")

            with st.expander("Detaljerte svar fra hver modell"):
                for navn, svar in alle_svar.items():
                    st.markdown(f"### {navn}")
                    st.markdown(svar)

            st.info(f"**Konsensus:** {konsensus_tekst}")

    # Lagre AI-svaret i historikken
    st.session_state.messages.append({"role": "ai", "content": final})

