import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from data_engine import (
    load_data,
    build_context,
    build_system_prompt,
    get_groq_client,
    build_verification_facts,
    verify_answer,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Sales AI Analyst",
    page_icon="💊",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_data(DATA_DIR)

@st.cache_data(show_spinner="Building analytics context...")
def get_context():
    return build_context(get_data())

@st.cache_data(show_spinner=False)
def get_verification_facts():
    return build_verification_facts(get_data())

@st.cache_resource
def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    try:
        return get_groq_client(api_key)
    except ValueError:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pill.png", width=60)
    st.title("Pharma Analyst AI")
    st.caption("Powered by Groq · llama-3.3-70b")
    st.divider()

    try:
        data = get_data()
        st.markdown("**Dataset**")
        col1, col2 = st.columns(2)
        col1.metric("Territories", len(data["territory_dim"]))
        col1.metric("Reps",        len(data["rep_dim"]))
        col1.metric("HCPs",        len(data["hcp_dim"]))
        col2.metric("Accounts",    len(data["account_dim"]))
        col2.metric("Rx Records",  f"{len(data['fact_rx']):,}")
        col2.metric("Activities",  f"{len(data['fact_rep_act']):,}")
    except Exception as e:
        st.error(f"Data load error: {e}")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Ask about reps, territories, brands, HCPs, payor mix, or market share.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("💊 Pharma Sales AI Analyst")

client = get_client()
if client is None:
    st.error(
        "**GROQ_API_KEY not found.**  \n"
        "Set it as an environment variable and restart:\n"
        "```\nset GROQ_API_KEY=your_key_here\nstreamlit run app.py\n```"
    )
    st.stop()

try:
    context = get_context()
    system_prompt = build_system_prompt(context)
except Exception as e:
    st.error(f"Failed to build analytics context: {e}")
    st.stop()

# ── Init session ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTIONS = [
    "Which rep has the lowest Tier A HCP coverage?",
    "Is NRx growing or declining quarter over quarter?",
    "Which territory is the most efficient (TRx per HCP)?",
    "Which accounts are most exposed to Medicare?",
    "Are there any Tier A HCPs who have never been called?",
    "Which brand has the highest new patient acquisition rate?",
    "Which rep has the lowest call completion rate?",
    "What is our overall market share trend?",
]

if not st.session_state.messages:
    st.markdown("### What would you like to know?")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": s})
            st.rerun()
    st.divider()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "💊"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("disclaimer"):
                st.caption(f"*{msg['disclaimer']}*")
            if msg.get("source_sections"):
                with st.expander("📂 Data sections used", expanded=False):
                    st.caption(msg["source_sections"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about the pharma sales data..."):
    if not prompt.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💊"):
        placeholder = st.empty()
        full_response = ""
        source_sections = ""
        disclaimer = None

        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}] + history +
                         [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            # Verification — silent if verified, one line only if flagged
            disclaimer = verify_answer(full_response, get_verification_facts())
            if disclaimer:
                st.caption(f"*{disclaimer}*")

            section_keywords = {
                "REP SCORECARD":       "Rep Scorecard",
                "TERRITORY SCORECARD": "Territory Scorecard",
                "BRANDS":              "Brand TRx/NRx Totals",
                "QUARTERLY RX TREND":  "Quarterly Rx Trend",
                "HCP CALL COVERAGE":   "HCP Call Coverage",
                "FLAGGED INSIGHTS":    "Flagged Insights & Alerts",
                "PAYOR MIX":           "Payor Mix",
                "MARKET SHARE":        "Market Share (LN Metrics)",
                "HCP TIER":            "HCP Tier Breakdown",
                "ACTIVITY TYPE":       "Activity Type Mix",
            }
            used = [
                label for kw, label in section_keywords.items()
                if kw.lower() in full_response.lower()
            ]
            source_sections = ", ".join(used) if used else ""

            if source_sections:
                with st.expander("📂 Data sections used", expanded=False):
                    st.caption(source_sections)

        except Exception as e:
            full_response = (
                "I'm sorry, I ran into an issue processing your question. "
                "Please try rephrasing or ask something else."
            )
            placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "source_sections": source_sections,
        "disclaimer": disclaimer,
    })
