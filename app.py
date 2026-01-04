"""
Streamlit Application for AI Hallucination Detection System
Dark UI – Text only – Full results visible
"""

import streamlit as st
import requests
from typing import Dict, Optional

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Content Verification",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Hide Streamlit defaults
# -------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Dark UI CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 3rem 1rem;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(45,212,191,0.15);
    color: #2dd4bf;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

h1 {
    font-size: 3rem;
    font-weight: 800;
    color: white;
}

h1 span {
    color: #2dd4bf;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 1rem;
    max-width: 750px;
}

.card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.section-title {
    color: white;
    font-weight: 600;
    margin-bottom: 1rem;
}

.stat-card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2dd4bf;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.9rem;
}

.issue {
    border-left: 4px solid;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.issue.high { border-color: #ef4444; background: rgba(239,68,68,0.08); }
.issue.medium { border-color: #facc15; background: rgba(250,204,21,0.08); }
.issue.low { border-color: #22c55e; background: rgba(34,197,94,0.08); }

.verify-btn button {
    background: #2dd4bf !important;
    color: #020617 !important;
    font-weight: 700;
    border-radius: 12px !important;
    height: 48px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# API
# -------------------------------------------------
API_URL = "http://localhost:8000"

def verify_text(text: str) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_URL}/verify",
            json={
                "text": text,
                "verify_citations": True,
                "verify_facts": True
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown('<div class="badge">🛡 AI Content Verification</div>', unsafe_allow_html=True)

st.markdown("""
<h1>Verify <span>AI-Generated Content</span></h1>
<p class="subtitle">
Paste AI-generated text below to detect hallucinations, fake citations,
contradicted facts, and get a detailed trust report.
</p>
""", unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Input Text</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "",
    height=220,
    placeholder="Paste AI-generated content here...",
    label_visibility="collapsed"
)

if st.button("Verify Content", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text to verify.")
    else:
        with st.spinner("Analyzing text, verifying claims and citations..."):
            result = verify_text(text_input)
            st.session_state["result"] = result

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Results
# -------------------------------------------------
if "result" in st.session_state and st.session_state["result"]:
    data = st.session_state["result"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Overall Risk</div>', unsafe_allow_html=True)

    st.success(
        f"Risk: {data['overall_risk'].upper()} "
        f"({data['risk_score']:.1f}/100)"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Statistics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    stats = [
        ("Total Claims", data["total_claims"]),
        ("Total Citations", data["total_citations"]),
        ("Verified Claims", data["verified_claims"]),
        ("Fake Citations", data["fake_citations"]),
        ("Unverified Claims", data["unverified_claims"]),
        ("Contradicted", data["contradicted_claims"]),
    ]

    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], stats):
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Issues
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detected Issues</div>', unsafe_allow_html=True)

    if data["issues"]:
        for issue in data["issues"]:
            st.markdown(f"""
                <div class="issue {issue['severity']}">
                    <strong>{issue['type'].replace('_',' ').title()}</strong><br>
                    {issue['detail']}<br>
                    <em>Recommendation:</em> {issue.get('recommendation','')}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No issues detected. Content appears reliable.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed results
    with st.expander("View Detailed Results (Raw)"):
        st.json(data["detailed_results"])

st.markdown('</div>', unsafe_allow_html=True)
