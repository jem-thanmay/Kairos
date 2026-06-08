import streamlit as st
import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features
from src.predictor import predict
from src.ollama_guide import generate_guidance, test_ollama_connection

st.set_page_config(
    page_title="Kairos",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        background-color: #FAFAF7;
    }
    .stApp { background-color: #FAFAF7; }
    .block-container {
        max-width: 680px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    .kairos-header { text-align: center; margin-bottom: 0.5rem; }
    .kairos-title {
        font-size: 2.2rem;
        font-weight: 400;
        color: #2C3E35;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .kairos-subtitle {
        font-size: 1rem;
        color: #7A8C84;
        font-style: italic;
        margin-top: 0.3rem;
    }
    .soft-divider {
        border: none;
        border-top: 1px solid #E8EDE9;
        margin: 1.8rem 0;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7A8C84;
        margin-bottom: 0.6rem;
    }
    .stTextArea textarea {
        font-family: 'Georgia', serif;
        font-size: 15px;
        background: #FFFFFF;
        border: 1px solid #DDE5DF;
        border-radius: 10px;
        color: #2C3E35;
        line-height: 1.7;
        padding: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #7A8C84;
        box-shadow: 0 0 0 2px #E8EDE9;
    }
    .stRadio > div { gap: 0.6rem; }
    .stRadio label {
        font-family: 'Georgia', serif;
        font-size: 14px;
        color: #2C3E35;
    }
    .stSelectbox > div > div {
        background: #FFFFFF;
        border: 1px solid #DDE5DF;
        border-radius: 10px;
        font-family: 'Georgia', serif;
        font-size: 14px;
        color: #2C3E35;
    }
    .stButton > button {
        background: #2C3E35;
        color: #FAFAF7;
        border: none;
        border-radius: 10px;
        font-family: 'Georgia', serif;
        font-size: 15px;
        padding: 0.7rem 2rem;
        letter-spacing: 0.04em;
        transition: background 0.2s;
        width: 100%;
    }
    .stButton > button:hover { background: #3D5247; }
    .stage-card {
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .guidance-card {
        background: #F4F7F5;
        border-left: 3px solid #7A8C84;
        padding: 1.1rem 1.4rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0;
        font-size: 14px;
        color: #3D4F46;
        line-height: 1.8;
        font-style: italic;
    }
    .avoid-card {
        background: #FAF7F4;
        border-left: 3px solid #C9A87A;
        padding: 1.1rem 1.4rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0;
        font-size: 14px;
        color: #5C4A35;
        line-height: 1.8;
    }
    .crisis-card {
        background: #FDF4F4;
        border: 1px solid #E8C5C5;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        line-height: 1.8;
    }
    .disclaimer-card {
        background: #F4F7F5;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 13px;
        color: #7A8C84;
        line-height: 1.7;
        margin-top: 0.5rem;
    }
    .ollama-badge {
        display: inline-block;
        font-size: 11px;
        color: #7A8C84;
        background: #F0F4F1;
        border-radius: 6px;
        padding: 2px 8px;
        margin-bottom: 0.8rem;
        font-family: monospace;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8EDE9;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Georgia', serif;
        font-size: 12px;
        color: #7A8C84;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Georgia', serif;
        font-size: 1.6rem;
        color: #2C3E35;
    }
    .kairos-footer {
        text-align: center;
        font-size: 12px;
        color: #A8B8B0;
        margin-top: 3rem;
        line-height: 1.8;
    }
    .kairos-footer a { color: #7A8C84; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="kairos-header">
    <div class="kairos-title">🌿 Kairos</div>
    <div class="kairos-subtitle">A quiet tool for the people who notice.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

# Disclaimer
with st.expander("Before you begin — please read"):
    st.markdown("""
    <div style="font-size:13px; color:#5C6E65; line-height:1.9; font-family:Georgia,serif;">
    Kairos is not a clinical tool and does not replace professional mental health care.
    It surfaces patterns in language to help you be more present for someone you care about.
    It never diagnoses. If you believe someone is in immediate danger, please reach out to
    emergency services or a crisis line right away.
    <br><br>
    <strong>USA — 988 Suicide and Crisis Lifeline:</strong> Call or text 988 (24/7)<br>
    <strong>USA — Crisis Text Line:</strong> Text HOME to 741741<br>
    <strong>India — iCall:</strong> 9152987821<br>
    <strong>India — Vandrevala Foundation:</strong> 1860-2662-345 (24/7)<br>
    <strong>Worldwide:</strong> <a href="https://www.iasp.info/resources/Crisis_Centres/"
    style="color:#7A8C84;">iasp.info</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Mode
st.markdown('<div class="section-label">This observation is about</div>',
    unsafe_allow_html=True)
mode = st.radio(
    label="mode",
    options=["Someone I care about", "Myself"],
    label_visibility="collapsed",
    horizontal=True
)
input_type = "observer" if mode == "Someone I care about" else "first_person"

st.markdown("<br>", unsafe_allow_html=True)

# Input
if input_type == "observer":
    st.markdown('<div class="section-label">What have you noticed?</div>',
        unsafe_allow_html=True)
    placeholder = (
        "Describe what you have observed in plain language. "
        "No need for clinical terms — just what you have seen or heard. "
        "For example: She has not been eating properly. "
        "She laughed it off but sounded really flat. "
        "She keeps saying she is fine but I can hear how exhausted she is."
    )
else:
    st.markdown('<div class="section-label">How have you been feeling?</div>',
        unsafe_allow_html=True)
    placeholder = (
        "There is no right or wrong way to write this. "
        "Just say what is true for you right now. "
        "For example: I do not see the point anymore. "
        "I am exhausted and nothing I do seems to matter."
    )

text_input = st.text_area(
    label="input",
    placeholder=placeholder,
    height=180,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# Relationship
relationship = None
if input_type == "observer":
    st.markdown('<div class="section-label">Your relationship to this person</div>',
        unsafe_allow_html=True)
    relationship = st.selectbox(
        label="relationship",
        options=[
            "Select...",
            "Parent",
            "Child / Adult child",
            "Partner / Spouse",
            "Friend",
            "Sibling",
            "Colleague",
            "Other"
        ],
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Button
analyse = st.button("Read the patterns", use_container_width=True)

# Results
if analyse:
    if not text_input.strip():
        st.warning("Please write something before continuing.")
    elif input_type == "observer" and relationship == "Select...":
        st.warning("Please select your relationship to this person.")
    else:
        with st.spinner("Reading the patterns..."):
            features = extract_features(text_input)
            result = predict(features, input_type=input_type)

        stage = result['stage']
        confidence = result['confidence']
        probs = result['probabilities']
        static_guidance = result['guidance']

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        # Stage card
        stage_colors = {
            'stress':     {'bg': '#FDF8F0', 'border': '#C9A87A', 'text': '#6B4F1E'},
            'depression': {'bg': '#F2F1FB', 'border': '#7B72D4', 'text': '#3B3280'},
            'crisis':     {'bg': '#FDF4F4', 'border': '#C97A7A', 'text': '#6B1E1E'},
        }
        c = stage_colors[stage]

        st.markdown(f"""
        <div class="stage-card" style="background:{c['bg']};
             border-left: 4px solid {c['border']};">
            <div style="font-size:13px; font-weight:600; letter-spacing:0.1em;
                 text-transform:uppercase; color:{c['border']}; margin-bottom:0.5rem;">
                {static_guidance['label']}
            </div>
            <div style="font-size:15px; color:{c['text']}; font-family:Georgia,serif;">
                {static_guidance['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Signal bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Signal distribution</div>',
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stress", f"{probs.get('stress', 0)*100:.0f}%")
        with col2:
            st.metric("Depression", f"{probs.get('depression', 0)*100:.0f}%")
        with col3:
            st.metric("Crisis", f"{probs.get('crisis', 0)*100:.0f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        # Ollama guidance
        ollama_available = test_ollama_connection()

        if input_type == 'observer':
            if relationship and relationship != "Select...":
                st.markdown(
                    f'<div class="section-label">How to respond · {relationship}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<div class="section-label">How to respond</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label">A few thoughts for you</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if ollama_available:
            with st.spinner("Generating personalised guidance..."):
                ollama_result = generate_guidance(
                    user_input=text_input,
                    stage=stage,
                    confidence=confidence,
                    probabilities=probs,
                    input_type=input_type,
                    relationship=relationship,
                    key_signals=features
                )

            if ollama_result['success']:
                st.markdown('<div class="ollama-badge">✦ personalised guidance</div>',
                    unsafe_allow_html=True)

                if input_type == 'observer':
                    st.markdown("**What tends to help — for you to do**")
                else:
                    st.markdown("**Something that might help**")

                st.markdown(
                    f'<div class="guidance-card">{ollama_result["what_to_say"]}</div>',
                    unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if input_type == 'observer':
                    st.markdown("**What to avoid**")
                else:
                    st.markdown("**A gentle reminder**")

                st.markdown(
                    f'<div class="avoid-card">{ollama_result["what_not_to_do"]}</div>',
                    unsafe_allow_html=True)

            else:
                # Fallback to static
                st.markdown("**What tends to help**")
                st.markdown(
                    f'<div class="guidance-card">{static_guidance["what_to_say"]}</div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**What to avoid**")
                st.markdown(
                    f'<div class="avoid-card">{static_guidance["what_not_to_do"]}</div>',
                    unsafe_allow_html=True)
        else:
            # Ollama not running — use static
            st.markdown("**What tends to help**")
            st.markdown(
                f'<div class="guidance-card">{static_guidance["what_to_say"]}</div>',
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**What to avoid**")
            st.markdown(
                f'<div class="avoid-card">{static_guidance["what_not_to_do"]}</div>',
                unsafe_allow_html=True)

        # Crisis alert
        if stage == 'crisis' and confidence >= 0.65:
            st.markdown("<br>", unsafe_allow_html=True)
            if input_type == 'observer':
                crisis_msg = "This level of signal warrants closer attention."
                crisis_body = (
                    "If this person has expressed thoughts of self-harm or not wanting "
                    "to be here, please consider reaching out to a professional or crisis line. "
                    "You do not have to navigate this alone either."
                )
            else:
                crisis_msg = "You do not have to carry this alone."
                crisis_body = (
                    "What you are feeling is real and it is serious. "
                    "Please consider reaching out to someone — a person you trust, "
                    "or a crisis line where someone will listen without judgment. "
                    "You reached out here. That same instinct can take you one step further."
                )

            st.markdown(f"""
            <div class="crisis-card">
                <div style="font-size:14px; font-weight:600; color:#8B2E2E;
                     margin-bottom:0.8rem;">{crisis_msg}</div>
                <div style="font-size:13px; color:#6B3E3E; line-height:1.9;">
                    {crisis_body}<br><br>
                    <strong>USA — 988 Lifeline:</strong> Call or text 988 (24/7)<br>
                    <strong>USA — Crisis Text Line:</strong> Text HOME to 741741<br>
                    <strong>India — iCall:</strong> 9152987821<br>
                    <strong>India — Vandrevala Foundation:</strong> 1860-2662-345 (24/7)<br>
                    <strong>Worldwide:</strong>
                    <a href="https://www.iasp.info/resources/Crisis_Centres/"
                    style="color:#8B2E2E;">iasp.info</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="disclaimer-card">
            Kairos surfaces patterns, not diagnoses. These signals reflect linguistic
            patterns in what you wrote — they are not clinical assessments.
            Use them as a prompt for your own judgment, not a replacement for it.
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div class="kairos-footer">
            Built with care by
            <a href="https://jem-thanmay.vercel.app">Thanmay Jembige</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/jem-thanmay/Kairos">GitHub</a>
        </div>
        """, unsafe_allow_html=True)
