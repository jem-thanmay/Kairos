import streamlit as st
import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features
from src.predictor import predict
from src.ollama_guide import generate_guidance, test_ollama_connection
from src.tracker import save_observation, get_trajectory, list_people, load_history, delete_person
from src.conversation import get_response, get_opening_message, check_crisis_signal
from src.explainer import get_shap_explanation, plot_shap_bar

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
    .stTextInput input {
        font-family: 'Georgia', serif;
        font-size: 14px;
        background: #FFFFFF;
        border: 1px solid #DDE5DF;
        border-radius: 10px;
        color: #2C3E35;
        padding: 0.6rem 1rem;
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
    .trajectory-card {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        margin: 1rem 0;
        line-height: 1.8;
        border: 1px solid #E8EDE9;
        background: #FFFFFF;
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
    .chat-bubble-kairos {
        background: #F4F7F5;
        border-radius: 0 12px 12px 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 14px;
        color: #2C3E35;
        line-height: 1.8;
        max-width: 85%;
        font-style: italic;
    }
    .chat-bubble-user {
        background: #2C3E35;
        color: #FAFAF7;
        border-radius: 12px 0 12px 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 0.5rem auto;
        font-size: 14px;
        line-height: 1.8;
        max-width: 85%;
        text-align: right;
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

with st.expander("Before you begin — please read"):
    st.markdown("""
    <div style="font-size:13px; color:#5C6E65; line-height:1.9; font-family:Georgia,serif;">
    Kairos is not a clinical tool and does not replace professional mental health care.
    It never diagnoses. If you believe someone is in immediate danger, please reach out
    to emergency services or a crisis line right away.
    <br><br>
    <strong>USA — 988 Lifeline:</strong> Call or text 988 (24/7)<br>
    <strong>USA — Crisis Text Line:</strong> Text HOME to 741741<br>
    <strong>India — iCall:</strong> 9152987821<br>
    <strong>India — Vandrevala Foundation:</strong> 1860-2662-345 (24/7)<br>
    <strong>Worldwide:</strong> <a href="https://www.iasp.info/resources/Crisis_Centres/"
    style="color:#7A8C84;">iasp.info</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state
if 'obs_result' not in st.session_state:
    st.session_state.obs_result = None
if 'obs_context' not in st.session_state:
    st.session_state.obs_context = None
if 'obs_followup_history' not in st.session_state:
    st.session_state.obs_followup_history = []
if 'obs_input_key' not in st.session_state:
    st.session_state.obs_input_key = 0
if 'last_input_type' not in st.session_state:
    st.session_state.last_input_type = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False
if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = None
if 'talk_input_key' not in st.session_state:
    st.session_state.talk_input_key = 0

tab1, tab2, tab3 = st.tabs(["Observe", "Talk", "History"])

# ─────────────────────────────────────────
# TAB 1 — OBSERVE
# ─────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">This observation is about</div>',
        unsafe_allow_html=True)
    mode = st.radio(
        label="mode",
        options=["Someone I care about", "Myself"],
        label_visibility="collapsed",
        horizontal=True,
        key="obs_mode"
    )
    input_type = "observer" if mode == "Someone I care about" else "first_person"

    # Clear obs state when mode switches
    if st.session_state.get('last_input_type') != input_type:
        st.session_state.obs_result = None
        st.session_state.obs_context = None
        st.session_state.obs_followup_history = []
        st.session_state['last_input_type'] = input_type

    st.markdown("<br>", unsafe_allow_html=True)

    if input_type == "myself_mode":
        pass  # placeholder
    
    if input_type == "observer":
        st.markdown('<div class="section-label">Who is this about?</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px; color:#9AAB9F; margin-bottom:8px;">'
            'Use a nickname — stored only on your device.</div>',
            unsafe_allow_html=True)
        person_id = st.text_input(
            label="person",
            placeholder="e.g. mom, best friend, colleague",
            label_visibility="collapsed",
            key=f"obs_person_{st.session_state.obs_input_key}"
        )
        existing = list_people('observer')
        if existing:
            names = [p['person_id'] for p in existing]
            st.markdown(
                '<div style="font-size:12px; color:#9AAB9F; margin-top:4px;">Previously tracked: ' +
                ', '.join(names) + '</div>',
                unsafe_allow_html=True)
    else:
        person_id = "self"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MYSELF MODE — conversational vent space ──
    if input_type == "first_person":
        st.markdown("""
        <div style="font-size:14px; color:#7A8C84; font-family:Georgia,serif;
             font-style:italic; margin-bottom:1.2rem; line-height:1.8;">
        This is your space. Say whatever is on your mind — vent, think out loud,
        or just describe how you have been feeling. No judgement, no pressure.
        </div>
        """, unsafe_allow_html=True)

        if 'self_chat_history' not in st.session_state:
            st.session_state.self_chat_history = []
        if 'self_input_key' not in st.session_state:
            st.session_state.self_input_key = 0
        if 'self_full_text' not in st.session_state:
            st.session_state.self_full_text = ""

        # Show conversation so far
        for turn in st.session_state.self_chat_history:
            if turn['role'] == 'assistant':
                st.markdown(
                    f'<div class="chat-bubble-kairos">{turn["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-bubble-user">{turn["content"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Voice text injection
        if 'self_voice_text' not in st.session_state:
            st.session_state.self_voice_text = ""

        self_msg = st.text_area(
            label="self_input",
            placeholder="Say whatever comes to mind... or press 🎙 to speak",
            label_visibility="collapsed",
            height=100,
            value=st.session_state.self_voice_text,
            key=f"self_msg_{st.session_state.self_input_key}"
        )

        col1, col2, col3 = st.columns([3, 0.7, 0.7])
        with col1:
            self_send = st.button("Send  ↵", use_container_width=True, key="self_send")
        with col2:
            self_voice = st.button("🎙", use_container_width=True, key="self_voice")
        with col3:
            self_clear = st.button("↺", use_container_width=True, key="self_clear", help="Start over")

        if self_voice:
            from src.voice import record_and_transcribe
            with st.spinner("Listening... speak now (10 seconds)"):
                vresult = record_and_transcribe(timeout=10, phrase_limit=30)
            if vresult['success']:
                st.session_state.self_voice_text = vresult['text']
                st.session_state.self_input_key += 1
                st.rerun()
            else:
                st.warning(vresult['error'])
                st.session_state.self_voice_text = ""

        if self_clear:
            st.session_state.self_chat_history = []
            st.session_state.self_full_text = ""
            st.session_state.self_input_key += 1
            st.session_state.self_voice_text = ""
            st.session_state.obs_result = None
            st.rerun()

        if self_send and self_msg.strip():
            st.session_state.self_voice_text = ""
            # Accumulate full text for analysis
            st.session_state.self_full_text += " " + self_msg

            # Get conversational response
            with st.spinner(""):
                from src.conversation import get_response
                result = get_response(
                    history=st.session_state.self_chat_history,
                    new_message=self_msg,
                    stage=None
                )

            st.session_state.self_chat_history.append(
                {'role': 'user', 'content': self_msg}
            )
            if result['success']:
                st.session_state.self_chat_history.append(
                    {'role': 'assistant', 'content': result['response']}
                )
            else:
                st.session_state.self_chat_history.append(
                    {'role': 'assistant',
                     'content': "I hear you. Take your time — what else is on your mind?"}
                )

            st.session_state.self_input_key += 1
            st.rerun()

        # Optional analysis button
        if st.session_state.self_full_text.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:12px; color:#9AAB9F; font-family:Georgia,serif;'
                'font-style:italic;">When you are ready, Kairos can read the patterns in what you have shared.</div>',
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Analyse what I have shared", use_container_width=True, key="self_analyse"):
                with st.spinner("Reading the patterns..."):
                    features = extract_features(st.session_state.self_full_text)
                    result = predict(features, input_type='first_person')
                st.session_state.obs_result = result
                st.session_state.obs_context = {
                    'person_id': 'self',
                    'relationship': None,
                    'input_type': 'first_person',
                    'stage': result['stage'],
                    'text': st.session_state.self_full_text
                }
                save_observation(
                    person_id='self',
                    text=st.session_state.self_full_text,
                    stage=result['stage'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    input_type='first_person'
                )
                st.rerun()

        # Show analysis if available for self
        if st.session_state.obs_result and st.session_state.obs_context and            st.session_state.obs_context.get('input_type') == 'first_person':

            result = st.session_state.obs_result
            stage = result['stage']
            confidence = result['confidence']
            probs = result['probabilities']
            static_guidance = result['guidance']
            features = extract_features(st.session_state.obs_context.get('text', ''))

            st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

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

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stress", f"{probs.get('stress', 0)*100:.0f}%")
            with col2:
                st.metric("Depression", f"{probs.get('depression', 0)*100:.0f}%")
            with col3:
                st.metric("Crisis", f"{probs.get('crisis', 0)*100:.0f}%")

            with st.expander("Why did Kairos read it this way?"):
                try:
                    explanation = get_shap_explanation(features, stage)
                    img = plot_shap_bar(explanation, stage)
                    st.image(img, use_container_width=True)
                except:
                    pass

            st.markdown('<div class="section-label">A few thoughts for you</div>',
                unsafe_allow_html=True)

            ollama_available = test_ollama_connection()
            if ollama_available:
                with st.spinner(""):
                    ollama_result = generate_guidance(
                        user_input=st.session_state.obs_context.get('text', ''),
                        stage=stage,
                        confidence=confidence,
                        probabilities=probs,
                        input_type='first_person',
                        relationship=None,
                        key_signals=features
                    )
                if ollama_result['success']:
                    st.markdown('<div class="ollama-badge">✦ personalised guidance</div>',
                        unsafe_allow_html=True)
                    st.markdown("**Something that might help**")
                    st.markdown(f'<div class="guidance-card">{ollama_result["what_to_say"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("**A gentle reminder**")
                    st.markdown(f'<div class="avoid-card">{ollama_result["what_not_to_do"]}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown("**Something that might help**")
                    st.markdown(f'<div class="guidance-card">{static_guidance["what_to_say"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("**A gentle reminder**")
                    st.markdown(f'<div class="avoid-card">{static_guidance["what_not_to_do"]}</div>',
                        unsafe_allow_html=True)

            if stage == 'crisis' and confidence >= 0.65:
                st.markdown(f"""
                <div class="crisis-card">
                    <div style="font-size:14px; font-weight:600; color:#8B2E2E;
                         margin-bottom:0.8rem;">You do not have to carry this alone.</div>
                    <div style="font-size:13px; color:#6B3E3E; line-height:1.9;">
                        What you are feeling is real. Please consider reaching out.<br><br>
                        <strong>USA — 988 Lifeline:</strong> Call or text 988 (24/7)<br>
                        <strong>USA — Crisis Text Line:</strong> Text HOME to 741741<br>
                        <strong>India — iCall:</strong> 9152987821<br>
                        <strong>India — Vandrevala Foundation:</strong> 1860-2662-345 (24/7)
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Stop here for first_person mode — don't fall through to observer flow
    # ── OBSERVER MODE ──
    if input_type == "observer":
        text_input = st.text_area(
            label="input",
            placeholder="Describe what you have observed in plain language. For example: She has not been eating properly. She laughed it off but sounded really flat.",
            height=180,
            label_visibility="collapsed",
            key=f"obs_text_{input_type}_{st.session_state.obs_input_key}"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Your relationship to this person</div>',
            unsafe_allow_html=True)
        relationship = st.selectbox(
            label="relationship",
            options=["Select...", "Parent", "Child / Adult child",
                     "Partner / Spouse", "Friend", "Sibling", "Colleague", "Other"],
            label_visibility="collapsed",
            key=f"obs_rel_{st.session_state.obs_input_key}"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        analyse = st.button("Read the patterns", use_container_width=True, key="obs_btn")

        if analyse:
            if not text_input.strip():
                st.warning("Please write something before continuing.")
            elif not person_id.strip():
                st.warning("Please enter a name or label for this person.")
            elif relationship == "Select...":
                st.warning("Please select your relationship to this person.")
            else:
                with st.spinner("Reading the patterns..."):
                    features = extract_features(text_input)
                    result = predict(features, input_type=input_type)

                stage = result['stage']
                confidence = result['confidence']
                probs = result['probabilities']
                static_guidance = result['guidance']

                save_observation(
                    person_id=person_id.strip(),
                    text=text_input,
                    stage=stage,
                    confidence=confidence,
                    probabilities=probs,
                    relationship=relationship,
                    input_type=input_type
                )

                # Increment key to clear text area next render
                st.session_state.obs_input_key += 1

                # Store result in session state for follow-up
                st.session_state.obs_result = result
                st.session_state.obs_context = {
                    'person_id': person_id.strip(),
                    'relationship': relationship,
                    'input_type': input_type,
                    'stage': stage,
                    'text': text_input
                }
                st.session_state.obs_followup_history = []

                trajectory = get_trajectory(person_id.strip())

                st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

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

                # SHAP explanation
                with st.expander("Why did Kairos read it this way?"):
                    try:
                        explanation = get_shap_explanation(features, stage)
                        img = plot_shap_bar(explanation, stage)
                        st.markdown(
                            '<div style="font-size:13px; color:#7A8C84; font-family:Georgia,serif;'
                            'font-style:italic; margin-bottom:0.8rem; line-height:1.7;">'
                            'These are the signals that most influenced the prediction. '
                            'Colored bars pushed toward the detected stage. '
                            'Grey bars pulled away from it.</div>',
                            unsafe_allow_html=True
                        )
                        st.image(img, use_container_width=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        top3 = explanation["top_features"][:3]
                        for f in top3:
                            label = f["label"]
                            direction = "detected, pushing toward " + stage if f["shap_value"] > 0 else "not strongly present"
                            st.markdown(
                                f'<div style="font-size:12px; color:#5C6E65; margin-bottom:4px; font-family:Georgia,serif;">' +
                                f'· <strong>{label}</strong> — {direction}</div>',
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.markdown(
                            '<div style="font-size:12px; color:#9AAB9F;">Explanation unavailable.</div>',
                            unsafe_allow_html=True
                        )

                if trajectory['has_trajectory']:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Trajectory</div>',
                        unsafe_allow_html=True)
                    trend = trajectory['trend']
                    trend_color = trajectory['trend_color']
                    n = trajectory['total_observations']
                    stages = trajectory['recent_stages']
                    trend_icons = {'worsening': '↗', 'improving': '↘', 'stable': '→'}
                    trend_labels = {
                        'worsening': 'Signal is intensifying',
                        'improving': 'Signal is easing',
                        'stable': 'Signal is holding steady'
                    }
                    st.markdown(f"""
                    <div class="trajectory-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:13px; font-weight:600; color:{trend_color};">
                                    {trend_icons[trend]} {trend_labels[trend]}
                                </span>
                                <div style="font-size:12px; color:#7A8C84; margin-top:4px;">
                                    Based on {n} observation{'s' if n > 1 else ''}
                                </div>
                            </div>
                            <div style="font-size:12px; color:#9AAB9F; text-align:right;">
                                {' → '.join(stages)}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if trajectory.get('stage_changed', False):
                    prev = trajectory.get('previous_stage', '')
                    curr = trajectory.get('current_stage', '')
                    stage_order = ['stress', 'depression', 'crisis']
                    if stage_order.index(curr) > stage_order.index(prev):
                        st.markdown(f"""
                        <div style="background:#FDF4F4; border-left:3px solid #C97A7A;
                             padding:0.8rem 1.2rem; border-radius:0 8px 8px 0;
                             font-size:13px; color:#6B1E1E; margin-bottom:1rem;">
                            Pattern has shifted from <strong>{prev}</strong> to
                            <strong>{curr}</strong>. This change warrants attention.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#F0F7F4; border-left:3px solid #7AB89A;
                             padding:0.8rem 1.2rem; border-radius:0 8px 8px 0;
                             font-size:13px; color:#1E6B3E; margin-bottom:1rem;">
                            Pattern has shifted from <strong>{prev}</strong> to
                            <strong>{curr}</strong>. This is a positive sign.
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

            if input_type == 'observer':
                label = f"How to respond · {relationship}" if relationship and relationship != "Select..." else "How to respond"
            else:
                label = "A few thoughts for you"

            st.markdown(f'<div class="section-label">{label}</div>',
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            ollama_available = test_ollama_connection()

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
                    st.markdown("**What tends to help**" if input_type == 'observer' else "**Something that might help**")
                    st.markdown(f'<div class="guidance-card">{ollama_result["what_to_say"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**What to avoid**" if input_type == 'observer' else "**A gentle reminder**")
                    st.markdown(f'<div class="avoid-card">{ollama_result["what_not_to_do"]}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown("**What tends to help**")
                    st.markdown(f'<div class="guidance-card">{static_guidance["what_to_say"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**What to avoid**")
                    st.markdown(f'<div class="avoid-card">{static_guidance["what_not_to_do"]}</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown("**What tends to help**")
                st.markdown(f'<div class="guidance-card">{static_guidance["what_to_say"]}</div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**What to avoid**")
                st.markdown(f'<div class="avoid-card">{static_guidance["what_not_to_do"]}</div>',
                    unsafe_allow_html=True)

            if stage == 'crisis' and confidence >= 0.65:
                st.markdown("<br>", unsafe_allow_html=True)
                crisis_msg = "This level of signal warrants closer attention." if input_type == 'observer' else "You do not have to carry this alone."
                crisis_body = (
                    "If this person has expressed thoughts of self-harm, please consider reaching out to a crisis line."
                    if input_type == 'observer' else
                    "What you are feeling is real. Please consider reaching out to someone you trust or a crisis line."
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

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="disclaimer-card">
                Kairos surfaces patterns, not diagnoses. Use these signals as a prompt
                for your own judgment, not a replacement for it.
            </div>
            """, unsafe_allow_html=True)

    # ── Observer follow-up conversation ──
    ctx_check = st.session_state.obs_context or {}
    if st.session_state.obs_result and ctx_check.get('input_type') == 'observer':

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Continue the conversation</div>',
            unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:13px; color:#7A8C84; font-family:Georgia,serif;
             font-style:italic; margin-bottom:1rem; line-height:1.7;">
        Ask Kairos anything about this situation. What should I say tonight?
        How do I bring this up without making it worse?
        </div>
        """, unsafe_allow_html=True)

        # Show follow-up history
        for turn in st.session_state.obs_followup_history:
            if turn['role'] == 'assistant':
                st.markdown(
                    f'<div class="chat-bubble-kairos">{turn["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-bubble-user">{turn["content"]}</div>',
                    unsafe_allow_html=True)

        # Follow-up input with auto-clear
        followup_key = f"obs_followup_{st.session_state.get('followup_counter', 0)}"
        followup_msg = st.text_input(
            label="followup",
            placeholder="Ask anything about this situation...",
            label_visibility="collapsed",
            key=followup_key
        )

        if st.button("Ask", use_container_width=True, key="obs_followup_btn"):
            if followup_msg.strip():
                ctx = st.session_state.obs_context

                # Build context-aware prompt
                context_prompt = (
                    f"You are Kairos helping an observer support their {ctx.get('relationship', 'loved one')}. "
                    f"The observer described: \"{ctx.get('text', '')[:100]}\". "
                    f"The detected pattern is {ctx.get('stage', 'stress')}-stage. "
                    f"Answer their follow-up question as a warm, specific advisor. "
                    f"Keep it under 80 words. No therapy-speak."
                )

                history_with_context = [
                    {'role': 'assistant', 'content': context_prompt}
                ] + st.session_state.obs_followup_history

                with st.spinner(""):
                    result = get_response(
                        history=history_with_context,
                        new_message=followup_msg,
                        stage=ctx.get('stage')
                    )

                st.session_state.obs_followup_history.append(
                    {'role': 'user', 'content': followup_msg}
                )
                if result['success']:
                    st.session_state.obs_followup_history.append(
                        {'role': 'assistant', 'content': result['response']}
                    )
                else:
                    st.session_state.obs_followup_history.append(
                        {'role': 'assistant',
                         'content': "I am here. What would you like to know?"}
                    )

                # Increment counter to clear input
                st.session_state['followup_counter'] = \
                    st.session_state.get('followup_counter', 0) + 1
                st.rerun()

    st.markdown("""
    <div class="kairos-footer">
        Built with care by
        <a href="https://jem-thanmay.vercel.app">Thanmay Jembige</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/jem-thanmay/Kairos">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 3 — HISTORY
# ─────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:14px; color:#7A8C84; font-family:Georgia,serif;'
        'font-style:italic; margin-bottom:1.5rem; line-height:1.8;">'
        'A record of everyone you have been tracking. All data is stored only on your device.'
        '</div>',
        unsafe_allow_html=True
    )

    stage_colors = {
        'stress': '#C9A87A',
        'depression': '#7B72D4',
        'crisis': '#C97A7A'
    }
    stage_icons = {
        'stress': '🟡',
        'depression': '🟣',
        'crisis': '🔴'
    }

    # Toggle between observer and self history
    history_mode = st.radio(
        label="history_mode",
        options=["People I am tracking", "My own journey"],
        horizontal=True,
        label_visibility="collapsed",
        key="history_mode"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if history_mode == "People I am tracking":
        people = list_people('observer')
        if not people:
            st.markdown(
                '<div style="font-size:14px; color:#9AAB9F; font-family:Georgia,serif;'
                'font-style:italic; text-align:center; margin-top:3rem;">'
                'No observations logged yet. Start in the Observe tab.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            selected_person = st.selectbox(
                label="Select person",
                options=[p["person_id"] for p in people],
                label_visibility="collapsed",
                key="history_person"
            )
    else:
        # Self history
        self_history = load_history('self')
        if not self_history or not self_history.get('observations'):
            st.markdown(
                '<div style="font-size:14px; color:#9AAB9F; font-family:Georgia,serif;'
                'font-style:italic; text-align:center; margin-top:3rem;">'
                'No personal observations logged yet. Use Myself mode in the Observe tab.'
                '</div>',
                unsafe_allow_html=True
            )
            selected_person = None
        else:
            selected_person = 'self'

    if history_mode == "People I am tracking":
        show_history = bool(people)
    else:
        show_history = selected_person == 'self'

    if show_history and selected_person:

        history = load_history(selected_person)
        trajectory = get_trajectory(selected_person)
        obs_list = history.get("observations", [])

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary card
        rel = history.get("relationship", "")
        display_name = "Your journey" if selected_person == "self" else selected_person.capitalize()
        st.markdown(
            f'<div style="font-size:13px; color:#7A8C84; font-family:Georgia,serif;'
            f'margin-bottom:1rem;">{display_name}'
            f'{" · " + rel if rel and rel != "Select..." else ""}'
            f' · {len(obs_list)} observation{"s" if len(obs_list) != 1 else ""}</div>',
            unsafe_allow_html=True
        )

        # Trajectory summary
        if trajectory["has_trajectory"]:
                trend = trajectory["trend"]
                trend_color = trajectory["trend_color"]
                trend_icons = {"worsening": "↗", "improving": "↘", "stable": "→"}
                trend_labels = {
                    "worsening": "Signal is intensifying",
                    "improving": "Signal is easing",
                    "stable": "Signal is holding steady"
                }
                st.markdown(
                    f'<div style="font-size:13px; font-weight:600; color:{trend_color}; margin-bottom:1.2rem;">' +
                    f'{trend_icons[trend]} {trend_labels[trend]}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<hr style="border:none; border-top:1px solid #E8EDE9; margin:0.5rem 0 1.2rem 0;">', unsafe_allow_html=True)

        # Observation list — most recent first
        for obs in reversed(obs_list):
            stage = obs["stage"]
            color = stage_colors.get(stage, "#7A8C84")
            icon = stage_icons.get(stage, "⚪")
            date = obs.get("date", "")
            text = obs.get("text", "")
            probs = obs.get("probabilities", {})

            st.markdown(
                f'<div style="border-left:3px solid {color}; padding:0.8rem 1.2rem;' +
                'background:#FFFFFF; border-radius:0 10px 10px 0; margin-bottom:1rem;">' +
                f'<div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">' +
                f'<span style="font-size:12px; font-weight:600; color:{color}; text-transform:uppercase; letter-spacing:0.08em;">{icon} {stage}</span>' +
                f'<span style="font-size:11px; color:#9AAB9F;">{date}</span></div>' +
                f'<div style="font-size:13px; color:#3D4F46; font-family:Georgia,serif; font-style:italic; line-height:1.7; margin-bottom:0.5rem;">&ldquo;{text[:200]}{"..." if len(text) > 200 else ""}&rdquo;</div>' +
                f'<div style="font-size:11px; color:#9AAB9F;">' +
                f'Stress {probs.get("stress",0)*100:.0f}% · ' +
                f'Depression {probs.get("depression",0)*100:.0f}% · ' +
                f'Crisis {probs.get("crisis",0)*100:.0f}%</div>' +
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Delete option
        delete_label = "Delete my personal history" if selected_person == "self" else f"Delete all records for {selected_person}"
        if st.button(delete_label, key="history_delete"):
            delete_person(selected_person)
            st.success("Records deleted.")
            st.rerun()

    st.markdown("""
    <div class="kairos-footer">
        Built with care by
        <a href="https://jem-thanmay.vercel.app">Thanmay Jembige</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/jem-thanmay/Kairos">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 2 — TALK
# ─────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:14px; color:#7A8C84; font-family:Georgia,serif;
         font-style:italic; margin-bottom:1.5rem; line-height:1.8;">
    This is a quiet space to think out loud. Kairos will listen and ask
    one gentle question at a time. Nothing here is stored or shared.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_started:
        st.markdown('<div class="section-label">How are you feeling right now?</div>',
            unsafe_allow_html=True)
        mood = st.selectbox(
            label="mood",
            options=["I am not sure", "Stressed and overwhelmed",
                     "Low and withdrawn", "Really struggling"],
            label_visibility="collapsed",
            key="talk_mood"
        )
        mood_to_stage = {
            "I am not sure": None,
            "Stressed and overwhelmed": "stress",
            "Low and withdrawn": "depression",
            "Really struggling": "crisis"
        }
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Begin", use_container_width=True, key="talk_start"):
            st.session_state.chat_stage = mood_to_stage[mood]
            st.session_state.chat_started = True
            opening = get_opening_message(st.session_state.chat_stage)
            st.session_state.chat_history = [
                {'role': 'assistant', 'content': opening}
            ]
            st.session_state.talk_input_key = 0
            st.rerun()

    else:
        for turn in st.session_state.chat_history:
            if turn['role'] == 'assistant':
                st.markdown(
                    f'<div class="chat-bubble-kairos">{turn["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-bubble-user">{turn["content"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Voice transcription into input
        if 'voice_text' not in st.session_state:
            st.session_state.voice_text = ""

        # Auto-clearing input using key counter
        input_value = st.session_state.voice_text or ""
        user_msg = st.text_area(
            label="message",
            placeholder="Say whatever comes to mind... or press 🎙 to speak",
            label_visibility="collapsed",
            height=80,
            value=input_value,
            key=f"talk_msg_{st.session_state.talk_input_key}"
        )

        col1, col2, col3 = st.columns([3, 0.7, 0.7])
        with col1:
            send = st.button("Send  ↵", use_container_width=True, key="talk_send")
        with col2:
            voice_btn = st.button("🎙", use_container_width=True, key="talk_voice",
                help="Click to speak")
        with col3:
            if st.button("End", use_container_width=True, key="talk_end"):
                st.session_state.chat_history = []
                st.session_state.chat_started = False
                st.session_state.chat_stage = None
                st.session_state.talk_input_key = 0
                st.session_state.voice_text = ""
                st.rerun()

        if voice_btn:
            from src.voice import record_and_transcribe
            with st.spinner("Listening... speak now"):
                result = record_and_transcribe(timeout=5, phrase_limit=30)
            if result['success']:
                st.session_state.voice_text = result['text']
                st.session_state.talk_input_key += 1
                st.rerun()
            else:
                st.warning(result['error'])
                st.session_state.voice_text = ""

        # Clear voice text after it's been loaded into input
        if st.session_state.voice_text and user_msg == st.session_state.voice_text:
            pass  # keep it until send
        elif not st.session_state.voice_text:
            pass

        if send and user_msg.strip():
            st.session_state.voice_text = ""
            is_crisis = check_crisis_signal(user_msg)

            st.session_state.chat_history.append(
                {'role': 'user', 'content': user_msg}
            )

            with st.spinner(""):
                result = get_response(
                    history=st.session_state.chat_history[:-1],
                    new_message=user_msg,
                    stage=st.session_state.chat_stage
                )

            if result['success']:
                response = result['response']
                if is_crisis:
                    response += "\n\nIf things feel urgent right now — please reach out. USA: call or text 988. India: 9152987821 (iCall)."
                st.session_state.chat_history.append(
                    {'role': 'assistant', 'content': response}
                )
            else:
                st.session_state.chat_history.append(
                    {'role': 'assistant',
                     'content': "I am here. Take your time — what would you like to share?"}
                )

            # Increment key to clear input
            st.session_state.voice_text = ""
            st.session_state.talk_input_key += 1
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="disclaimer-card">
            Kairos listens but does not replace professional support.
            If you are in crisis, please reach out to a trained person.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="kairos-footer">
            Built with care by
            <a href="https://jem-thanmay.vercel.app">Thanmay Jembige</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/jem-thanmay/Kairos">GitHub</a>
        </div>
        """, unsafe_allow_html=True)
