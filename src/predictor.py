import joblib
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE, 'models', 'cascade_classifier.joblib'))
scaler = joblib.load(os.path.join(BASE, 'models', 'scaler.joblib'))
le = joblib.load(os.path.join(BASE, 'models', 'label_encoder.joblib'))

FEATURE_COLS = [
    'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
    'isolation_score', 'hopelessness_score', 'burnout_score', 'self_blame_score',
    'first_person_ratio', 'question_count', 'negation_ratio',
    'word_count', 'avg_sentence_length', 'exclamation_ratio'
]

STAGE_DESCRIPTIONS = {
    'stress': {
        'label': 'Stress',
        'color': '#e0a25a',
        'description': 'The patterns you described suggest this person is under significant external pressure. They are likely still functioning but feeling strained.',
        'what_not_to_do': "Don't offer solutions or tell them to relax. They already know what they should do.",
        'what_to_say': 'Try: "That sounds like a lot to carry. What has been the hardest part?" Then just listen.',
    },
    'depression': {
        'label': 'Depression-stage signals',
        'color': '#534AB7',
        'description': 'The patterns suggest this person may be withdrawing inward. Energy is low, motivation is flat, and they may feel like nothing will change.',
        'what_not_to_do': "Don't say just think positive or others have it worse. This increases shame.",
        'what_to_say': 'Try: "I have noticed you seem quieter lately. I am not going anywhere — whenever you want to talk, I am here." No pressure, no expectation.',
    },
    'crisis': {
        'label': 'High-distress signals',
        'color': '#A32D2D',
        'description': 'The patterns suggest this person may be at a high-distress point. Hopelessness language and self-directed pain are present. This warrants closer attention.',
        'what_not_to_do': "Don't panic visibly or immediately push professional help — it can cause shutdown.",
        'what_to_say': 'Try: "I care about you and I have been worried. Can we just sit together for a bit?" Presence matters more than words right now.',
    }
}

def predict(features: dict, input_type: str = 'first_person') -> dict:
    x = pd.DataFrame([{col: features[col] for col in FEATURE_COLS}])
    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[0].copy()

    if input_type == 'observer':
        crisis_idx = list(le.classes_).index('crisis')
        depression_idx = list(le.classes_).index('depression')
        stress_idx = list(le.classes_).index('stress')

        functioning = features.get('functioning_score', 0)
        withdrawal = features.get('withdrawal_score', 0)
        hopelessness = features.get('hopelessness_score', 0)
        first_person = features.get('first_person_ratio', 0)

        # Base dampening — observer text always over-triggers crisis
        base_dampen = 0.35

        # More functioning signals = more dampening
        if functioning > 0:
            base_dampen += 0.12 * functioning

        # Withdrawal without hopelessness = depression not crisis
        if withdrawal >= 1 and hopelessness == 0:
            base_dampen += 0.25

        # No first person = definitely observer, dampen more
        if first_person == 0:
            base_dampen += 0.10

        base_dampen = min(base_dampen, 0.75)

        overflow = proba[crisis_idx] * base_dampen
        proba[crisis_idx] -= overflow

        # Where to redistribute
        if withdrawal >= 1 and hopelessness == 0:
            # Behavioral withdrawal = depression stage
            proba[depression_idx] += overflow * 0.85
            proba[stress_idx] += overflow * 0.15
        elif functioning > 0:
            # Still functioning = stress stage
            proba[stress_idx] += overflow * 0.75
            proba[depression_idx] += overflow * 0.25
        else:
            proba[depression_idx] += overflow * 0.65
            proba[stress_idx] += overflow * 0.35

    pred_idx = np.argmax(proba)
    stage = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    all_probs = {
        le.inverse_transform([i])[0]: round(float(p), 3)
        for i, p in enumerate(proba)
    }

    return {
        'stage': stage,
        'confidence': round(confidence, 3),
        'probabilities': all_probs,
        'guidance': STAGE_DESCRIPTIONS[stage]
    }
