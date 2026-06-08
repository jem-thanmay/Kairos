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
        'observer': {
            'description': 'The patterns you described suggest this person is under significant external pressure. They are likely still functioning but feeling strained.',
            'what_to_say': 'Try: "That sounds like a lot to carry. What has been the hardest part this week?" Then just listen — no fixing, no advice.',
            'what_not_to_do': "Don't offer solutions or tell them to relax. They already know what they should do. What they need is to feel heard.",
        },
        'first_person': {
            'description': "What you are carrying sounds genuinely heavy. Stress at this level is real — it is not weakness and it is not permanent.",
            'what_to_say': 'One small thing that often helps: name one thing that is actually within your control today, however small. Not a solution — just one thing.',
            'what_not_to_do': "Try not to measure yourself against what you think you should be handling. You are dealing with a lot.",
        }
    },
    'depression': {
        'label': 'Depression-stage signals',
        'color': '#534AB7',
        'observer': {
            'description': 'The patterns suggest this person may be withdrawing inward. Energy is low, motivation is flat, and they may feel like nothing will change.',
            'what_to_say': 'Try: "I have noticed you seem quieter lately. I am not going anywhere — whenever you want to talk, I am here." No pressure, no expectation. Just leave the door open.',
            'what_not_to_do': "Don't say just think positive or others have it worse. Don't push them to do things they are not ready for. Presence without pressure is what matters right now.",
        },
        'first_person': {
            'description': "What you are describing — the flatness, the low energy, the feeling that things will not change — is something a lot of people carry silently. You are not broken.",
            'what_to_say': "You do not have to fix everything today. One tiny act of care for yourself — drinking water, stepping outside for two minutes, texting one person — is enough for right now.",
            'what_not_to_do': "Try not to judge yourself for not being able to do more. Depression makes ordinary things hard. That is not a character flaw.",
        }
    },
    'crisis': {
        'label': 'High-distress signals',
        'color': '#A32D2D',
        'observer': {
            'description': 'The patterns suggest this person may be at a high-distress point. Hopelessness language and self-directed pain are present. This warrants closer attention.',
            'what_to_say': 'Try: "I care about you and I have been worried. Can we just sit together for a bit?" Presence matters more than the right words right now. You do not need to fix anything.',
            'what_not_to_do': "Don't panic visibly or immediately push professional help as the first response — it can cause shutdown. Stay calm, stay close, and listen without judgment first.",
        },
        'first_person': {
            'description': "What you are feeling right now sounds exhausting and painful. The weight of it is real. You do not have to pretend otherwise.",
            'what_to_say': "You reached out by being here — that matters. If you have someone you trust, even a little, this would be a good time to let them know you are struggling. You do not have to explain everything. Just say you are not okay.",
            'what_not_to_do': "Try not to be alone with this right now if you can help it. The thoughts that come in isolation are not the full picture, even when they feel absolute.",
        }
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

        base_dampen = 0.35
        if functioning > 0:
            base_dampen += 0.12 * functioning
        if withdrawal >= 1 and hopelessness == 0:
            base_dampen += 0.25
        if first_person == 0:
            base_dampen += 0.10
        base_dampen = min(base_dampen, 0.75)

        overflow = proba[crisis_idx] * base_dampen
        proba[crisis_idx] -= overflow

        if withdrawal >= 1 and hopelessness == 0:
            proba[depression_idx] += overflow * 0.85
            proba[stress_idx] += overflow * 0.15
        elif functioning > 0:
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

    stage_data = STAGE_DESCRIPTIONS[stage]
    guidance_key = 'observer' if input_type == 'observer' else 'first_person'
    guidance = {
        'label': stage_data['label'],
        'color': stage_data['color'],
        'description': stage_data[guidance_key]['description'],
        'what_to_say': stage_data[guidance_key]['what_to_say'],
        'what_not_to_do': stage_data[guidance_key]['what_not_to_do'],
    }

    return {
        'stage': stage,
        'confidence': round(confidence, 3),
        'probabilities': all_probs,
        'guidance': guidance
    }
