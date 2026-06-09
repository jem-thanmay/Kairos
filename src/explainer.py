import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import io
import base64

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE, 'models', 'cascade_classifier.joblib'))
scaler = joblib.load(os.path.join(BASE, 'models', 'scaler.joblib'))
le = joblib.load(os.path.join(BASE, 'models', 'label_encoder.joblib'))

explainer = shap.TreeExplainer(model)

FEATURE_COLS = [
    'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
    'isolation_score', 'hopelessness_score', 'burnout_score', 'self_blame_score',
    'first_person_ratio', 'question_count', 'negation_ratio',
    'word_count', 'avg_sentence_length', 'exclamation_ratio'
]

FEATURE_LABELS = {
    'sentiment_neg': 'Negative sentiment',
    'sentiment_neu': 'Neutral tone',
    'sentiment_pos': 'Positive sentiment',
    'sentiment_compound': 'Overall emotional tone',
    'isolation_score': 'Isolation language',
    'hopelessness_score': 'Hopelessness language',
    'burnout_score': 'Burnout language',
    'self_blame_score': 'Self-blame language',
    'first_person_ratio': 'Self-focus (I/me/my)',
    'question_count': 'Questions asked',
    'negation_ratio': 'Negation (no/not/never)',
    'word_count': 'Length of text',
    'avg_sentence_length': 'Sentence length',
    'exclamation_ratio': 'Emotional intensity'
}

def get_shap_explanation(features: dict, stage: str) -> dict:
    x = pd.DataFrame([{col: features[col] for col in FEATURE_COLS}])
    x_scaled = scaler.transform(x)

    shap_values = explainer.shap_values(x_scaled)

    stage_idx = list(le.classes_).index(stage)

    if isinstance(shap_values, list):
        sv = shap_values[stage_idx][0]
    else:
        sv = shap_values[0, :, stage_idx]

    feature_importance = []
    for i, col in enumerate(FEATURE_COLS):
        feature_importance.append({
            'feature': col,
            'label': FEATURE_LABELS[col],
            'shap_value': float(sv[i]),
            'feature_value': float(x[col].iloc[0])
        })

    feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    top_features = feature_importance[:6]

    return {
        'top_features': top_features,
        'stage': stage
    }

def plot_shap_bar(explanation: dict, stage: str) -> str:
    stage_colors = {
        'stress': '#C9A87A',
        'depression': '#7B72D4',
        'crisis': '#C97A7A'
    }
    color = stage_colors.get(stage, '#7A8C84')

    features = explanation['top_features']
    labels = [f['label'] for f in features][::-1]
    values = [abs(f['shap_value']) for f in features][::-1]
    directions = [f['shap_value'] for f in features][::-1]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#FAFAF7')
    ax.set_facecolor('#FAFAF7')

    bars = ax.barh(
        labels,
        values,
        color=[color if d > 0 else '#A8B8B0' for d in directions],
        alpha=0.85,
        height=0.55,
        edgecolor='none'
    )

    ax.set_xlabel('Signal strength', fontsize=9, color='#7A8C84')
    ax.tick_params(axis='y', labelsize=9, colors='#2C3E35')
    ax.tick_params(axis='x', labelsize=8, colors='#7A8C84')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#E8EDE9')
    ax.xaxis.grid(True, color='#E8EDE9', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#FAFAF7')
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"
