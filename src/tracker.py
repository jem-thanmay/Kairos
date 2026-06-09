import json
import os
from datetime import datetime
from typing import Optional

STORAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'sessions'
)

def _ensure_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)

def _session_path(person_id: str) -> str:
    safe_id = "".join(c for c in person_id.lower() if c.isalnum() or c == '_')
    return os.path.join(STORAGE_DIR, f"{safe_id}.json")

def save_observation(
    person_id: str,
    text: str,
    stage: str,
    confidence: float,
    probabilities: dict,
    relationship: str = None,
    input_type: str = 'observer'
) -> dict:
    _ensure_dir()
    path = _session_path(person_id)

    # Load existing
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'person_id': person_id,
            'relationship': relationship,
            'observations': []
        }

    # Add new observation
    observation = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%B %d, %Y'),
        'text': text,
        'stage': stage,
        'confidence': confidence,
        'probabilities': probabilities,
        'input_type': input_type
    }
    data['observations'].append(observation)

    # Save
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return observation

def load_history(person_id: str) -> Optional[dict]:
    path = _session_path(person_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def get_trajectory(person_id: str) -> dict:
    data = load_history(person_id)
    if not data or len(data['observations']) < 2:
        return {'has_trajectory': False}

    obs = data['observations']
    n = len(obs)

    # Get last 4 observations for trend
    recent = obs[-4:]

    # Crisis signal trend
    crisis_scores = [o['probabilities'].get('crisis', 0) for o in recent]
    depression_scores = [o['probabilities'].get('depression', 0) for o in recent]
    stress_scores = [o['probabilities'].get('stress', 0) for o in recent]

    # Stage history
    stages = [o['stage'] for o in recent]
    current_stage = stages[-1]
    previous_stage = stages[-2] if len(stages) >= 2 else None

    # Trend direction — compare first half vs second half
    mid = len(crisis_scores) // 2
    crisis_early = sum(crisis_scores[:mid]) / max(mid, 1)
    crisis_late = sum(crisis_scores[mid:]) / max(len(crisis_scores) - mid, 1)

    crisis_delta = crisis_late - crisis_early

    if crisis_delta > 0.08:
        trend = 'worsening'
        trend_color = '#A32D2D'
    elif crisis_delta < -0.08:
        trend = 'improving'
        trend_color = '#0F6E56'
    else:
        trend = 'stable'
        trend_color = '#854F0B'

    # Stage change
    stage_changed = current_stage != previous_stage

    return {
        'has_trajectory': True,
        'total_observations': n,
        'recent_stages': stages,
        'current_stage': current_stage,
        'previous_stage': previous_stage,
        'stage_changed': stage_changed,
        'trend': trend,
        'trend_color': trend_color,
        'crisis_delta': round(crisis_delta, 3),
        'crisis_scores': crisis_scores,
        'depression_scores': depression_scores,
        'stress_scores': stress_scores,
        'dates': [o['date'] for o in recent]
    }

def list_people(input_type: str = 'observer') -> list:
    _ensure_dir()
    people = []
    for f in os.listdir(STORAGE_DIR):
        if f.endswith('.json'):
            path = os.path.join(STORAGE_DIR, f)
            with open(path, 'r') as fp:
                data = json.load(fp)
            obs = data.get('observations', [])
            if obs and obs[0].get('input_type') == input_type:
                people.append({
                    'person_id': data['person_id'],
                    'relationship': data.get('relationship'),
                    'observation_count': len(obs),
                    'last_stage': obs[-1]['stage'],
                    'last_date': obs[-1]['date']
                })
    return people

def delete_person(person_id: str) -> bool:
    path = _session_path(person_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
