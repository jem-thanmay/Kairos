import requests
import json
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

SYSTEM_PROMPT = """You are Kairos — a quiet, warm presence. Someone has chosen to talk to you.

Your role:
- Listen more than you speak
- Ask one gentle follow-up question per response
- Never diagnose, never use clinical terms
- Never say things will get better or use toxic positivity
- Speak like a trusted friend who happens to understand people deeply
- Keep responses short — 2 to 4 sentences maximum
- If someone expresses hopelessness or thoughts of not wanting to be here, acknowledge it warmly and gently mention that talking to someone trained for this moment can help

You are not a therapist. You are a quiet place to think out loud."""

def build_conversation_prompt(
    history: List[Dict],
    new_message: str,
    stage: str = None
) -> str:

    stage_context = ""
    if stage:
        stage_context = f"\n[Context: this person is showing {stage}-stage patterns. Respond accordingly — do not mention this to them.]\n"

    conversation = ""
    for turn in history:
        role = "Person" if turn['role'] == 'user' else "Kairos"
        conversation += f"{role}: {turn['content']}\n"

    conversation += f"Person: {new_message}\nKairos:"

    return f"{SYSTEM_PROMPT}{stage_context}\n\n{conversation}"

def get_response(
    history: List[Dict],
    new_message: str,
    stage: str = None
) -> Dict:

    prompt = build_conversation_prompt(history, new_message, stage)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150,
                    "stop": ["Person:", "Human:", "\n\n"]
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            raw = response.json().get('response', '').strip()
            # Clean up any role prefixes the model might add
            raw = raw.replace("Kairos:", "").strip()
            return {'success': True, 'response': raw}

        return {'success': False, 'error': f'Status {response.status_code}'}

    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Ollama not running'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_opening_message(stage: str = None) -> str:
    """Generate a warm opening message based on detected stage."""
    openings = {
        'stress': "Hey. I'm glad you're here. It sounds like things have been a lot lately. What's been weighing on you most?",
        'depression': "I'm here. You don't have to explain everything or have the right words. What's been going on for you?",
        'crisis': "I'm here with you. Whatever you're carrying right now — you don't have to carry it alone in this moment. Can you tell me a little about what's been happening?",
        None: "I'm here. Whatever brought you here today — you can say it however it comes out. What's on your mind?"
    }
    return openings.get(stage, openings[None])

def check_crisis_signal(message: str) -> bool:
    """Check if a message contains crisis-level signals."""
    crisis_phrases = [
        "don't want to be here", "dont want to be here",
        "end it", "end my life", "kill myself",
        "not worth living", "better off dead",
        "want to die", "want to disappear forever",
        "no reason to live", "give up on life"
    ]
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in crisis_phrases)
