import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def generate_guidance(
    user_input: str,
    stage: str,
    confidence: float,
    probabilities: dict,
    input_type: str,
    relationship: str = None,
    key_signals: dict = None
) -> dict:

    input_preview = user_input[:150]

    if input_type == 'observer':
        prompt = f"""A {relationship or "person"} wrote this about someone they care about:
"{input_preview}"

The situation shows {stage}-stage patterns.

Respond with ONLY a JSON object. Reference the specific details they wrote about.
Do not use generic advice. Do not use therapy language.

{{"description": "one sentence naming what you noticed in their specific words, warm not clinical", "what_to_say": "exact natural words they can say, starting with Try:, specific to this {relationship or "relationship"}", "what_not_to_do": "one specific thing to avoid based on what they described, under 25 words"}}"""

    else:
        prompt = f"""Someone wrote this about how they are feeling:
"{input_preview}"

The situation shows {stage}-stage patterns.

Respond with ONLY a JSON object. Reference what they specifically wrote.
Speak like a caring friend, not a therapist. No generic wellness advice.

{{"description": "two sentences showing you read their specific words, make them feel genuinely seen", "what_to_say": "one small doable thing connected directly to what they wrote, not generic", "what_not_to_do": "one gentle reframe about how they might be judging themselves, reference their actual words, under 25 words"}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.85,
                    "num_predict": 400
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            raw = response.json().get('response', '').strip()

            # Clean formatting
            raw = raw.replace('```json', '').replace('```', '').strip()

            # Find JSON boundaries
            start = raw.find('{')
            end = raw.rfind('}') + 1
            if start != -1 and end > start:
                json_str = raw[start:end]

                # Fix escaped single quotes that break JSON
                json_str = json_str.replace("\\'", "'")

                try:
                    parsed = json.loads(json_str)
                    required = ['description', 'what_to_say', 'what_not_to_do']
                    if all(k in parsed for k in required):
                        return {
                            'success': True,
                            'description': parsed['description'],
                            'what_to_say': parsed['what_to_say'],
                            'what_not_to_do': parsed['what_not_to_do'],
                            'source': 'ollama'
                        }
                except json.JSONDecodeError:
                    # Try extracting values with regex as fallback
                    try:
                        desc = re.search(r'"description"\s*:\s*"([^"]+)"', json_str)
                        say = re.search(r'"what_to_say"\s*:\s*"([^"]+)"', json_str)
                        avoid = re.search(r'"what_not_to_do"\s*:\s*"([^"]+)"', json_str)
                        if desc and say and avoid:
                            return {
                                'success': True,
                                'description': desc.group(1),
                                'what_to_say': say.group(1),
                                'what_not_to_do': avoid.group(1),
                                'source': 'ollama'
                            }
                    except:
                        pass

            return {'success': False, 'error': f'Could not parse response'}

        return {'success': False, 'error': f'Status {response.status_code}'}

    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Ollama not running'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_ollama_connection() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False
