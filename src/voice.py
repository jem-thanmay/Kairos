import speech_recognition as sr

# Common speech recognition misheard words in mental health context
SPEECH_CORRECTIONS = {
    'digging on': 'dealing with',
    'digging with': 'dealing with',
    'dilling with': 'dealing with',
    'dealing on': 'dealing with',
    'i am drown': 'i am drowning',
    'i feel say': 'i feel sad',
    'i feel bad lee': 'i feel badly',
    'exhausted': 'exhausted',
    'a zinc': 'anxious',
    'a shush': 'anxious',
    'over helm': 'overwhelmed',
    'over whelmed': 'overwhelmed',
    'de motivated': 'demotivated',
    'un motivated': 'unmotivated',
    'burnt out': 'burnout',
    'burn tout': 'burnout',
}

def correct_speech(text: str) -> str:
    text_lower = text.lower()
    for wrong, right in SPEECH_CORRECTIONS.items():
        text_lower = text_lower.replace(wrong, right)
    return text_lower

def record_and_transcribe(timeout: int = 10, phrase_limit: int = 30) -> dict:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit
            )

        text = recognizer.recognize_google(audio)
        text = correct_speech(text)
        return {'success': True, 'text': text, 'auto_send': True}

    except sr.WaitTimeoutError:
        return {'success': False, 'error': 'No speech detected after 10 seconds.'}
    except sr.UnknownValueError:
        return {'success': False, 'error': 'Could not understand. Please try again.'}
    except sr.RequestError:
        return {'success': False, 'error': 'Speech service unavailable. Check internet.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
