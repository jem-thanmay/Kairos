import speech_recognition as sr

def record_and_transcribe(timeout: int = 5, phrase_limit: int = 30) -> dict:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit
            )

        text = recognizer.recognize_google(audio)
        return {'success': True, 'text': text}

    except sr.WaitTimeoutError:
        return {'success': False, 'error': 'No speech detected. Please try again.'}
    except sr.UnknownValueError:
        return {'success': False, 'error': 'Could not understand. Please speak clearly.'}
    except sr.RequestError:
        return {'success': False, 'error': 'Speech service unavailable. Check your internet.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
