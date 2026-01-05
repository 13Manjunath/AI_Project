import speech_recognition as sr

def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use microphone as source
    with sr.Microphone() as source:
        print("🎤 Speak something...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        print("✅ Recognized Text:")
        print(text)

    except sr.UnknownValueError:
        print("❌ Could not understand the audio")

    except sr.RequestError as e:
        print(f"❌ API error: {e}")

if __name__ == "__main__":
    speech_to_text()
