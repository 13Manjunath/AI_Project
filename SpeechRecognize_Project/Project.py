from flask import Flask, render_template, request
import speech_recognition as sr

project = Flask(__name__)

@project.route("/", methods=["GET", "POST"])
def index():
    recognized_text = ""

    if request.method == "POST":
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source)

        try:
            recognized_text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            recognized_text = "Could not understand audio"
        except sr.RequestError:
            recognized_text = "Speech service error"

    return render_template("index.html", text=recognized_text)

if __name__ == "__main__":
    project.run(debug=True)
