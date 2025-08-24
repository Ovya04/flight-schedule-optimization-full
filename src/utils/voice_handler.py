# src/utils/voice_handler.py
import speech_recognition as sr, pyttsx3
from googletrans import Translator

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.translator = Translator()
    def listen(self,lang='en'):
        with sr.Microphone() as src:
            audio = self.recognizer.listen(src)
        text = self.recognizer.recognize_google(audio,language=lang)
        return text
    def speak(self,text,lang='en'):
        if lang!='en':
            text = self.translator.translate(text,dest=lang).text
        self.tts.say(text); self.tts.runAndWait()
