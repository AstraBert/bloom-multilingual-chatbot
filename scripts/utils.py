from langdetect import detect
from deep_translator import GoogleTranslator

class Translation:
    def __init__(self, text, destination):
        self.text = text
        self.destination = destination
        try:
            self.original = detect(self.text)
        except Exception as e:
            self.original = "auto"
    def translatef(self):
        translator = GoogleTranslator(source=self.original, target=self.destination)
        translation = translator.translate(self.text)
        return translation