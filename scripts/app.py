import gradio as gr
from load_model import pipe
from utils import Translation



def reply(message, history):
    txt = Translation(message, "en")
    if txt.original == "en":
        response = pipe(message)
        return response[0]["generated_text"]
    else:
        translation = txt.translatef()
        response = pipe(translation)
        t = Translation(response[0]["generated_text"], txt.original)
        res = t.translatef()
        return res


demo = gr.ChatInterface(fn=reply, title="Multilingual-Bloom Bot")
demo.launch(server_name="0.0.0.0", share=False)