# Bloom Multilingual Chatbot

## Conversate effortlessly in more than 50 languages!

<div align="center">
    <img src="https://img.shields.io/github/languages/top/AstraBert/bloom-multilingual-chatbot" alt="GitHub top language">
   <img src="https://img.shields.io/github/commit-activity/t/AstraBert/bloom-multilingual-chatbot" alt="GitHub commit activity">
   <img src="https://img.shields.io/badge/chatbot-stable-green" alt="Static Badge">
   <img src="https://img.shields.io/badge/Release-v0.0.0-purple" alt="Static Badge">
   <img src="https://img.shields.io/badge/Docker_image_size-5.98GB-red" alt="Static Badge">
   <img src="https://img.shields.io/badge/Supported_platforms-linux/amd64-brown" alt="Static Badge">
   <div>
        <a href="https://astrabert.github.io/bloom-multilingual-chat"><img src="./multilingualbloom.png"></a>
        <p><i>This logo was generated with <a href="https://www.coze.com/s/ZmFqxkofJ/">CoderLogon</a>, a Coze bot that generates logos for your GitHub repos, exploiting <a href="https://pollinations.ai/">Pollinations AI</a> API</i></p>
   </div>
</div>

## Yes, ChatGPT is multilingual, but...
...It does not yield the same high performances that you can get by querying it in English. 

For non-native speakers this can represent an initial barrier, for two reasons:🚧

### 1. Engineering effective prompts 
When English is not your first language, generating on-point questions that fully express what you mean can be hard, and it is not unusual that ChatGPT or other language model get confused about what you are asking for, at least with their first answers.🤔

### 2. Unreliable results in your mother-tongue
On the other hand, when trying to speak with the LLM in your native language (especially if it is not well represented in the World-Wide-Web cultural products), you can bump into awkward phrasing, errors or difficulties in interpreting idioms and other everyday expressions.🤨

## What can we do?
It would be great if we could generate a multilingual LLM from scratch, and [Bigscience](https://bigscience.huggingface.co/), for instance, is doing a lot in this direction with Bloom🌸.

Nevertheless, we can also decide to build upon already-existent English-based models, without finetuning or retraining them, but with a clever workaround: we can use a filtering function that is able to translate the user's native language query in English, feeding it to the LLM and retrieving the response, which will be eventually back-translated from English to the original language.㊗️

Curious of trying? Let's use some python to build it!🐍

### 1. Import all the necessary dependencies
To build a multi-lingual chatbot, you'll need several dependencies, which you can install via `pip`:

```bash
python3 -m pip install transformers==4.39.3 \
langdetect==1.0.9 \
deep-translator==1.11.4 \
torch==2.1.2 \
gradio==4.28.3
```
Let's see what these packages do:

- **transformers** is a package by Hugging Face, that helps you interact with models on HF Hub ([GitHub](https://github.com/huggingface/transformers))
- **langdetect** is a package for automated language detection ([Github](https://github.com/Mimino666/langdetect))
- **deep-translator** is a package to translate sentences, based on several translation services ([GitHub](https://github.com/nidhaloff/deep-translator))
- **torch** is a package to manage tensors and dynamic neural networks in python ([GitHub](https://github.com/pytorch/pytorch))
- **gradio** is a package developed to ease the development of app interfaces in python and other languages ([GitHub](https://github.com/gradio-app/gradio))

### 2. Build the back-end architecture
We need to build a back-end architecture that looks like this (realized with [Drawio](https://app.diagrams.net/)):

![Multilingual chatbot flowchart](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/njyj6q8h96z8yhrb0dv5.png) 

Let's define a `Translation` class that helps us with detecting the original language and translating it:

```python
from langdetect import detect
from deep_translator import GoogleTranslator

class Translation:
    def __init__(self, text, destination):
        self.text = text
        self.destination = destination
        try:
            self.original = detect(self.text) # detect original
        except Exception as e:
            self.original = "auto" # if it does not work, default to "auto"
    def translatef(self):
        translator = GoogleTranslator(source=self.original, target=self.destination) # use Google Translate, one of the fastest translators available
        translation = translator.translate(self.text)
        return translation
```

As you can see, the class takes, as arguments, the text we want to translate (`text`) and the language we want to translate it into (`destination`). 

Let's now load the LLM that we want to use for our purposes: we'll start with Bigscience's Bloom-1.7B, which is a medium-sized LLM and a good match for a 16GB RAM, 2-core CPU hardware.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7") # import the model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7") # load the tokenizer

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, repetition_penalty=1.2, temperature=0.4) # prepare the inference pipeline
```
We define a maximum number of generated tokens (2048), set the repetition penalty to 1.2 (fairly high) in order to avoid the model repeating the same thing over and over again, and we keep the temperature ("creativity" in generating the response) quite low. 

Now, let's create a function that is able to take a message from the chat, translate it to English (unless it is already in English), feed it as a prompt to Bloom, retrieve the English response and back-translate it into the original language:

```python
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
```
We have all we need for our back-end architecture, it is time to build the front-end interface!

### 3. Build the front-end user interface
With Gradio, building the user's interface is as simple as one line of code:

```python
demo = gr.ChatInterface(fn=reply, title="Multilingual-Bloom Bot")
```

Now we can launch the application with:

```python
demo.launch()
```

And, imagining that we saved the whole script in a file titled `chat.py`, to make the chatbot run we go to our terminal and type:

```bash
python3 chat.py
```

Then we patiently wait and head over to the local server link that Gradio will give us once everything is loaded and ready to work!

If you want to find the source code, go to the [scripts](./scripts/) folder.

## Demo
Do you want to try what we just created? Make sure to visit this Hugging Face Space I built: [as-cle-bert/bloom-multilingual-chat](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat)💻.

## Run `bloom-multilingual-chatbot` on your machine

**bloom-multilingual-chatbot** is also available as a Docker image:

```bash
docker pull ghcr.io/astrabert/bloom-multilingual-chatbot:latest
```
You can then make it run with the following command:

```bash
docker run -p 7860:7860 ghcr.io/astrabert/bloom-multilingual-chatbot:latest
```
**IMPORTANT NOTE**: running the app within `docker run` does not log the port on which the app is running until you press `Ctrl+C`, but in that moment it also interrupt the execution! The app will run on port `0.0.0.0:7860` (or `localhost:7860` if your browser is Windows-based), so just make sure to open your browser on that port and to refresh it after 1 to 5 mins (depending on your computer and network capacities), when the model and the tokenizer should be loaded and the app should be ready to work!

Another fundamental caveat is that we are dealing here with a relatively small language model (approx. 3GB), so the it is CPU-friendly (you can run it GPUless): to make the docker container work, indeed, 8GB RAM + 12 cores CPU can be enough, but language generation will be really slow. 

**You will need at least 16 to 32 GB RAM and/or a GPU to speed up the model.**

## Support
If you like the idea, make sure to show your support by leaving a little ⭐ on GitHub!

If you please, support my open-source work by [funding me on GitHub](https://github.com/sponsors/AstraBert): in this way, it will be possible for me to improve my multilingual chatbot performances by hosting it on a more powerful hardware on HF.
