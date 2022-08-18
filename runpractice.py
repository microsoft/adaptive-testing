import transformers
import adatest
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

huggingface = 'BellaAndBria/distilbert-base-uncased-finetuned-emotion'
# create a HuggingFace sentiment analysis model

import logging
logging.basicConfig(
    filename='practice.log',
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(huggingface)

model = AutoModelForSequenceClassification.from_pretrained(huggingface)
classifier = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
labels = ['negative','negative', 'neutral', 'positive', 'positive']
model = adatest.Model(classifier, output_names = labels)
# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-002')



# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("practice.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
# adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8089)
adatest.serve(tests.adapt(classifier, generator=generator, auto_save=True, control=False, description="statement"), port=8069)


