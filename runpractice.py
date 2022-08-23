import transformers
import adatest
import os
import openai
import numpy as np


if os.path.exists('~/.openai_api_key.txt'):
    with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
        openai.api_key = file.read().replace('\n', '')
elif os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    raise Exception("You need to set your OpenAI API key in ~/.openai_api_key.txt or OPENAI_API_KEY environment variable")


huggingface =  "textattack/distilbert-base-uncased-CoLA"
# 'BellaAndBria/distilbert-base-uncased-finetuned-emotion'
# create a HuggingFace sentiment analysis model

import logging
formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
formatter.datefmt = '%Y-%m-%d %H:%M:%S'

study_handler = logging.FileHandler("practice.log")
study_handler.setFormatter(formatter)
study_handler.setLevel(logging.STUDY)

dbg_handler = logging.FileHandler('practice_debug.log')
dbg_handler.setFormatter(formatter)
dbg_handler.setLevel(logging.DEBUG)

logging.root.addHandler(dbg_handler)
logging.root.addHandler(study_handler)
logging.root.setLevel(logging.DEBUG)

# logging.basicConfig(
#     filename='practice.log',
#     format='{asctime}\t{levelname}\t{message}',
#     style='{',
#     level=logging.STUDY,
#     datefmt='%Y-%m-%d %H:%M:%S'
# )


formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
formatter.datefmt = '%Y-%m-%d %H:%M:%S'
study_handler = logging.FileHandler("practice.log")
study_handler.setFormatter(formatter)
study_handler.setLevel(logging.STUDY)
dbg_handler = logging.FileHandler('practice_debug.log')
dbg_handler.setFormatter(formatter)
dbg_handler.setLevel(logging.DEBUG)
logging.root.addHandler(dbg_handler)
logging.root.addHandler(study_handler)
logging.root.setLevel(logging.DEBUG)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(huggingface)

model = AutoModelForSequenceClassification.from_pretrained(huggingface)
classifier = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
labels = ['Unacceptable', 'Acceptable']
model = adatest.Model(classifier, output_names = labels)
# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-002')



# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("practiceStart.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
# adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8089)
adatest.serve(tests.adapt(model, generator=generator, auto_save=True, control=True, description="sentence"), port=8069)


