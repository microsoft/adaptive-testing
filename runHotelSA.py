import transformers
import adatest
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

id = '0000'
logname  = 'hotelsa' + id + '.log'
import logging
logging.basicConfig(
    filename=logname,
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

huggingfacemodel = 'nlptown/bert-base-multilingual-uncased-sentiment'
# 'BellaAndBria/distilbert-base-uncased-finetuned-emotion'
# huggingfacemodel = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(huggingfacemodel)

model = AutoModelForSequenceClassification.from_pretrained(huggingfacemodel)


# create a HuggingFace sentiment analysis model
# classifier = transformers.pipeline("sentiment-analysis", return_all_scores=True)

classifier = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
labels = ['Negative','Negative', 'Neutral', 'Positive', 'Positive']
model = adatest.Model(classifier, output_names = labels)
# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-001')

# print(classifier(['how are you?', 'where are y9ou?']))
#  [[{'label': '5 stars', 'score': 0.47160935401916504}], [{'label': '1 star', 'score': 0.3922109305858612}]]

# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
# tests = adatest.TestTree("hotel_reviews.csv")
csv_filename = 'hotelsa' +id + '.csv'
# create a new test tree
tests = adatest.TestTree(csv_filename)


# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)

adatest.serve(tests.adapt(model, generator=generator, auto_save=True, control=False, description="short hotel review"), port=8067)