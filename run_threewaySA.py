import transformers
import adatest
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')



from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


classifier = transformers.pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)


generator = adatest.generators.OpenAI('text-curie-001')

# create a new test tree
tests = adatest.TestTree("forough_review_sentiment_analysis_scratch1.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8098)
