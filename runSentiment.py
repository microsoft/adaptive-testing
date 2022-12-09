import transformers
import toxicity
import adatest
import os
import openai
import numpy as np
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

id = '0002'
logname  = 'azuresa' + id + '.log'
import logging
logging.basicConfig(
    filename=logname,
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open(os.path.expanduser('~/.azure_language_key.txt'), 'r') as file:
    language_key = file.read().replace('\n', '')


with open(os.path.expanduser('~/.azure_language_endpoint.txt'), 'r') as file:
    language_endpoint = file.read().replace('\n', '')


with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')



# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

# Example method for detecting sentiment and opinions in text 
def sentiment_analysis(documents, client=client):

    outcome =[]
    result = client.analyze_sentiment(documents, show_opinion_mining=False)
    doc_result = [doc for doc in result if not doc.is_error]


    for document in doc_result:
        outcome.append([])
        # outcome[-1] = [{'label':'Positive', 'score':document.confidence_scores.positive},
        # {'label':'Neutral', 'score':document.confidence_scores.neutral},
        # {'label':'Negative', 'score':document.confidence_scores.negative}, ]
        # res = 
        outcome[-1] = [document.confidence_scores.positive , document.confidence_scores.neutral,document.confidence_scores.negative ]
        # score = np.max(res)
        # label = ['Positive', 'Neutral', 'Negative'][np.argmax(res)]
        # outcome[-1] = [{'label':label, 'score':score}]
    print(outcome)
        # print("Document Sentiment: {}".format(document.sentiment))
        # print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
        #     document.confidence_scores.positive,
        #     document.confidence_scores.neutral,
        #     document.confidence_scores.negative,
        # ))
    return outcome
        
doc_ex = [
        "The food and service were unacceptable. The concierge was nice, however.", 'fuck off', 'i love you'
    ]

# classifier = 
labels = ['Positive', 'Neutral', 'Negative']
model = adatest.Model(sentiment_analysis, output_names = labels)
# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-002')
print(sentiment_analysis(doc_ex))
csv_filename = 'azuresa' +id + '.csv'
# create a new test tree
tests = adatest.TestTree(csv_filename)


# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)

adatest.serve(tests.adapt(model, generator=generator, auto_save=True, control=False, description="sentence"), port=8967)
