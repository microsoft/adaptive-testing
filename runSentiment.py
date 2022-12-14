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
    print(type(documents), len(documents))
    doc_result_total = []
    batch_size = 10
    # for num,i in enumerate(range(len(documents)//10 + 1 )) : 
    for i in range(0, len(documents), batch_size):
        print( i, i+batch_size)
  
        result = client.analyze_sentiment(documents[i:i+batch_size], show_opinion_mining=False)
        
        doc_result_batch = [doc for doc in result if not doc.is_error]
        
        doc_result_total.extend(doc_result_batch)
    for document in doc_result_total:
        outcome.append([])
    
        outcome[-1] = [document.confidence_scores.positive , document.confidence_scores.neutral,document.confidence_scores.negative ]
        # res = [document.confidence_scores.positive , document.confidence_scores.neutral,document.confidence_scores.negative ]

        # score = np.max(res)
        # label = ['Positive', 'Neutral', 'Negative'][np.argmax(res)]
        # outcome[-1] = [{'label':label, 'score':score}]

        # ))
    return outcome
        
doc_ex = [
        "The food and service were unacceptable. The concierge was nice, however.", 'fuck off', 'i love you', 'sdf' , 'wef', 'yhrhtyhb', 'fef', 'rgege', 'grgs',
        'grgg', 'weada', 'awad' ,'bmnjm', 'vnghn'
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

adatest.serve(tests.adapt(model, generator=generator, auto_save=True, control=False, description="sentence"), port=8977)
