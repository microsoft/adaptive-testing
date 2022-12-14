import transformers
import toxicity
import adatest
import os
import openai
import numpy as np

id = '0002'
logname  = 'perspective' + id + '.log'
import logging
logging.basicConfig(
    filename=logname,
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

with open(os.path.expanduser('~/.googlePerspectiveAPI.txt'), 'r') as file:
    googleapikey = file.read().replace('\n', '')


t = toxicity.ToxicityModel('cache_file', googleapikey)
classifier = t.predict_proba
model = adatest.Model(classifier, output_names = ["Non-toxic", "Toxic"])
# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-002',)                               

# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)
csv_filename = 'perspective' +id + '.csv'
# create a new test tree
tests = adatest.TestTree(csv_filename)

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(model, generator=generator, auto_save=True, control=False, description="tweet"), port=8099)
