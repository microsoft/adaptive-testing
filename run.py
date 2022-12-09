import transformers
import adatest
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')


import logging

id = '0001'
logname  = 'run' + id + '.log'

logging.basicConfig(
    filename=logname,
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
# create a HuggingFace sentiment analysis model
classifier = transformers.pipeline("sentiment-analysis", top_k = 1)

# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-curie-001',
                                     )
print(classifier(['hey','hi']))
# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("charvi_scrat_new.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
# adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8089)
adatest.serve(tests.adapt(classifier, generator=generator, auto_save=True, control=False, description="internet comment"),port = 8081)


