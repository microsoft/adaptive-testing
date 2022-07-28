import transformers
import adatest
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')


# create a HuggingFace sentiment analysis model
classifier = transformers.pipeline("sentiment-analysis", return_all_scores=True)

# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-curie-001',
                                     )

# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("bias_besa_thinkaloud_midway.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8089)
