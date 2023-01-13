import transformers
import adatest
import os
import openai
import numpy as np

id = 'dw1'
logname  = 'qna' + id + '.log'
import logging
logging.basicConfig(
    filename=logname,
    format='{asctime}\t{levelname}\t{message}',
    style='{',
    level=logging.STUDY,
    datefmt='%Y-%m-%d %H:%M:%S'
)


# openai.api_key = os.environ.get("OPENAI_API_KEY")
with open(os.path.expanduser('~/.openai_api_key.txt'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

def run_openai(prompts):
    completions = []
    for prompt in prompts:
        response = openai.Completion.create(
                    engine='text-curie-001', prompt=prompt, max_tokens=40,
                    temperature=0.7, top_p= 0.95 , n=1, stop="")
        suggestion_texts = [choice["text"] for choice in response["choices"]]
        completions.append(suggestion_texts[0].strip())
    return completions



# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('text-davinci-001',)

# print(run_openai('hello how are you'))
csv_filename = 'qna' +id + '.csv'
# create a new test tree
# tests = adatest.TestTree(csv_filename)
tests = adatest.TestTree('ankurgpt3.csv')


# # adapt the tests to our model to launch a notebook-based testing interface
# # (wrap with adatest.serve to launch a standalone server)
# # adatest.serve(tests.adapt(classifier, generator, auto_save=True), port=8089)
adatest.serve(tests.adapt(run_openai, generator=generator, auto_save=True, control=False,description="question"),port = 8081)


