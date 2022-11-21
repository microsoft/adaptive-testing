import argparse

import json

import os

import collections

import numpy as np

import adatest

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from torch.nn import functional as F

import openai

from adatest.utils import cogservices, toxicity

import importlib

import sentence_transformers

import logging

import sys

import copy

import shutil

 

log = logging.getLogger(__name__)

 

def chunks(l, n):

    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):

        yield l[i:i + n]

 

def get_gpt_model_scorer():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained('gpt2')

    space_prefix = tokenizer.tokenize(' john')[0].split('john')[0]

    special_chars = set([i for x, i in tokenizer.get_vocab().items() if not x.strip(space_prefix).isalnum()])

    tokens = []

    token_id = {}

    for x, i in tokenizer.get_vocab().items():

        x = x.strip(space_prefix)

        if x not in token_id:

            token_id[x] = len(tokens)

            tokens.append(x)

    after_special_mapping = []

    mapping = []

    vocab = tokenizer.get_vocab()

    for i, token in enumerate(tokens):

        after_special_mapping.append(vocab.get(token.strip(space_prefix)))

        if vocab.get(token) in special_chars:

            mapping.append(vocab.get(token.strip(space_prefix)))

        else:

            mapping.append(vocab.get(space_prefix+token, vocab.get(token)))

 

    tokens = np.array(tokens)

    token_map = dict([(x, i) for i, x in enumerate(tokens)])

    def predict_next_word(context):

        context = context.strip()

        inputs = tokenizer(context, return_tensors="pt")['input_ids']

        with torch.no_grad():

            outputs = model(inputs, labels=inputs)

        loss, logits = outputs[:2]

        next_word_logits = logits[0][-1].detach()

        logits = F.softmax(next_word_logits, dim=0)

        if inputs[-1] in special_chars:

            logits =  logits[after_special_mapping]

        else:

            logits =  logits[mapping]

        return logits.numpy()

#         return list(zip(tokens, logits.numpy()))

    def predict_next(sentences):

        out = []

        for x in sentences:

            try:

                val = predict_next_word(x)

            except Exception:

                val = np.zeros(len(predict_next.output_names)) * np.nan

            out.append(val)

        return np.vstack(out)

    predict_next.tokens = tokens

    predict_next.token_map=token_map

    tensor_output_model = predict_next

    tensor_output_model.output_names = tokens

    scorer = adatest.TextScorer(tensor_output_model, topk=3)

    return tensor_output_model, scorer

 

def get_sentiment_model_scorer():

    with open(os.path.expanduser('~/.sentiment_key'), 'r') as file:

        key = file.read().replace('\n', '')

    model = cogservices.SentimentModel('./newazure.pkl', key, wait_time=0)

    labels = ['negative', 'neutral', 'positive']

    tensor_output_model = lambda x: model.predict_proba(x)

    tensor_output_model.output_names = labels

    # make thes output come out as a nice vector

    scorer = adatest.TextScorer(tensor_output_model)

    return tensor_output_model, scorer

 

def get_toxicity_model_scorer():

    with open(os.path.expanduser('~/.toxicity_key'), 'r') as file:

        key = file.read().replace('\n', '')

    model = toxicity.ToxicityModel('./toxicitypreds.pkl', key)

    labels = ['acceptable', 'toxic']

    tensor_output_model = lambda x: model.predict_proba(x)

    tensor_output_model.output_names = labels

    # make thes output come out as a nice vector

    scorer = adatest.TextScorer(tensor_output_model)

    return tensor_output_model, scorer