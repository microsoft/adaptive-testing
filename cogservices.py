import sys
from tqdm.auto import tqdm
import os
import pickle
import urllib
import json
import http
import aiohttp
import asyncio
import numpy as np
import time
import requests


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class SentimentModel:
    def __init__(self, pred_file, key, batch_size=10, wait_time=1):
        self.preds = {}
        self.post_fn = None
        self.key = key
        if os.path.exists(pred_file):
            self.preds = pickle.load(open(pred_file, 'rb'))
        self.pred_file = pred_file
        self.pp_fn = self.async_predict_proba_azure
        self.batch_size = batch_size
        self.wait_time = wait_time

    def binary_to_three(self, pr):
        # This is what google recommends
        margin_neutral = 0.25
        mn = margin_neutral / 2.
        pp = np.zeros((pr.shape[0], 3))
        neg = pr < 0.5 - mn
        pp[neg, 0] = 1 - pr[neg]
        pp[neg, 2] = pr[neg]
        pos = pr > 0.5 + mn
        pp[pos, 0] = 1 - pr[pos]
        pp[pos, 2] = pr[pos]
        neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
        pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_pos] - 0.5)
        pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
        neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
        pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_neg] - 0.5)
        pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
        return (pp.T / pp.sum(1)).T # normalize to sum to 1

    def predict_proba(self, exs, silent=False):
        loop = asyncio.get_event_loop()
        to_pred = [x for x in exs if x not in self.preds]
        chunked = list(chunks(to_pred, self.batch_size))
        pps_chunked = loop.run_until_complete(asyncio.gather(*[self.pp_fn(docs) for docs in chunked]))
        for docs,pps in zip(chunked,pps_chunked):
            # pps = loop.run_until_complete(self.pp_fn(docs))
            for x, pp in zip(docs, pps):
                self.preds[x] = pp
            time.sleep(self.wait_time)
        ret = np.array([self.preds.get(x) for x in exs])
        pickle.dump(self.preds, open(self.pred_file, 'wb'))
        if self.post_fn:
            ret = self.post_fn(ret)
        return (ret.T / ret.sum(1)).T # normalize to sum to 1

    def predict_proba_sync(self, exs, silent=False):
        to_pred = [x for x in exs if x not in self.preds]
        chunked = list(chunks(to_pred, self.batch_size))
        for docs in tqdm(chunked, disable=silent):
            pps = self.predict_proba_azure(docs)
            for x, pp in zip(docs, pps):
                self.preds[x] = pp
            time.sleep(self.wait_time)
        ret = np.array([self.preds.get(x) for x in exs])
        pickle.dump(self.preds, open(self.pred_file, 'wb'))
        if self.post_fn:
            ret = self.post_fn(ret)
        return ret

    def predict_and_confidences(self, exs):
        confs = self.predict_proba(exs)
        preds = np.argmax(confs, axis=1)
        return preds, confs

    async def async_predict_proba_azure(self, exs):
        print('Azure: predicting %d examples' % len(exs))
        headers = {
        # Request headers
        'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.key,
        }
        params = urllib.parse.urlencode({
            # Request parameters
            'showStats': 'false',
        })
        body = json.dumps({'documents': [{'id': str(i), 'text': x, 'language': 'en'} for i, x in enumerate(exs)]})

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post('https://westus2.api.cognitive.microsoft.com/text/analytics/v3.1/sentiment?opinionMining=false%s" % params', data=body) as resp:
                azureresp = await resp.text()
        # print(azureresp)
        try:
            pps = np.array([[x['confidenceScores'][a] for a in ['negative', 'neutral', 'positive']] for x in json.loads(azureresp)['documents']])
        except:
            print(json.loads(azureresp))
            raise Exception()
        return pps

    def predict_proba_azure(self, exs):

        print('Azure: predicting %d examples' % len(exs))
        headers = {
        # Request headers
        'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.key,
        }
        params = urllib.parse.urlencode({
            # Request parameters
            'showStats': 'false',
        })
        body = json.dumps({'documents': [{'id': str(i), 'text': x, 'language': 'en'} for i, x in enumerate(exs)]})
        try:
            conn = http.client.HTTPSConnection('westus2.api.cognitive.microsoft.com')
#             conn.request("POST", "/text/analytics/v2.1/sentiment?%s" % params, body, headers)
            conn.request("POST", "/text/analytics/v3.1-preview.3/sentiment?opinionMining=false%s" % params, body, headers)
            response = conn.getresponse()
            azureresp = response.read()
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
            print(exs)
        try:
            pps = np.array([[x['confidenceScores'][a] for a in ['negative', 'neutral', 'positive']] for x in json.loads(azureresp)['documents']])
        except:
            print(json.loads(azureresp))
            raise Exception()
        return pps

    def predict_proba_azurea(self, exs):
        print('Azure: predicting %d examples' % len(exs))
        headers = {
        # Request headers
        'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.key,
        }
        params = urllib.parse.urlencode({
            # Request parameters
            'showStats': 'false',
        })
        body = json.dumps({'documents': [{'id': str(i), 'text': x, 'language': 'en'} for i, x in enumerate(exs)]})
        try:
            conn = http.client.HTTPSConnection('westus2.api.cognitive.microsoft.com')
#             conn.request("POST", "/text/analytics/v2.1/sentiment?%s" % params, body, headers)
            conn.request("POST", "/text/analytics/v3.1-preview.3/sentiment?opinionMining=false%s" % params, body, headers)
            response = conn.getresponse()
            azureresp = response.read()
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
            print(exs)
        if not json.loads(azureresp)['documents']:
            print(json.loads(azureresp))
            raise Exception()
        try:
            pps = np.array([[x['confidenceScores'][a] for a in ['negative', 'neutral', 'positive']] for x in json.loads(azureresp)['documents']])
        except:
            print(json.loads(azureresp))
            raise Exception()
        return pps

class Translator:
    def __init__(self, pred_file, key, from_language='en', to_language='pt-br', batch_size=1000, wait_time=0):
        self.preds = {}
        self.post_fn = None
        self.key = key
        if os.path.exists(pred_file):
            self.preds = pickle.load(open(pred_file, 'rb'))
        self.pred_file = pred_file
        self.pred_fn = self.translate
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.from_language = from_language
        self.to_language = to_language

    def predict_proba(self, exs):
        to_pred = [x for x in exs if x not in self.preds]
        chunked = list(chunks(to_pred, self.batch_size))
        for docs in tqdm(chunked):
            pps = self.pred_fn(docs)
            for x, translation in zip(docs, pps):
                self.preds[x] = translation
            pickle.dump(self.preds, open(self.pred_file, 'wb'))
            if self.wait_time:
                time.sleep(self.wait_time)
        ret = np.array([self.preds.get(x) for x in exs])
        if self.post_fn:
            ret = self.post_fn(ret)
        return ret

    def translate(self, exs):
        print('Azure: predicting %d examples' % len(exs))
        subscription_key = self.key
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Ocp-Apim-Subscription-Region': 'westus2',
            'Content-type': 'application/json',
            # 'X-ClientTraceId': str(uuid.uuid4())
        }
        params = {
            'api-version': '3.0',
            'from': self.from_language,
            'to': [self.to_language]
        }
        body = [{'text': x} for x in exs]
        endpoint = 'https://api.cognitive.microsofttranslator.com/translate'
        try:
            request = requests.post(endpoint, params=params, headers=headers, json=body)
            response = request.json()
            return [x['translations'][0]['text'] for x in response]
        except Exception as e:
            print('ERROR!')
            print(e)
            # print("[Errno {0}] {1}".format(e.errno, e.strerror))
            print(exs[:10])
            print(response)

class CogNER:
    def __init__(self, pred_file, key, batch_size=1000, wait_time=0):
        self.preds = {}
        self.key = key
        if os.path.exists(pred_file):
            self.preds = pickle.load(open(pred_file, 'rb'))
        self.pred_file = pred_file
        self.pred_fn = self.ner
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.post_fn = None

    def predict_proba(self, exs):
        to_pred = [x for x in exs if x not in self.preds]
        chunked = list(chunks(to_pred, self.batch_size))
        for docs in tqdm(chunked):
            pps = self.pred_fn(docs)
            for x, translation in zip(docs, pps):
                self.preds[x] = translation
            pickle.dump(self.preds, open(self.pred_file, 'wb'))
            if self.wait_time:
                time.sleep(self.wait_time)
        ret = np.array([self.preds.get(x) for x in exs])
        if self.post_fn:
            ret = self.post_fn(ret)
        return ret

    def ner(self, exs):
        print('Azure: predicting %d examples' % len(exs))
        subscription_key = self.key
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Ocp-Apim-Subscription-Region': 'westus2',
            'Content-type': 'application/json',
            # 'X-ClientTraceId': str(uuid.uuid4())
        }
        params = {}
        body = json.dumps({'documents': [{'id': str(i), 'text': x, 'language': 'en'} for i, x in enumerate(exs)]})
        conn = http.client.HTTPSConnection('westus2.api.cognitive.microsoft.com')
        try:
            conn.request("POST", "/text/analytics/v3.2-preview.1/entities/recognition/general?%s" % params, body, headers)
            response = conn.getresponse()
            azureresp = response.read()
            response = json.loads(azureresp)
            conn.close()
            return [x['entities'] for x in json.loads(azureresp)['documents']]
        except Exception as e:
            print('ERROR!')
            print(e)
            # print("[Errno {0}] {1}".format(e.errno, e.strerror))
            print(exs[:10])
            print(response)
