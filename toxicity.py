import sys
from tqdm.auto import tqdm
import os
import pickle
import urllib
import json
import http
import numpy as np
import time
import requests
from googleapiclient import discovery
import json



class ToxicityModel:
    def __init__(self, pred_file, key):
        self.preds = {}
        self.key = key
        if os.path.exists(pred_file):
            self.preds = pickle.load(open(pred_file, 'rb'))
        self.pred_file = pred_file
        self.perspective_calls = np.zeros(100)
        self.perspective_idx = 0
        self.client = discovery.build(
          "commentanalyzer",
          "v1alpha1",
          developerKey=key,
          discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
          static_discovery=False,
        )

    def predict_single(self, text):
        analyze_request = {
          'comment': { 'text': text},
          'requestedAttributes': {'TOXICITY': {}},
           'doNotStore': True,
            'languages': ['en']
        }
        if text in self.preds:
            return self.preds[text]
        while ((time.time() - self.perspective_calls) < 60).sum() >= 55:
            time.sleep(1)
        ctime = time.time()
        self.perspective_calls[self.perspective_idx] = ctime
        self.perspective_idx = (self.perspective_idx + 1) % 100
        response = self.client.comments().analyze(body=analyze_request).execute()
        toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
        self.preds[text] = toxicity
        if self.perspective_idx == 50:
            pickle.dump(self.preds, open(self.pred_file, 'wb'))
        return self.preds[text]

    def predict_proba(self, exs, silent=False):
        ret = []
        for doc in tqdm(exs, disable=silent):
            try:
                ret.append(self.predict_single(doc))
            except:
                ret.append(np.nan)
        ret = np.array(ret).reshape(-1, 1)
        # ret = [[{'label' : 'Non-toxic', 'score': 1 - i[0]}, {'label' : 'Toxic', 'score':i[0] }] for i in ret]

        # return ret
        ret = np.hstack((1 - ret, ret))
        return ret
