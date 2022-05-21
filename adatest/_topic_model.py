import sklearn
import numpy as np
from sklearn import multioutput
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
import adatest
import re


def is_subtopic(topic, candidate):
    # Returns true if candidate is a subtopic of topic
    return True if re.search(r'^%s(/|$)' % topic.replace('+', r'\+'), candidate) else False


class TopicModel:
    def __init__(self, topic, test_tree):
        self.topic = topic
        self.test_tree = test_tree

        self.test_tree.compute_embeddings()

        # compute the weighting of samples for each test for this topic
        # TODO: make this code shared correctly with prompt builder
        parts = topic.split("/")
        topic_scaling = np.ones(test_tree.shape[0])
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i+1])
            topic_scaling *= 1 + 9 * np.array([v.startswith(prefix) for v in test_tree["topic"]])

        # collect the embeddings
        # TODO: like in prompt builder we should have a max of how many samples we work with
        null_embedding = np.hstack(adatest.embed(["", ""])) 
        embeddings = []
        labels = []
        for i, (id, test) in enumerate(test_tree.iterrows()):
            if test.labeler == "imputed" or test.label == "topic_marker":
                topic_scaling[i] = 0
                embeddings.append(null_embedding)
                labels.append("pass")
            else:
                embeddings.append(np.hstack(adatest.embed([test.input, test.output])))
                labels.append(test.label)
        embeddings = np.vstack(embeddings)

        # normalize the weights
        topic_scaling /= topic_scaling.max()

        # fit our topic model
        self.model = LinearSVC()
        self.model.fit(embeddings, labels, sample_weight=topic_scaling)

    def __call__(self, embeddings):
        if not hasattr(embeddings[0], "__len__"):
            return self.model.predict([embeddings])[0]
        return self.model.predict(embeddings)

class ChainTopicModel:
    def __init__(self, model=None):
        if model is None:
            self.base_model = RidgeClassifierCV()
        else:
            self.base_model = model
    def fit(self, X, y):
        topics = y
        max_levels = max([len(x.split('>')) for x in topics])
        self.model = sklearn.multioutput.ClassifierChain(self.base_model, order=list(range(max_levels)))
        y = [list(map(str.strip, x.split('>'))) for x in topics]
        y = np.array([x + ['-'] * (max_levels - len(x)) for x in y])
        self.encoders = [preprocessing.LabelEncoder() for _ in range(max_levels)]
        self.possible_topics = set()
        for x in topics:
            self.possible_topics.add(x)
            a = x.split(' > ')
            for i in range(1, len(a)):
                self.possible_topics.add(' > '.join(a[:i]))

        self.classes_ = list(self.possible_topics)
        new_y = np.zeros(y.shape)
        for i in range(y.shape[1]):
            self.encoders[i].fit(y[:, i])
            new_y[:, i] = self.encoders[i].transform(y[:, i])
        self.model.fit(X, new_y)
    def predict(self, X):
        y = self.model.predict(X)
        ret = []
        for i in range(y.shape[1]):
            ret.append(self.encoders[i].inverse_transform(y[:, i].astype(int)))
        y = np.array(ret).T
        ret = []
        for x in y:
            x = [z for z in x if z != '-']
            a = ' > '.join(x)
            while a not in self.possible_topics:
                x = x[:-1]
                a = ' > '.join(x)
            ret.append(a)
        return np.array(ret)

    def predict_proba(self, X):
        # This is just a fake function for now, puts 1 in the predicted class and 0 elsewhere
        y = self.predict(X)
        ret = np.zeros((len(X), len(self.classes_)))
        for i, r in enumerate(y):
            ret[i, self.classes_.index(r)] = 1
        return ret

class StandardTopicModel:
    def __init__(self, threshold=0.5):
        self.model= sklearn.linear_model.RidgeClassifierCV()
        self.threshold=threshold
        # add the missing predict_proba method to RidgeClassifierCV
        def predict_proba(self, X):
            if len(self.classes_) == 1:
                return np.ones((len(X), 1))
            d = self.decision_function(X)
            if len(self.classes_) == 2:
                probs = np.exp(d) / (np.exp(d) + np.exp(-d))
                return np.array([1 - probs, probs]).T
            probs = np.exp(d).T / np.sum(np.exp(d), axis=1)
            return probs.T
        self.model.predict_proba = predict_proba.__get__(self.model, self.model.__class__)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def predict(self, X):
        if self.threshold is None:
            return self.model.predict(X)
        pps = self.model.predict_proba(X)
        zero_index = list(self.model.classes_).index('Not problematic')
        ret = []
        for p in pps:
            if p[zero_index] >= self.threshold:
                ret.append(self.model.classes_[zero_index])
                continue
            else:
                best = np.argsort(p)
                if best[-1] == zero_index:
                    best = best[:-1]
                ret.append(self.model.classes_[best[-1]])
        return np.array(ret)
        # return self.model.predict(X)
