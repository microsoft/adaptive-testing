import sklearn
import numpy as np
from sklearn import multioutput
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import adaptivetesting
import re

class ConstantModel():
    def __init__(self, probability):
        self.probability = probability
    def predict_prob(self, embeddings):
        if not hasattr(embeddings[0], "__len__"):
            return self.probability
        else:
            return [self.probability] * len(embeddings)

class CVModel():
    def __init__(self, embeddings, labels):
        self.inner_model = RidgeClassifierCV(class_weight={"pass": 1, "fail": 1})
        self.inner_model.fit(embeddings, labels)

    def predict_prob(self, embeddings):
        assert len(self.inner_model.classes_) == 2
        d = self.inner_model.decision_function(embeddings)
        probs = np.exp(d) / (np.exp(d) + np.exp(-d))

        return probs

class OutputNearestNeighborLabelModel():
    def __init__(self, embeddings, labels):
        embeddings[:,:embeddings.shape[1]//2] = 0 # zero out the embedding for the input value so we only depend on the output
        self.model = sklearn.neighbors.KNeighborsClassifier(1)
        self.model.fit(embeddings, labels)
    def predict(self, embeddings):
        embeddings[:,:embeddings.shape[1]//2] = 0
        return self.model.predict(embeddings)

class TopicLabelingModel:
    def __init__(self, topic, test_tree):
        self.topic = topic
        self.test_tree = test_tree

        # mask out entries that do not have a pass/fail label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "topic_marker") | (test_tree["label"] == "off_topic"))
        
        # try and select samples from the current topic
        topic_mask = (test_tree["topic"] == topic) & valid_mask
        
        # if we didn't find enough samples then expand to include subtopics
        if topic_mask.sum() <= 1:
            topic_mask = test_tree["topic"].str.startswith(topic) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent topics
        parts = topic.split("/")
        for i in range(len(parts), 0, -1):
            prefix = "/".join(parts[:i+1])
            if topic_mask.sum() <= 1:
                topic_mask = test_tree["topic"].str.startswith(prefix) & valid_mask
            else:
                break

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][topic_mask]) + list(test_tree["output"][topic_mask])
        labels = list(test_tree["label"][topic_mask])
        unrolled_embeds = adaptivetesting.embed(strings)
        embeddings = np.hstack([unrolled_embeds[:len(labels)], unrolled_embeds[len(labels):]])

        # empty test tree
        if len(labels) == 0:
            self.model = ConstantModel(0.0)

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(0.0 if labels[0] == "pass" else 1.0)
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            # TODO: SML: It seems to me that the SVC seems to do very well as long as there are no "errors" in the data labels. But it will
            # do very poorly if there are errors in the data labels since it will fit them exactly. Perhaps we can help this by
            # ensembling several SVCs together each trained on a different bootstrap sample? This might add the roubustness (against label mismatches)
            # that is lacking with hard-margin SVC fitting (it is also motivated a bit by the connections between SGD and hard-margin SVC fitting, and that
            # in practice SGD works on subsamples of the data so it should be less sensitive to label misspecification).
            # self.model = LinearSVC()

            # self.model = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000)

            # This seemed to be reasonably well calibrated on simple tests, so we use it instead of SVC
            self.model = CVModel(embeddings, labels)

            # # add the missing predict_proba method to the base model
            # def predict_proba(self, X):
            #     if len(self.classes_) == 1:
            #         return np.ones((len(X), 1))
            #     d = self.decision_function(X)
            #     if len(self.classes_) == 2:
            #         probs = np.exp(d) / (np.exp(d) + np.exp(-d))
            #         return np.array([1 - probs, probs]).T
            #     probs = np.exp(d).T / np.sum(np.exp(d), axis=1)
            #     return probs.T
            # self.model.predict_proba = predict_proba.__get__(self.model, self.model.__class__)
            
            # self.model.fit(embeddings, labels)

    def __call__(self, input, output):
        embeddings = np.hstack(adaptivetesting.embed([input, output]))
        if not hasattr(embeddings[0], "__len__"):
            return self.model.predict_prob([embeddings])[0]
        return self.model.predict_prob(embeddings)

class TopicMembershipModel:
    """ A model that predicts if a given test fits in a given topic.

    Note that this model only depends on the inputs not the output values for a test.
    """
    def __init__(self, topic, test_tree):
        self.topic = topic
        self.test_tree = test_tree

        # mask out entries that do not have a topic membership label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "topic_marker"))
        
        # try and select samples from the current topic
        topic_mask = (test_tree["topic"] == topic) & valid_mask
        
        # if we didn't find enough samples then expand to include subtopics
        if topic_mask.sum() <= 1:
            topic_mask = test_tree["topic"].str.startswith(topic) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent topics
        parts = topic.split("/")
        for i in range(len(parts), 0, -1):
            prefix = "/".join(parts[:i+1])
            if topic_mask.sum() <= 1:
                topic_mask = test_tree["topic"].str.startswith(prefix) & valid_mask
            else:
                break

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][topic_mask])
        labels = [l if l == "off_topic" else "on_topic" for l in test_tree["label"][topic_mask]]
        embeddings = np.array(adaptivetesting.embed(strings))

        # empty test tree (default to on-topic)
        if len(labels) == 0:
            self.model = ConstantModel(1.0)

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(0.0 if labels[0] == "off_topic" else 1.0)
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            self.model = CVModel()
            self.model.fit(embeddings, labels)

    def __call__(self, input):
        embeddings = adaptivetesting.embed([input])[0]
        if not hasattr(embeddings[0], "__len__"):
            return "on_topic" if self.model.predict_prob([embeddings])[0] > 0.5 else "off_topic"
        return ["on_topic" if v > 0.5 else "off_topic" for v in self.model.predict_prob(embeddings)]

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
