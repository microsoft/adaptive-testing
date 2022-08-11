import sklearn
import numpy as np
from sklearn import multioutput
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import adatest
import re

class ConstantModel():
    def __init__(self, label):
        self.label = label
    def predict(self, embeddings):
        if not hasattr(embeddings[0], "__len__"):
            return self.label
        else:
            return [self.label] * len(embeddings)

class OutputNearestNeighborLabelModel():
    def __init__(self, embeddings, labels):
        embeddings[:,:embeddings.shape[1]//2] = 0 # zero out the embedding for the input value so we only depend on the output
        self.model = sklearn.neighbors.KNeighborsClassifier(1)
        self.model.fit(embeddings, labels)
    def predict(self, embeddings):
        embeddings[:,:embeddings.shape[1]//2] = 0
        return self.model.predict(embeddings)

class TagSetLabelingModel:
    def __init__(self, tags, test_tree):
        self.tags = tags
        self.tags_list = tags.split(":")
        self.test_tree = test_tree

        # mask out entries that do not have a pass/fail label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "tag_marker") | (test_tree["label"] == "off_topic"))
        
        # try and select samples from the current tag set
        tags_mask = test_tree.has_exact_tags(tags) & valid_mask
        
        # if we didn't find enough samples then expand to include subtags
        if tags_mask.sum() <= 1:
            tags_mask = test_tree.has_tags(tags) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent tags
        # we do this by going up the tree of "/Topic/..." tags and seeing if we can find any samples
        new_tag_list = self.tags_list[:]
        while tags_mask.sum() <= 1 and len(new_tag_list) > 0:
            found_something_to_shorten = False
            for i in range(len(new_tag_list)):
                if new_tag_list[i].startswith("/Topic/"):
                    new_tag_list[i] = new_tag_list[i].rsplit("/", 1)[0]
                    found_something_to_shorten = True
                    break
            if not found_something_to_shorten:
                break
            tags_mask = test_tree.has_tags(new_tag_list) & valid_mask

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][tags_mask]) + list(test_tree["output"][tags_mask])
        labels = list(test_tree["label"][tags_mask])
        unrolled_embeds = adatest.embed(strings)
        embeddings = np.hstack([unrolled_embeds[:len(labels)], unrolled_embeds[len(labels):]])

        # empty test tree
        if len(labels) == 0:
            self.model = ConstantModel("Unknown")

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(labels[0])
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            # TODO: SML: It seems to me that the SVC seems to do very well as long as there are no "errors" in the data labels. But it will
            # do very poorly if there are errors in the data labels since it will fit them exactly. Perhaps we can help this by
            # ensembling several SVCs together each trained on a different bootstrap sample? This might add the roubustness (against label mismatches)
            # that is lacking with hard-margin SVC fitting (it is also motivated a bit by the connections between SGD and hard-margin SVC fitting, and that
            # in practice SGD works on subsamples of the data so it should be less sensitive to label misspecification).
            self.model = LinearSVC()
            # self.model = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000)
            self.model.fit(embeddings, labels)

    def __call__(self, input, output):
        embeddings = np.hstack(adatest.embed([input, output]))
        if not hasattr(embeddings[0], "__len__"):
            return self.model.predict([embeddings])[0]
        return self.model.predict(embeddings)

class TagMembershipModel:
    """ A model that predicts if a given test fits in a given tag set.

    Note that this model only depends on the inputs not the output values for a test.
    """
    def __init__(self, tag, test_tree):
        self.tag = tag
        self.test_tree = test_tree

        # mask out entries that do not have a topic membership label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "tag_marker"))
        
        # try and select samples from the current tag
        tag_mask = test_tree.has_exact_tag(tag) & valid_mask
        
        # if we didn't find enough samples then expand to include subtags
        if tag_mask.sum() <= 1:
            tag_mask = test_tree.has_tag(tag) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent tags
        # we do this by going up the tree of "/Topic/..." tags and seeing if we can find any samples
        new_tag = self.tag
        while tags_mask.sum() <= 1 and new_tag.startswtih("/"):
            new_tag = new_tag.rsplit("/", 1)[0]
            tags_mask = test_tree.has_tag(new_tag) & valid_mask

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][tag_mask])
        labels = [l if l == "off_topic" else "on_topic" for l in test_tree["label"][tag_mask]]
        embeddings = np.array(adatest.embed(strings))

        # empty test tree
        if len(labels) == 0:
            self.model = ConstantModel("Unknown")

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(labels[0])
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            self.model = LinearSVC()
            self.model.fit(embeddings, labels)

    def __call__(self, input):
        embeddings = adatest.embed([input])[0]
        if not hasattr(embeddings[0], "__len__"):
            return self.model.predict([embeddings])[0]
        return self.model.predict(embeddings)
