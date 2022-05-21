import pandas as pd
import uuid
import logging
import random
import os
import io
import time
import numpy as np
import pandas as pd
from ._prompt_builder import PromptBuilder
from ._test_tree_browser import TestTreeBrowser, is_subtopic
from ._model import Model
from ._topic_model import TopicModel
import adatest
from pathlib import Path
import torch

# class TestTreeIterator():
#     def __init__(self, test_tree):
#         self.test_tree = test_tree
#         self.position = 0

#     def __next__(self):
#         if self.position >= len(self.test_tree):
#             raise StopIteration
#         else:
#             self.position += 1
#             return self.test_tree.iloc[self.position - 1]

class TestTree():
    """ A hierarchically organized set of tests represented as a DataFrame.

    This represents a hierarchically organized set of tests that all target a specific class of models (such as sentiment
    analysis models, or translation models). To interact with a test tree you can use either the `__call__` method to
    view and create tests directly in a Jupyter notebook, or you can call the `serve` method to launch a standalone
    webserver. A TestTree object also conforms to most of the standard pandas DataFrame API.
    """

    def __init__(self, tests=None, index=None, compute_embeddings=False, **kwargs):
        """ Create a new test tree.

        Parameters
        ----------
        tests : str or DataFrame or list or tuple or None
            The tests to load as a test tree. If a string is provided, it is assumed to be a path to a CSV file containing
            the tests. If tests is a tuple of two elements, it is assumed to be a dataset of (data, labels) which will be used to build a test tree.
            Otherwise tests is passed to the pandas DataFrame constructor to load the tests as a DataFrame.

        index : list or list-like or None
            Assigns an index to underlying tests frame, or auto generates if not provided.

        compute_embeddings: boolean
            If True, use the global adatest.embed to build embeddings of tests in the TestTree.

        kwargs : dict
            Additional keyword arguments are passed to the pandas DataFrame constructor.
        """

        # the canonical ordered list of test tree columns
        column_names = ['topic', 'input', 'output', 'label', 'labeler', 'description']

        # create a new test tree in memory
        if tests is None:
            self._tests = pd.DataFrame([], columns=column_names, dtype=str)
            self._tests_location = None

        # create a new test tree on disk (lazily saved)
        elif isinstance(tests, str) and not os.path.isfile(tests):
            self._tests = pd.DataFrame([], columns=column_names)
            self._tests_location = tests

        # load the test tree from a file or IO stream
        elif isinstance(tests, str) or isinstance(tests, io.TextIOBase):
            self._tests_location = tests
            if os.path.isfile(tests) or isinstance(tests, io.TextIOBase):
                self._tests = pd.read_csv(tests, index_col=0, dtype=str, keep_default_na=False)
            else:
                raise Exception(f"The provided tests file is not supported: {tests}")

        elif isinstance(tests, tuple) and len(tests) == 2: # Dataset loader TODO: fix this for topic models
            # column_names = ['topic', 'type', 'value1', 'value2', 'value3', 'author', 'description', \
            # 'model value1 outputs', 'model value2 outputs', 'model value3 outputs', 'model score']

            self._tests = pd.DataFrame(columns=column_names)
            self._tests_location = None

            self._tests['input'] = tests[0]
            self._tests['output'] = tests[1]

            # Constants
            self._tests['topic'] = ''
            self._tests['label'] = ''
            self._tests['labeler'] = "dataset"
            self._tests['description'] = ''

        else:
            if index is None:
                index = [uuid.uuid4().hex for _ in range(len(tests))]
            self._tests = pd.DataFrame(tests, **kwargs)
            self._tests.index = index
            self._tests_location = None

        # # ensure auto saving is possible when requested
        # if auto_save and self._tests_location is None:
        #     raise Exception("auto_save=True is only supported when loading from a file or IO stream")
        # self.auto_save = auto_save

        # ensure we have required columns
        for c in ["input", "output", "label"]:
            if c not in self._tests.columns:
                raise Exception("The test tree being loaded must contain a '"+c+"' column!")

        # fill in any other missing columns
        for column in ["topic", "description"]:
            if column not in self._tests.columns:
                self._tests[column] = ["" for _ in range(self._tests.shape[0])]
        if "labeler" not in self._tests.columns:
            self._tests["labeler"] = ["imputed" for _ in range(self._tests.shape[0])]

        # ensure that all topics have a topic_marker entry
        marked_topics = {t: True for t in set(self._tests.loc[self._tests["label"] == "topic_marker"]["topic"])}
        for topic in set(self._tests["topic"]):
            parts = topic.split("/")
            for i in range(1, len(parts)+1):
                parent_topic = "/".join(parts[:i])
                if parent_topic not in marked_topics:
                    self._tests.loc[uuid.uuid4().hex] = {
                        "label": "topic_marker",
                        "topic": parent_topic,
                        "labeler": "imputed",
                        "input": "",
                        "output": "",
                        "description": ""
                    }
                    marked_topics[parent_topic] = True

        # drop any duplicate index values
        self._tests = self._tests.groupby(level=0).first()

        # drop any duplicate rows
        self._tests.drop_duplicates(["topic", "input", "output", "labeler"], inplace=True)

        # put the columns in a consistent order
        self._tests = self._tests[column_names + [c for c in self._tests.columns if c not in column_names]]

        if compute_embeddings:
            self.compute_embeddings()

        self._topic_models = {}

        # # keep track of our original state
        # if self.auto_save:
        #     self._last_saved_tests = self._tests.copy()

    def compute_embeddings(self):
        """ Compute the embeddings for all test cases in the test tree.

        TODO: Shared logic with TestTreeBrowser._compute_embeddings_and_scores. Refactor! (SML: I think it will live here now)
        """

        # see what new embeddings we need to compute
        all_strings = []
        for id, test in self._tests.iterrows():
            if test.label == "topic_marker":
                parts = test.topic.rsplit("/", 1)
                str = parts[1] if len(parts) == 2 else ""
                all_strings.append(str)
            else:
                for str in [test.input, test.output]:
                    all_strings.append(str)

        # suggestions topics don't have topic markers so we check for them separately
        all_strings.append("__suggestions__")
        
        # we don't use the output of the embedding, just do this to get the embeddings cached
        adatest.embed(all_strings)

        # compute the embeddings
        # if len(new_strings) > 0:
        #     string_embeddings = adatest.embed(new_strings).cpu()
        #     for i,s in enumerate(new_strings):
        #         adatest._embedding_cache[s] = string_embeddings[i]

        # compute the embeddings
        # if len(new_embedding_ids) > 0:
        #     inputs = []
        #     outputs = []
        #     for id in new_embedding_ids:
        #         if self._tests.loc[id, "label"] == "topic_marker":
        #             parts = self._tests.loc[id, "topic"].rsplit("/", 1)
        #             inputs.append(parts[1] if len(parts) == 2 else "")
        #             outputs.append("")
        #         else:
        #             inputs.append(str(self._tests.loc[id, "input"]))
        #             outputs.append(str(self._tests.loc[id, "output"]))
        #     new_strings = list(set(new_strings))
        #     string_embeddings = adatest.embedding_model.encode(strings, convert_to_tensor=True, show_progress_bar=False).cpu()
        #     for i,s in enumerate(strings):
        #         adatest._embedding_cache[s] = string_embeddings[i]


    def __getitem__(self, key):
        """ TestSets act just like a DataFrame when sliced. """
        subset = self._tests[key]
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            return self.__class__(subset, index=subset.index)
        return subset

    def __setitem__(self, key, value):
        """ TestSets act just like a DataFrame when sliced, including assignment. """
        self._tests[key] = value

    # all these methods directly expose the underlying DataFrame API
    @property
    def loc(self):
        return TestTreeLocIndexer(self)
    @property
    def iloc(self):
        return TestTreeILocIndexer(self)
    @property
    def index(self):
        return self._tests.index
    @property
    def columns(self):
        return self._tests.columns
    @property
    def shape(self):
        return self._tests.shape
    @property
    def iterrows(self):
        return self._tests.iterrows
    @property
    def groupby(self):
        return self._tests.groupby
    @property
    def drop(self):
        return self._tests.drop
    @property
    def insert(self):
        return self._tests.insert
    @property
    def copy(self):
        return self._tests.copy
    
    # NOTE: Can't delegate to df.append as it is depricated in favor of pd.concat, which we can't use due to type checks 
    def append(self, test_tree, axis=0):
        if isinstance(test_tree, pd.DataFrame):
            self._tests = pd.concat([self._tests, test_tree], axis=axis)
        elif isinstance(test_tree, TestTree):
            self._tests = pd.concat([self._tests, test_tree._tests], axis=axis)

        self.deduplicate()
        self.compute_embeddings()
        return None # TODO: Rethink append logic -- return copy vs. in place update?

    def __len__(self):
        return self._tests.__len__()
    def __setitem__(self, key, value):
        return self._tests.__setitem__(key, value)
    def to_csv(self, file=None):
        no_suggestions = self._tests.loc[["/__suggestions__" not in topic for topic in self._tests["topic"]]]
        if file is None:
            no_suggestions.to_csv(self._tests_location)
        else:
            no_suggestions.to_csv(file)

    def topic(self, topic):
        """ Return a subset of the test tree containing only tests that match the given topic.

        Parameters
        ----------
        topic : str
            The topic to filter the test tree by.
        """
        ids = [id for id, test in self._tests.iterrows() if is_subtopic(topic, test.topic)]
        return self.loc[ids]

    def __call__(self, scorer=None, generator=None, auto_save=False, user="anonymous", recompute_scores=False, drop_inactive_score_columns=False,
                 max_suggestions=100, suggestion_thread_budget=0.5, prompt_builder=PromptBuilder(), active_generator="default", starting_path="",
                 score_filter=-1e10, topic_model_scale=0): # TODO: remove active_generator and replace with the ability to set the generator?
        """ Apply this test tree to a scorer/model and browse/edit the tests to adapt them to the target model.

        Applying a test tree to a target model (wrapped by a scorer) creates a TestTreeBrowser object that can be used to
        browse the tree and add new tests to adapt it to the target model.
        
        Parameters
        ----------
        scorer : adatest.Scorer or callable
            The scorer (that wraps a model) to used to score the tests. If a function is provided, it will be wrapped in a scorer.
            Passing a dictionary of scorers will score multiple models at the same time. Note that the models are expected to take
            a list of strings as input, and output either a classification probability vector or a string.

        generator : adatest.Generator or dict[adatest.Generators]
            A source to generate new tests from. Currently supported generator types are language models, existing test trees, or datasets.

        auto_save : bool
            Whether to automatically save the test tree after each edit.

        user : str
            The user name to author new tests with.

        recompute_scores : bool
            Whether to recompute the scores of the tests that already have score values in the test tree.

        drop_inactive_score_columns : bool
            Whether to drop the score columns in the test tree that do not match any of the passed scorers.

        max_suggestions : int
            The maximum number of suggestions to generate each time the user asks for test suggestions.

        suggestion_thread_budget : float
            This controls how many parallel suggestion processes to use when generating suggestions. A value of 0 means we create
            no parallel threads (i.e. we use a single thread), 0.5 means we create as many parallel threads as possible without
            increase the number of tokens we process by more than 50% (1.5 would mean 150%, etc.). Each thread process will use a
            different randomized LM prompt for test generation, so more threads will result in more diversity, but come at the cost
            of reading more prompt variations.

        prompt_builder : adatest.PromptBuilder
            A prompt builder to use when generating prompts for new tests. This object controls how the LM prompts
            are created when generating new tests.

        active_generator : "default", or a key name if generators is a dictionary
            Which generator from adatest.generators to use when generating new tests. This should always be set to "default" if
            generators is just a single generator and not a dictionary of generators.

        starting_path : str
            The path to start browsing the test tree from.
        """

        # build the test tree browser
        return TestTreeBrowser(
            self,
            scorer=scorer,
            generators=generator,
            auto_save=auto_save,
            user=user,
            recompute_scores=recompute_scores,
            drop_inactive_score_columns=drop_inactive_score_columns,
            max_suggestions=max_suggestions,
            suggestion_thread_budget=suggestion_thread_budget,
            prompt_builder=prompt_builder,
            active_generator=active_generator,
            starting_path=starting_path,
            score_filter=score_filter,
            topic_model_scale=topic_model_scale
        )

    def __repr__(self):
        return self._tests.__repr__()

    def _repr_html_(self):
        return self._tests._repr_html_()

    def deduplicate(self):
        """ Remove duplicate tests from the test tree.
        
        Note that we give precendence to the first test in a set of duplicates.
        """

        already_seen = {}
        drop_ids = []
        for id, test in self._tests.iterrows():
            k = test.topic + "|_ADA_JOIN_|" + test.input + "|_ADA_JOIN_|" + test.output
            if k in already_seen:
                drop_ids.append(id)
            else:
                already_seen[k] = True
        self._tests = self._tests.drop(drop_ids, axis=0)

    def impute_labels(self):
        """ Impute missing labels in the test tree. """
        # TODO: this is just a random mock, it needs to implement real local topic models

        self.compute_embeddings()

        for id, test in self._tests.iterrows():
            if test.label == "":
                features = np.hstack(adatest.embed([test.input, test.output]))
                self._tests.loc[id, "label"] = self.topic_model(test.topic)(features)
                self._tests.loc[id, "labeler"] = "imputed"

    def predict_labels(self, topical_io_pairs):
        """ Return the label probabilities for a set of input-output pairs.

        Parameters
        ----------
        io_pairs : list[(str, str)]
            A list of input-output pairs to score.

        Returns
        -------
        list[float]
            A list of label probabilities.
        """

        out = np.zeros(len(topical_io_pairs))

        to_embed = []
        topics = {}
        for i,(topic,input,output) in enumerate(topical_io_pairs):
            if topic not in topics:
                topics[topic] = []
            to_embed.append(input)
            to_embed.append(output)
            topics[topic].append((i, len(to_embed) - 2, len(to_embed) - 1))
        embeddings = adatest.embed(to_embed)
        features = [None for i in range(len(topical_io_pairs))]
        for topic in topics:
            features = []
            for i,ind1,ind2 in topics[topic]:
                features.append(np.hstack([embeddings[ind1], embeddings[ind2]]))
            features = np.vstack(features)

            label = np.array([v == "pass" for v in self.topic_model(topic)(features)], dtype=np.float32)
            for i, (j,_,_) in enumerate(topics[topic]):
                out[j] = label[i]

        return np.array(out)




    def topic_model(self, topic):
        if topic not in self._topic_models:
            self._topic_models[topic] = TopicModel(topic, self)
        return self._topic_models[topic]

    def retrain_topic_model(self, topic):
        self._topic_models[topic] = TopicModel(topic, self)

    def drop_topic(self, topic):
        """ Remove a topic from the test tree. """
        self._tests = self._tests.loc[self._tests["topic"] != topic]

class TestTreeLocIndexer():
    def __init__(self, test_tree):
        self.test_tree = test_tree

    def __repr__(self):
        return "TestTreeLocIndexer is an intermediate object for operating on TestTrees. Slice this object further to yield useful results."

    def __getitem__(self, key):
        # If all columns haven't changed, it's still a valid test tree
        # If columns have been dropped, return a Pandas object
        
        subset = self.test_tree._tests.loc[key]
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            test_tree_slice = TestTree(subset)
            test_tree_slice._tests_location = self.test_tree._tests_location
            return test_tree_slice
        else:
            return subset
    
    def __setitem__(self, key, value):
        self.test_tree._tests.loc[key] = value
    
class TestTreeILocIndexer():
    def __init__(self, test_tree):
        self.test_tree = test_tree

    def __repr__(self):
        return "TestTreeILocIndexer is an intermediate object for operating on TestTrees. Slice this object further to yield useful results."

    def __getitem__(self, key):
        # If all columns haven't changed, it's still a valid test tree
        # If columns have been dropped, return a Pandas object
        
        subset = self.test_tree._tests.iloc[key]
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            test_tree_slice = TestTree(subset)
            test_tree_slice._tests_location = self.test_tree._tests_location
            return test_tree_slice
        else:
            return subset
    
    def __setitem__(self, key, value):
        self.test_tree._tests.iloc[key] = value

def _test_tree_from_dataset(X, y, model=None, time_budget=60, min_samples=100):
    column_names = ['topic', 'type' , 'value1', 'value2', 'value3', 'author', 'description', \
        'model value1 outputs', 'model value2 outputs', 'model value3 outputs', 'model score']

    test_frame = pd.DataFrame(columns=column_names)

    if model is None: # All we can do without a model defined at this stage.
        test_frame['value1'] = X
        test_frame['type'] = "{} should output {}"
        test_frame['value2'] = y

        # Constants
        test_frame['topic'] = ''
        test_frame['author'] = "dataset"
        test_frame['description'] = ''

        return TestTree(test_frame)
    
    if not isinstance(model, Model):
        model = Model(model)

    # Validate output types
    output_names = model.output_names   
    unknown_labels = set(y) - set(output_names)
    assert len(unknown_labels) == 0, f"Unknown labels found: {unknown_labels}. \
    Please update the label vector or output names property."

    # Time how long inference takes on a single sample
    try:
        start = time.time()
        _ = model(X[0:1])
        end = time.time()
    except Exception as e: # TODO: Improve this message
        raise ValueError(f"Training data cannot be evaluated by model. Error recieved: {e}.")

    # Ensure min_samples <= n_samples <= len(data) and computes in {time_budget} seconds
    n_samples = int(min(max(time_budget // (end - start), min_samples), len(X)))

    if n_samples < len(X):
        print(f"Only using {n_samples} samples to meet time budget of {time_budget} seconds.")
        # TODO: unify input types
        sample_indices = np.random.choice(np.arange(len(X)), n_samples, replace=False)
        X = [X[sample] for sample in sample_indices]
        y = [y[sample] for sample in sample_indices]

    # Build intermediate convenience frame
    df = pd.DataFrame(columns=['sample', 'label', 'label_proba', \
                                        'pred', 'pred_proba', 'largest_error', 'largest_error_proba'])
    df['sample'] = X
    df['label'] = y

    # model's current prediction
    raw_model_output = model(X)
    pred_indices = np.argsort(raw_model_output, axis=1)
    
    df['pred_proba'] = raw_model_output[range(len(pred_indices)), pred_indices[:, -1]]
    df['pred'] = [output_names[i] for i in pred_indices[:, -1]]

    label_lookup = {output:index for index, output in enumerate(output_names)}
    label_indices = [label_lookup[label] for label in y]
    df['label_proba'] = raw_model_output[range(len(label_indices)), label_indices]
    
    correct_predictions = df['pred'] == df['label']
    mispredictions = ~correct_predictions
    
    # For mispredicted samples, the largest error is the current prediction.
    df.loc[mispredictions, 'largest_error'] = df.loc[mispredictions, 'pred']
    df.loc[mispredictions, 'largest_error_proba'] = df.loc[mispredictions, 'pred_proba']
    
    # For correct samples, we use the 2nd highest class as the largest error.
    largest_errors = pred_indices[correct_predictions][:, -2]
    df.loc[correct_predictions, 'largest_error'] = [output_names[i] for i in largest_errors]
    df.loc[correct_predictions, 'largest_error_proba'] = raw_model_output[range(len(largest_errors)), largest_errors]

    df.index = [uuid.uuid4().hex for _ in range(len(df))]

    # If we have a scorer, we prefer to format tests as {X} should not output {largest_error}
    test_frame['value1'] = df['sample']
    test_frame['type'] = "{} should not output {}"
    test_frame['value2'] = df['largest_error']

    # Constants
    test_frame['topic'] = ''
    test_frame['author'] = "dataset"
    test_frame['description'] = ''

    test_frame.index = df.index
    
    return TestTree(test_frame, index=test_frame.index)