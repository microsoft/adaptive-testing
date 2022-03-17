import pandas as pd
import uuid
import logging
import os
import io
from ._prompt_builder import PromptBuilder
from ._test_tree_browser import TestTreeBrowser, is_subtopic

class TestTree():
    """ A hierarchically organized set of tests represented as a DataFrame.

    This represents a hierarchically organized set of tests that all target a specific class of models (such as sentiment
    analysis models, or translation models). To interact with a test tree you can use either the `__call__` method to
    view and create tests directly in a Jupyter notebook, or you can call the `serve` method to launch a standalone
    webserver. A TestTree object also conforms to most of the standard pandas DataFrame API.
    """

    def __init__(self, tests=None, index=None, **kwargs):
        """ Create a new test tree.

        Parameters
        ----------
        tests : str or DataFrame or list or None
            The tests to load as a test tree. If a string is provided, it is assumed to be a path to a CSV file containing
            the tests. Otherwise tests is passed to the pandas DataFrame constructor to load the tests as a DataFrame.

        index : list or list-like or None
            Assigns an index to underlying tests frame, or auto generates if not provided.

        kwargs : dict
            Additional keyword arguments are passed to the pandas DataFrame constructor.
        """

        # the canonical ordered list of test tree columns
        column_names = ['topic', 'type', 'value1', 'value2', 'value3', 'author', 'description']

        # create a new test tree in memory
        if tests is None:
            self._tests = pd.DataFrame([], columns=column_names)
            self._tests_location = None

        # create a new test tree on disk (lazily saved)
        elif isinstance(tests, str) and not os.path.isfile(tests):
            self._tests = pd.DataFrame([], columns=column_names)
            self._tests_location = tests

        # load the test tree from a file or IO stream
        elif isinstance(tests, str) or isinstance(tests, io.TextIOBase):
            self._tests_location = tests
            if os.path.isfile(tests) or isinstance(tests, io.TextIOBase):
                self._tests = pd.read_csv(tests, index_col=0, dtype={
                    "topic": str, "type": str, "value1": str, "value2": str,
                    "value3": str, "author": str, "description": str
                    }, keep_default_na=False
                )
            else:
                raise Exception(f"The provided tests file does not exist: {tests}. If you wish to create a new file use `auto_save=True`")

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

        # ensure we at least have a type column
        if "type" not in self._tests.columns:
            raise Exception("The test tree being loaded must contain a 'type' column!")

        # fill in any other missing columns
        for column in ["topic", "value1", "value2", "value3", "description"]:
            if column not in self._tests.columns:
                self._tests[column] = ["" for _ in range(self._tests.shape[0])]
        if "author" not in self._tests.columns:
            self._tests["author"] = ["anonymous" for _ in range(self._tests.shape[0])]

        # ensure that all topics have a topic_marker entry
        marked_topics = {t: True for t in set(self._tests.loc[self._tests["type"] == "topic_marker"]["topic"])}
        for topic in set(self._tests["topic"]):
            parts = topic.split("/")
            for i in range(1, len(parts)+1):
                parent_topic = "/".join(parts[:i])
                if parent_topic not in marked_topics:
                    self._tests.loc[uuid.uuid4().hex] = {
                        "type": "topic_marker",
                        "topic": parent_topic,
                        "author": "anonymous",
                        "value1": "",
                        "value2": "",
                        "value3": "",
                        "description": ""
                    }
                    marked_topics[parent_topic] = True

        # drop any duplicate index values
        self._tests = self._tests.groupby(level=0).first()

        # drop any duplicate rows
        self._tests.drop_duplicates(["topic", "type", "value1", "value2", "value3"], inplace=True)

        # put the columns in a consistent order
        self._tests = self._tests[column_names + [c for c in self._tests.columns if c not in column_names]]

        # # keep track of our original state
        # if self.auto_save:
        #     self._last_saved_tests = self._tests.copy()

    def __getitem__(self, key):
        """ TestSets act just like a DataFrame when sliced. """
        subset = self._tests[key]
        if hasattr(subset, 'columns') and len(set(["type", "topic", "value1", "value2", "value3"]) - set(subset.columns)) == 0:
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
    def __len__(self):
        return self._tests.__len__()
    def __setitem__(self, key, value):
        return self._tests.__setitem__(key, value)
    def to_csv(self, file=None):
        if file is None:
            self._tests.to_csv(self._tests_location)
        else:
            self._tests.to_csv(file)

    def topic(self, topic):
        """ Return a subset of the test tree containing only tests that match the given topic.

        Parameters
        ----------
        topic : str
            The topic to filter the test tree by.
        """
        ids = [id for id, test in self._tests.iterrows() if is_subtopic(topic, test.topic)]
        return self.loc[ids]

    def __call__(self, scorer=None, dataset=None, auto_save=False, user="anonymous", recompute_scores=False, drop_inactive_score_columns=False,
                 max_suggestions=100, suggestion_thread_budget=0.5, prompt_builder=PromptBuilder(), active_backend="default", starting_path="",
                 embedding_model=None, score_filter=-1e10, topic_model_scale=0):
        """ Apply this test tree to a scorer/model and browse/edit the tests to adapt them to the target model.

        Applying a test tree to a target model (wrapped by a scorer) creates a TestTreeBrowser object that can be used to
        browse the tree and add new tests to adapt it to the target model.
        
        Parameters
        ----------
        scorer : adatest.Scorer or callable
            The scorer (that wraps a model) to used to score the tests. If a function is provided, it will be wrapped in a scorer.
            Passing a dictionary of scorers will score multiple models at the same time. Note that the models are expected to take
            a list of strings as input, and output either a classification probability vector or a string.

        dataset : adatest.Dataset
            A dataset to use when suggesting new tests for the test tree.

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

        active_backend : "default", or a key name if adatest.backend is a dictionary
            Which backend from adatest.backend to use when generating new tests. This should always be set to "default" if
            adatest.backend is just a single backend and not a dictionary of backends.

        starting_path : str
            The path to start browsing the test tree from.

        embedding_model : sentencetransformer.EmbeddingModel
            A SentenceTransformer embedding model to use for semantic similarity.
        """

        # build the test tree browser
        return TestTreeBrowser(
            self,
            scorer=scorer,
            dataset=dataset,
            auto_save=auto_save,
            user=user,
            recompute_scores=recompute_scores,
            drop_inactive_score_columns=drop_inactive_score_columns,
            max_suggestions=max_suggestions,
            suggestion_thread_budget=suggestion_thread_budget,
            prompt_builder=prompt_builder,
            active_backend=active_backend,
            starting_path=starting_path,
            embedding_model=embedding_model,
            score_filter=score_filter,
            topic_model_scale=topic_model_scale
        )

    def __repr__(self):
        return self._tests.__repr__()

    def _repr_html_(self):
        return self._tests._repr_html_()

class TestTreeLocIndexer():
    def __init__(self, test_tree):
        self.test_tree = test_tree

    def __repr__(self):
        return "TestTreeLocIndexer is an intermediate object for operating on TestTrees. Slice this object further to yield useful results."

    def __getitem__(self, key):
        # If all columns haven't changed, it's still a valid test tree
        # If columns have been dropped, return a Pandas object
        
        subset = self.test_tree._tests.loc[key]
        if hasattr(subset, 'columns') and len(set(["type", "topic", "value1", "value2", "value3"]) - set(subset.columns)) == 0:
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
        if hasattr(subset, 'columns') and len(set(["type", "topic", "value1", "value2", "value3"]) - set(subset.columns)) == 0:
            test_tree_slice = TestTree(subset)
            test_tree_slice._tests_location = self.test_tree._tests_location
            return test_tree_slice
        else:
            return subset
    
    def __setitem__(self, key, value):
        self.test_tree._tests.iloc[key] = value