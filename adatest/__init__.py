from ._test_tree import TestTree
from ._test_tree_browser import TestTreeBrowser
from ._scorer import Scorer, DummyScorer, ClassifierScorer, GeneratorScorer
from ._server import serve
from ._model import Model
from ._topic_model import ChainTopicModel, StandardTopicModel
from . import generators
from sentence_transformers import SentenceTransformer # necessary to initialize global

__version__ = '0.1.0'

default_generators = {"abstract": TestTree(r"test_trees/abstract_capabilities.csv")}
embedding_model = SentenceTransformer('stsb-roberta-base')
_embedding_cache = {}
