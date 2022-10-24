from ._test_tree import TestTree
from ._test_tree_browser import TestTreeBrowser
from ._scorer import Scorer, DummyScorer, ClassifierScorer, GeneratorScorer, RawScorer
from ._server import serve
from .embedders import _embed as embed
from ._model import Model
from ._topic_model import ChainTopicModel, StandardTopicModel
from . import generators

__version__ = '0.3.5'

default_generators = {
    "abstract": TestTree(r"test_trees/abstract_capabilities.csv")
}
text_embedding_model = None
image_embedding_model = None