from ._test_tree import TestTree
from ._test_tree_browser import TestTreeBrowser
from ._scorer import Scorer, DummyScorer, ClassifierScorer, GeneratorScorer
from ._server import serve
from ._embedding import embed
from ._model import Model
from ._topic_model import ChainTopicModel, StandardTopicModel
from . import generators

__version__ = '0.3.0'

default_generators = {
    "abstract": TestTree(r"test_trees/abstract_capabilities.csv")
}
text_embedding_model = None
image_embedding_model = None