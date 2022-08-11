from ._test_tree import TestTree
from ._test_tree_browser import TestTreeBrowser
from ._scorer import Scorer, DummyScorer, ClassifierScorer, GeneratorScorer
from ._server import serve
from ._embedding import embed
from ._model import Model
from ._topic_model import ChainTopicModel, StandardTopicModel
from . import generators

__version__ = '0.3.2'

default_generators = {
    "abstract": TestTree(r"test_trees/abstract_capabilities.csv")
}
text_embedding_model = None
image_embedding_model = None

# Set up custom logging level for User Study
# Inspired by Antoine Pitrou: https://bugs.python.org/issue31732#msg307164

import logging

logging.STUDY = logging.ERROR + 5 # Set to logging.INFO + 5 for warnings + errors captured too
logging.addLevelName(logging.STUDY, "STUDY")
def study(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    self._log(logging.STUDY, message, args, **kws) 
logging.Logger.study = study