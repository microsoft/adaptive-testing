import adatest
import pytest
import logging
from adatest._prompt_builder import PromptBuilder

logger = logging.getLogger("conftest")

class DummyGenerator(adatest.generators.Generator):
    def __init__(self):
        super().__init__(self)
        self.gen_type = "model"
        self.call_count = 0

    def __call__(self, prompts, topic, test_type=None, scorer=None, num_samples=1, max_length=100):
        logger.debug("DummyGenerator called with arguments %s", locals())
        self.call_count += 1
        logger.debug("DummyGenerator call_count %s", self.call_count)
        return ["generation" for _ in range(len(prompts))]

class DummyComm():
    def __init__(self):
        self.data = None
    
    def send(self, data):
        self.data = data

@pytest.fixture
def test_tree_browser():
    s = adatest.Scorer(lambda x: [f"{i}" for i in range(len(x))])
    gen1 = DummyGenerator()
    gen2 = DummyGenerator()
    tree = adatest.TestTree(r"test_trees/test_samples.csv")
    browser = adatest.TestTreeBrowser(
      test_tree=tree,
      scorer=s,
      generators={"gen1": gen1, "gen2": gen2},
      auto_save=False,
      user="testuser",
      recompute_scores=False,
      drop_inactive_score_columns=False,
      max_suggestions=10,
      suggestion_thread_budget=0.5,
      prompt_builder=PromptBuilder(),
      active_generator="gen1",
      starting_path="/Books/Nonfiction/Biography",
      score_filter=-1e10,
      topic_model_scale=0)
    browser.comm = DummyComm()
    return browser
