import adatest
from adatest._prompt_builder import PromptBuilder
import pytest

class DummyGenerator(adatest.generators.Generator):
    def __init__(self):
        super().__init__(self)

    def __call__(self, prompts, topic):
        return ["generation" for _ in range(len(prompts))]

@pytest.fixture
def test_tree_browser():
    s = adatest.Scorer(lambda x: ["" for _ in range(len(x))])
    gen = DummyGenerator()
    tree = adatest.TestTree()
    return adatest.TestTreeBrowser(
      test_tree=tree,
      scorer=s,
      generators={"dummygen": gen},
      auto_save=True,
      user="testuser",
      recompute_scores=False,
      drop_inactive_score_columns=False,
      max_suggestions=10,
      suggestion_thread_budget=0.5,
      prompt_builder=PromptBuilder(),
      active_generator="dummygen",
      starting_path="",
      score_filter=-1e10,
      topic_model_scale=0)

def test_redraw(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})