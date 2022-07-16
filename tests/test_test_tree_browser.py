import adatest
from adatest._prompt_builder import PromptBuilder
import pytest

import logging
logger = logging.getLogger("testlogger")

class DummyGenerator(adatest.generators.Generator):
    def __init__(self):
        super().__init__(self)
        self.gen_type = "model"

    def __call__(self, prompts, topic, test_type=None, scorer=None, num_samples=1, max_length=100):
        logger.info("DummyGenerator called with arguments %s", locals())
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
    tree = adatest.TestTree(r"test_trees/animals.csv")
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
      starting_path="/Mammals/Cats",
      score_filter=-1e10,
      topic_model_scale=0)
    browser.comm = DummyComm()
    return browser

def test_redraw(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert 'browser' in test_tree_browser.comm.data

def test_generate_suggestions(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "generate_suggestions"})
    logger.info(test_tree_browser.comm.data)
    assert 'browser' in test_tree_browser.comm.data
    assert 'suggestions' in test_tree_browser.comm.data['browser']
    assert len(test_tree_browser.comm.data['browser']['suggestions']) > 0

# def test_generate_suggestions_filter(test_tree_browser):
#     # TODO: Review with Harsha, is this supposed to work?
#     test_tree_browser.interface_event({"event_id": "generate_suggestions", "filter": "filter"})
#     logger.info(test_tree_browser.comm.data)
#     assert 'browser' in test_tree_browser.comm.data
#     assert 'suggestions' in test_tree_browser.comm.data['browser']
#     assert len(test_tree_browser.comm.data['browser']['suggestions']) == 0

def test_change_topic(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "change_topic", "topic": "/Mammals/Pigs"})
    assert 'browser' in test_tree_browser.comm.data
    assert 'topic' in test_tree_browser.comm.data['browser']
    assert test_tree_browser.comm.data['browser']['topic'] == "/Mammals/Pigs"

def test_clear_suggestions(test_tree_browser):
    # Generate suggestions, then clear them
    test_tree_browser.interface_event({"event_id": "generate_suggestions"})
    assert 'browser' in test_tree_browser.comm.data
    assert 'suggestions' in test_tree_browser.comm.data['browser']
    assert len(test_tree_browser.comm.data['browser']['suggestions']) > 0

    test_tree_browser.interface_event({"event_id": "clear_suggestions"})
    assert 'browser' in test_tree_browser.comm.data
    assert 'suggestions' in test_tree_browser.comm.data['browser']
    assert len(test_tree_browser.comm.data['browser']['suggestions']) == 0

def test_add_new_topic(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "add_new_topic"})
    assert 'browser' in test_tree_browser.comm.data
    assert 'tests' in test_tree_browser.comm.data['browser']
    assert '/Mammals/Cats/New topic' in test_tree_browser.comm.data['browser']['tests']

def test_add_new_test(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    pre_len = len(test_tree_browser.comm.data['browser']['tests'])
    test_tree_browser.interface_event({"event_id": "add_new_test"})
    assert 'browser' in test_tree_browser.comm.data
    assert 'tests' in test_tree_browser.comm.data['browser']
    post_len = len(test_tree_browser.comm.data['browser']['tests'])
    assert post_len == pre_len + 1

def test_set_first_model(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "set_first_model", "model": "testmodel"})

def test_change_generator(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data['browser']['active_generator'] == "gen1"
    test_tree_browser.interface_event({"event_id": "change_generator", "generator": "gen2"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data['browser']['active_generator'] == "gen2"

def test_change_mode(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_mode = test_tree_browser.comm.data['browser']['mode']
    test_tree_browser.interface_event({"event_id": "change_mode", "mode": "testmode"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data['browser']['mode'] != orig_mode
    assert test_tree_browser.comm.data['browser']['mode'] == "testmode"

def test_change_description(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_desc = test_tree_browser.comm.data['browser']['topic_description']
    test_tree_browser.interface_event({
        "event_id": "change_description",
        "topic_marker_id": test_tree_browser.comm.data['browser']['topic_marker_id'],
        "description": "test"
    })
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data['browser']['topic_description'] != orig_desc
    assert test_tree_browser.comm.data['browser']['topic_description'] == "test"

def test_change_filter(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "change_filter", "filter_text": "filter"})

def test_move_test(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_test = test_tree_browser.comm.data['browser']['tests'][0]
    test_tree_browser.interface_event({"event_id": "move_test", "test_ids": [orig_test], "topic": "foo"})
    assert orig_test not in test_tree_browser.comm.data['browser']['tests']

def test_delete_test(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_test = test_tree_browser.comm.data['browser']['tests'][0]
    orig_len = len(test_tree_browser.comm.data['browser']['tests'])
    test_tree_browser.interface_event({"event_id": "delete_test", "test_ids": [orig_test]})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert orig_len == len(test_tree_browser.comm.data['browser']['tests']) + 1

def test_change_label(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_test = test_tree_browser.comm.data['browser']['tests'][0]
    test_tree_browser.interface_event({"event_id": "change_label", "test_ids": [orig_test], "label": "pass"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data[orig_test]['label'] == "pass"
    test_tree_browser.interface_event({"event_id": "change_label", "test_ids": [orig_test], "label": "fail"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data[orig_test]['label'] == "fail"

def test_change_input(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_test = test_tree_browser.comm.data['browser']['tests'][0]
    test_tree_browser.interface_event({"event_id": "change_input", "test_ids": [orig_test], "input": "testinput"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data[orig_test]['input'] == "testinput"

def test_change_output(test_tree_browser):
    test_tree_browser.interface_event({"event_id": "redraw"})
    orig_test = test_tree_browser.comm.data['browser']['tests'][0]
    test_tree_browser.interface_event({"event_id": "change_output", "test_ids": [orig_test], "output": "testoutput"})
    test_tree_browser.interface_event({"event_id": "redraw"})
    assert test_tree_browser.comm.data[orig_test]['output'] == "testoutput"

# test_change_label(test_tree_browser=test_tree_browser())