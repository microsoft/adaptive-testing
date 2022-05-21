import time
from IPython.display import display, HTML
import openai
import numpy as np
import copy
import math
import sentence_transformers
import pandas as pd
import torch
import json
import re
import collections

from transformers import MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING
from adatest.generators import TestTreeSource

from .comm import JupyterComm
import uuid
import pathlib
import copy
import re
import logging
import os
import io
import statistics
import checklist
import checklist.editor
from threading import Timer
from ._scorer import expand_template, clean_template, ClassifierScorer, GeneratorScorer, Scorer
from ._prompt_builder import PromptBuilder
import builtins
import adatest # Need to import like this to prevent circular dependencies

# from https://gist.github.com/walkermatt/2871026
def throttle(interval):
    """ Decorator that will postpone a functions
        execution so it does not run more than once per
        interval of time.
    """
    def decorator(fn):
        def throttled(*args, **kwargs):
            if not hasattr(throttled, "t") or not throttled.t.is_alive():
                def call_it():
                    fn(*args, **kwargs)
                throttled.t = Timer(interval, call_it)
                throttled.t.start()
        return throttled
    return decorator

log = logging.getLogger(__name__)

# import sys
# sys.stderr = open('/tmp/err.txt', 'w')

def file_log(*args):
    """ Used for logging when we don't have a stdout.

    This is used for debugging when we are being called from the javascript client. When we are
    called from the client we don't have a stdout attached to any cell in the notebook.

    Note to also catch errors you could do:
    import sys
    sys.stderr = open('err.txt', 'w')
    """
    #print(*args)
    f = open("log.txt", "a")  # append mode
    f.write(" ".join([str(msg) for msg in args])+"\n")
    f.flush()
    f.close()

def is_subtopic(topic, candidate):
    # Returns true if candidate is a subtopic of topic
    return True if re.search(r'^%s(/|$)' % topic.replace('+', r'\+'), candidate) else False

special_outputs = [
    "{MAX}"
]

valid_comparators = [
    "should not be",
    "should be",
    "should be the same as for"
]
FILLIN_PREFIX = '/Fill-ins'


# model("this is english") => []
# output_sampling="topk(10)"
# output_sampling="topp(10)"
# output_sampling="max"
# output_sampling="temperature(0.9)"

class TestTreeBrowser():
    """ Used for browsing and expanding a test tree.
    """

    def __init__(self, test_tree, scorer, generators, user, auto_save, recompute_scores, drop_inactive_score_columns,
                 max_suggestions, suggestion_thread_budget, prompt_builder, active_generator, starting_path,
                 score_filter, topic_model_scale):
        """ Initialize the TestTreeBrowser.
        
        See the __call__ method of TreeBrowser for parameter documentation.
        """

        self.test_tree = test_tree
        self.scorer = scorer
        self.generators = generators
        self.user = user
        self.auto_save = auto_save
        self.recompute_scores = recompute_scores
        self.drop_inactive_score_columns = drop_inactive_score_columns
        self.max_suggestions = max_suggestions
        self.suggestion_thread_budget = suggestion_thread_budget
        self.prompt_builder = prompt_builder
        self.active_generator = active_generator
        self.current_topic = starting_path
        self.score_filter = score_filter
        self.topic_model_scale = topic_model_scale

        # convert single generator to the multi-generator format
        if not isinstance(self.generators, dict):
            self.generators = {'generator': self.generators}

        if adatest.default_generators is not None: # Merge default generators into generators
            self.generators = {**self.generators, **adatest.default_generators}

        # Find and cast any TestTrees in generators to TestTreeSource
        for generator_name, generator in self.generators.items():
            if isinstance(generator, adatest._test_tree.TestTree): # TODO: make this autoreload friendly
                self.generators[generator_name] = TestTreeSource(generator) 

        # get a reference to the active backend object
        if self.active_generator == "default":
            if isinstance(self.generators, dict):
                self._active_generator_obj = next(iter(self.generators.items()))[1]
            else:
                self._active_generator_obj = self.generators
        else:
            self._active_generator_obj = self.generators[self.active_generator]

        # if we are recomputing the scores then we erase all the old scores
        if recompute_scores is True:
            for c in self.test_tree.columns:
                if c.endswith("score"):
                    self.test_tree.drop(c, axis=1, inplace=True)

        # convert single scorer args to the multi-scorer format
        if callable(self.scorer):
            self.scorer = {"model": self.scorer}

        # note the score column of each scorer
        if isinstance(self.scorer, dict):
            self.score_columns = [k+" score" for k in self.scorer]
            for k in self.scorer:
                self.scorer[k] = Scorer(self.scorer[k])
        elif self.scorer is not None:
            self.score_columns = ["model score"]
            self.scorer = {"model": Scorer(self.scorer)}
        else:
            self.score_columns = []

        # find score columns that are not associated with a scorer
        for c in self.test_tree.columns:
            if c.endswith("score") and c not in self.score_columns:
                if drop_inactive_score_columns is True:
                    self.test_tree.drop(c, axis=1, inplace=True)
                else:
                    self.score_columns.append(c)

        # ensure that each scorer's score column is in the test tree dataframe
        for c in self.score_columns:
            if c not in self.test_tree.columns:
                self.test_tree[c] = ["" for _ in range(self.test_tree.shape[0])]

        # a unique identifier for this test set instance, used for UI connections
        self._id = uuid.uuid4().hex

        # these are all temporary state
        self._hidden_topics = {}
        self.comm = None

        # define our current mode, and set of supported modes
        self.mode = "tests"
        self.mode_options = [
            # "validity focused", # focus first on making valid in-topic tests, then secondarily on making those tests high scoring
            # "failure focused", # focus on making high scoring (failing) tests, then secondarily on making those tests valid and in-topic
            "tests", # suggest new tests
            "topics" # suggest new subtopics
        ]

        # apply all the scorers to the test tree (this updates the test tree)
        for k in self.scorer:
            self.scorer[k](self.test_tree, k+" score")

        # # make sure all the tests have scores (if we have a scorer)
        # self._compute_embeddings_and_scores(self.test_tree)

        # ensure any test tree based generator has embeddings calculated
        if isinstance(self.generators, dict):
            for name, gen in self.generators.items():
                if gen.gen_type == "test_tree":
                    gen.source.compute_embeddings()
        elif self.generators.gen_type == "test_tree": # HN: Probably unused code, should be safe to remove
            self.generators.source.compute_embeddings()

        # save the current state of the test tree
        self._auto_save()

        # init a blank set of suggetions
        # self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
        self._suggestions_error = "" # tracks if we failed to generate suggestions

    def _repr_html_(self, prefix="", environment="jupyter", websocket_server=None):
        """ Returns the HTML interface for this browser.

        Parameters
        ----------
        prefix : str
            The URL prefix this test tree browser is being served from.

        environment : str
            The environment this test tree browser is being served from (jupyter or web).
        """

        # spin up a JupyterComm object if we are called directly (which we assume is in a notebook)
        if self.comm is None and environment == "jupyter":
            self.comm = JupyterComm(f'adatest_interface_target_{self._id}', self.interface_event)

        # dump the client javascript to the interface
        file_path = pathlib.Path(__file__).parent.absolute()
        with open(file_path / ".." / "client" / "dist" / "main.js", encoding="utf-8") as f:
            js_data = f.read()
        interface_html = f"""
<div id="adatest_container_{self._id}" style="width: 100%; all: initial;"></div>
<script type='text/javascript'>
  {js_data};
  AdaTestReactDOM.render(
    AdaTestReact.createElement(AdaTest, {{
      interfaceId: "{self._id}", environment: "{environment}", startingTopic: "{self.current_topic}", prefix: "{prefix}",
      websocket_server: {"undefined" if websocket_server is None else '"'+websocket_server+'"'},\
    }}, null),
    document.getElementById('adatest_container_{self._id}')
  );
</script>
"""
        return interface_html

    def display(self):
        """ Manually display the HTML interface.
        """
        display(HTML(self._repr_html_()))

    def interface_event(self, msg):
        """ Handle interface events from the client.

        Parameters
        ----------
        msg : dict
            The event messages from the client. Each key in the dictionary is a separate message to either the row
            specified by the key or to whole browser object if the key is 'browser'.
        """

        log.debug(f"interface_event({msg})")

        # loop over each event message
        for k in msg:
            if k == "browser":
                action = msg[k].get("action", None)
                
                # rewdraw the entire interface
                if action == "redraw":
                    self._refresh_interface()
                
                # generate a new set of suggested tests/topics
                elif action == "generate_suggestions":
                    self._clear_suggestions()
                    self.test_tree.retrain_topic_model(self.current_topic)
                    self._generate_suggestions(filter=msg[k].get("filter", ""))
                    # if self._active_generator_obj is None:
                    #     self._suggestions_error = "No AdaTest generator has been set!"
                    # else:
                    #     self._generate_suggestions(filter=msg[k].get("filter", ""))
                        # # try:
                        # self.suggestions = self._generate_suggestions(filter=msg[k].get("filter", ""))
                        # # filter suggestions to relevant types
                        # if self.mode == "topics":
                        #     self.suggestions = self.suggestions[self.suggestions['type'] == "topic_marker"]
                        # elif self.mode == "tests":
                        #     self.suggestions = self.suggestions[self.suggestions['type'] != "topic_marker"]

                        # # Ensure valid suggestions exist.
                        # if self.suggestions.shape[0] > 0:  
                        #     self.suggestions.sort_values(self.score_columns[0], inplace=True, ascending=False, key=np.vectorize(score_max))
                        #     self._suggestions_error = ""
                        # else:
                        #     self._suggestions_error = True # Not sure if we should do this?
                        # except Exception as e:
                        #     log.debug(e)
                        #     self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                        #     self._suggestions_error = True
                    self._refresh_interface()
                
                # change the current topic
                elif action == "change_topic":
                    self.current_topic = msg[k]["topic"]
                    # self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)

                    # see if we have only topics are direct children, if so, we suggest topics, otherwise we suggest tests
                    has_direct_tests = False
                    has_known_subtopics = False
                    for k, test in self.test_tree.iterrows():
                        if test["topic"] == self.current_topic:
                            if test["label"] != "topic_marker":
                                has_direct_tests = True
                        elif is_subtopic(self.current_topic, test["topic"]):
                            has_known_subtopics = True
                    if not has_direct_tests and has_known_subtopics:
                        self.mode = "topics"
                    else:
                        self.mode = "tests"

                    self._refresh_interface()
                
                # clear the current set of suggestions
                elif action == "clear_suggestions":
                    self._clear_suggestions()
                    # self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                    self._refresh_interface()

                # add a new empty subtopic to the current topic
                elif action == "add_new_topic":
                    self.test_tree.loc[uuid.uuid4().hex] = {
                        "topic": self.current_topic + "/New topic",
                        "label": "topic_marker",
                        "input": "",
                        "output": "",
                        "labeler": self.user,
                        "description": ""
                    }
                    self._compute_embeddings_and_scores(self.test_tree)
                    self._auto_save()
                    self._refresh_interface()
                
                # add a new empty test to the current topic
                elif action == "add_new_test":

                    # add the new test row
                    row = {
                        "topic": self.current_topic,
                        "input": "New test", # The special value "New test" causes the interface to auto-select the text
                        "output": "",
                        "label": "",
                        "labeler": "imputed",
                        "description": ""
                    }
                    for c in self.score_columns:
                        row[c] = ""
                        row[c[:-6] + " raw outputs"] = "{}"
                    self.test_tree.loc[uuid.uuid4().hex] = row

                    self._compute_embeddings_and_scores(self.test_tree)
                    self._auto_save()
                    self._refresh_interface()

                # change which scorer/model is used for sorting tests
                elif action == "set_first_model":
                    name = msg[k]["model"]

                    # move to front of score columns
                    pos = len(self.test_tree.columns) - len(self.score_columns)
                    tmp = self.test_tree[name]
                    self.test_tree.drop(labels=[name], axis=1, inplace=True)
                    self.test_tree.insert(pos, name, tmp)

                    # update score columns list
                    self.score_columns.remove(name)
                    self.score_columns.insert(0, name)

                    self._auto_save()
                    self._refresh_interface()

                # change which generator is active
                elif action is None and "active_generator" in msg[k]:
                    self.active_generator = msg[k]["active_generator"]
                    self._active_generator_obj = self.generators[self.active_generator]

                # change which generator is active
                elif action is None and "mode" in msg[k]:
                    self.mode = msg[k]["mode"]

                elif action == 'change_description':
                    self.test_tree.loc[msg[k]['topic_marker_id']]['description'] = msg[k]['description']

            # if we are just updating a single row in tests then we only recompute the scores
            elif "topic" not in msg[k]:
                sendback_data = {}
                
                # convert template expansions into a standard value update
                if msg[k].get("action", "") == "template_expand":
                    template_value = self.templatize(self.test_tree.loc[k, msg[k]["value"]])
                    msg[k] = {msg[k]["value"]: template_value}
                    sendback_data[msg[k]["value"]] = template_value

                # update the row and recompute scores
                for k2 in msg[k]:
                    self.test_tree.loc[k, k2] = msg[k][k2]
                self.test_tree.loc[k, self.score_columns] = ""
                    
                self._compute_embeddings_and_scores(self.test_tree, overwrite_outputs="input" in msg[k])
                

                # send just the data that changed back to the frontend
                sendback_data["scores"] = {c: [[k, v] for v in score_parts(self.test_tree.loc[k, c])] for c in self.score_columns}
                outputs = {c: [[k, json.loads(self.test_tree.loc[k].get(c[:-6] + " raw outputs", "{}"))]] for c in self.score_columns}
                sendback_data["raw_outputs"] = outputs
                sendback_data["output"] = self.test_tree.loc[k, "output"]
                sendback_data["label"] = self.test_tree.loc[k, "label"]
                sendback_data["labeler"] = self.test_tree.loc[k, "labeler"]
                sendback_data.update(self.test_display_parts(self.test_tree.loc[k]))
                self.comm.send({k: sendback_data})
                
                self._auto_save()

            # if we are just changing the topic
            elif "topic" in msg[k] and len(msg[k]) == 1:
                
                # move a test that is in the test tree
                if k in self.test_tree.index:
                    if msg[k]["topic"] == "_DELETE_" or msg[k]["topic"] == "_OUT_OF_TOPIC_": # this means delete the test
                        self.test_tree.drop(k, inplace=True)
                    else:
                        self.test_tree.loc[k, "topic"] = msg[k]["topic"]
                        self.test_tree.loc[k, "author"] = self.user
                
                # move a whole topic around
                else:
                    for id, test in self.test_tree.iterrows():
                        if is_subtopic(k, test.topic):
                            if msg[k]["topic"] == "_DELETE_" or msg[k]["topic"] == "_OUT_OF_TOPIC_":
                                self.test_tree.drop(id, inplace=True)
                            else:
                                self.test_tree.loc[id, "topic"] = msg[k]["topic"] + test.topic[len(k):]

                # Recompute any missing embeddings to handle any changes
                self._compute_embeddings_and_scores(self.test_tree)
                self._refresh_interface()
                self._auto_save()

            else:
                log.debug(f"Unable to parse the interface message: {msg[k]}")

    def _refresh_interface(self):
        """ Send our entire current state to the frontend interface.
        """

        # get the children of the current topic
        data = {}

        def create_children(data, tests, topic):
            children = []
            
            # add tests and topics to the data lookup structure
            for k, test in tests.iterrows():
                if is_subtopic(topic, test.topic):
                    
                    # add a topic
                    if test.label == "topic_marker":
                        if is_subtopic(topic, test.topic) and test.topic != topic:
                            name = test.topic[len(topic)+1:]
                            if "/" not in name: # only add direct children
                                data[test.topic] = {
                                    "label": test.label,
                                    "labeler": test.labeler,
                                    "description": "",
                                    "scores": {c: [] for c in self.score_columns},
                                    "topic_marker_id": k,
                                    "topic_name": name,
                                    "editing": test.topic.endswith("/New topic")
                                }
                                children.append(test.topic)
                    
                    # add a test
                    else:
                        data[k] = {
                            "input": test.input,
                            "output": test.output,
                            "label": test.label,
                            "labeler": test.labeler,
                            "description": test.description,
                            "scores": {c: [[k, v] for v in score_parts(test[c])] for c in self.score_columns},
                            "editing": test.input == "New test"
                        }

                        data[k]["raw_outputs"] = {c: [[k, safe_json_load(test.get(c[:-6] + " raw outputs", "{}"))]] for c in self.score_columns}
                        data[k].update(self.test_display_parts(test))
                        if test.topic == topic:
                            children.append(k)
            
            # fill in the scores for the child topics
            for k, test in tests.iterrows():
                if "/__suggestions__" not in test.topic and is_subtopic(topic, test.topic) and test.topic != topic:
                    child_topic = test.topic[len(topic):].split("/", 2)[1]
                    scores = data[topic+"/"+child_topic]["scores"]
                    for c in self.score_columns:
                        scores[c].extend([[k, v] for v in score_parts(test[c])])

            # sort by score and always put new topics first
            def sort_key(id):
                try:
                    total = 0
                    count = 0
                    for s in data[id]["scores"][self.score_columns[0]]:
                        val = score_max(s[1], nan_val=np.nan)
                        if not np.isnan(val) and val is not None:
                            total += val
                            count += 1
                    if count == 0:
                        return 1e3
                    else:
                        return -total / count
                except Exception as e:
                    print(e)
                    print(id)
                    print(val)
            sorted_children = sorted(children, key=sort_key)
            sorted_children = sorted(sorted_children, key=lambda id: 0 if id.endswith("/New topic") or data[id].get("value1", "") == "New test" else 1)

            return sorted_children
        
        # get the children of the current topic
        children = create_children(data, self.test_tree, self.current_topic)
        suggestions_children = create_children(data, self.test_tree, self.current_topic + "/__suggestions__")

        # TODO: This is a complete hack to hide lower scoring suggestions when we are likely already in the exploit phase
        # this is just for users who don't know when to stop scrolling down...
        # SML: I expect we can delete this at some point?
        if self.score_filter == "auto":
            if len(children) < 10:
                score_filter = -1e12
            else:
                children_scores = sorted([np.max([score_max(x[1]) for x in data[key]['scores'][self.score_columns[0]]]) for key in children])
                suggestions_children_scores = sorted([np.max([score_max(x[1]) for x in data[key]['scores'][self.score_columns[0]]]) for key in suggestions_children])
                score_filter = children_scores[-5] - (children_scores[-1] - children_scores[-5]) * 0.2
                if len(suggestions_children_scores) > 0:
                    score_filter = min(score_filter, np.nanmax(suggestions_children_scores) - 1e-2)
        else:
            score_filter = self.score_filter

        # if self.scorer is not None:
        #     test_types = self.scorer[self.score_columns[0][:-6]].supported_test_types
        #     test_type_parts = {t: split_test_type(t) for t in self.scorer[self.score_columns[0][:-6]].supported_test_types}
        # else:
        #     test_types = []
        #     test_type_parts = {}

        topic_marker_id = self._get_topic_marker_id(self.current_topic)
        # compile the global browser state for the frontend
        data["browser"] = {
            "suggestions": suggestions_children,
            "tests": children,
            "user": self.user,
            "topic": self.current_topic,
            "topic_description": self.test_tree.loc[topic_marker_id]["description"] if topic_marker_id is not None else "",
            "topic_marker_id": topic_marker_id if topic_marker_id is not None else uuid.uuid4().hex,
            "score_filter": score_filter,
            "disable_suggestions": False,
            "read_only": False,
            "score_columns": self.score_columns,
            "suggestions_error": self._suggestions_error,
            "generator_options": [str(x) for x in self.generators.keys()] if isinstance(self.generators, dict) else [self.active_generator],
            "active_generator": self.active_generator,
            "mode": self.mode,
            "mode_options": self.mode_options,
            # "test_types": test_types,
            # "test_type_parts": test_type_parts,
        }

        self.comm.send(data)

    def _clear_suggestions(self):
        """ Clear the suggestions for the current topic.
        """
        ids = list(self.test_tree.index)
        for k in ids:
            if self.test_tree.loc[k, "topic"].startswith(self.current_topic + "/__suggestions__"):
                self.test_tree.drop(k, inplace=True)


    def _generate_suggestions(self, filter):
        """ Generate suggestions for the current topic.

        Parameters
        ----------
        filter : str
            The filter to apply to the tests while generating suggestions.
        """

        #--Backend-driven suggestions--

        # save a lookup we can use to detect duplicate tests
        test_map = {}
        for _, test in self.test_tree.iterrows():
            if test.label == "topic_marker":
                test_map[test.topic + " __topic_marker__"] = True
            else:
                test_map[test.topic + " __JOIN__ " + test.input] = True

        # see if we have a finite set of valid outputs
        # valid_outputs = getattr(self.scorer, "output_names", None)
        # if valid_outputs is not None and value2_filter is not None:
        #     valid_outputs = [s for s in valid_outputs if re.search(value2_filter, s) is not None]

        # # see if we have only topics are direct children, if so, we suggest topics
        # has_direct_tests = False
        # has_known_subtopics = False
        # for k, test in self.test_tree.iterrows():
        #     if test["topic"] == self.current_topic:
        #         if test["type"] == "test":
        #             has_direct_tests = True
        #     elif is_subtopic(self.current_topic, test["topic"]):
        #         has_known_subtopics = True
        # suggest_topics = not has_direct_tests and has_known_subtopics
        # suggest_topics = self.mode == "topics"
        # zero_shot_tests = not has_direct_tests and not has_known_subtopics
        
        
        
        # {} should output {}
        # {} should not output {}
        # {} should have the same output as {}
        # {} should not be more {} than {}
        # {} should be invertable. [] is the output.
        # {} should not output text containing {}
        # {} should output toxic text. [] is the output.
        # {} should not output toxic text. [] is the output.
        
        # {}'s output should not contain {}
        


        
        # validity focused (focus first on making valid in-topic tests, then secondarily on making those tests high scoring)
        # failure focused (focus on making high scoring (failing) tests, then secondarily on making those tests valid and in-topic)
        # topics (suggest new sub-topics)
        # file_name dataset (suggest tests based on samples from the provided dataset)


        # compute the maximum number of suggestion threads we can use given our suggestion_thread_budget
        p = self.prompt_builder.prompt_size
        budget = 1 + self.suggestion_thread_budget
        suggestion_threads = max(1, int(np.floor(budget * (p/(p+1) + 1/(p+1) * self.max_suggestions) - 1/(p+1) * self.max_suggestions) / (p/(p+1))))
        
        # generate the prompts for the backend
        prompts = self.prompt_builder(
            test_tree=self.test_tree,
            topic=self.current_topic,
            score_column=self.score_columns[0],
            repetitions=suggestion_threads,
            filter=filter,
            suggest_topics=self.mode == "topics"
        )

        # generate the suggestions
        proposals = self._active_generator_obj(prompts, self.current_topic, self.mode, self.scorer, num_samples=self.max_suggestions // len(prompts))
        
        # Build up suggestions catalog, unless generating from a test tree source.
        # NOTE: Doing safe checks for TestTree type in order to prevent circular imports
        if isinstance(proposals, pd.DataFrame) or proposals.__class__.__name__ == "TestTree":
            suggestions = proposals
            suggestions['topic'] = self.current_topic + "/__suggestions__" + suggestions['topic'].apply(lambda x: x[len(self.current_topic):] if x != "" else "/")
            self.test_tree.append(suggestions)
            print("appended suggestions into self.test_tree")
            # assert False, "This needs to be fixed to dump into /__suggestions__"
        else:
            # suggestions = []
            test_map_tmp = copy.copy(test_map)
            for input in proposals:
                if self.mode == "topics" and ("/" in input or "\n" in input):
                    input = input.replace("/", " or ").replace("\n", " ") # topics can't have newlines or slashes in their names
                    input = input.replace("  ", " ").strip() # kill any double spaces we may have introduced
                    str_val = self.current_topic + "/" + input + " __topic_marker__"
                else:
                    str_val = self.current_topic + " __JOIN__ " + input
                if str_val not in test_map_tmp:
                    id = uuid.uuid4().hex
                    self.test_tree.loc[id, "topic"] = self.current_topic + "/__suggestions__" + ("/"+input if self.mode == "topics" else "")
                    self.test_tree.loc[id, "input"] = "" if self.mode == "topics" else input
                    self.test_tree.loc[id, "output"] = ""
                    self.test_tree.loc[id, "label"] = "topic_marker" if self.mode == "topics" else ""
                    self.test_tree.loc[id, "labeler"] = "imputed"
                    self.test_tree.loc[id, "description"] = ""
                    for c in self.score_columns:
                        self.test_tree.loc[id, c] = ""

                    # s = {
                    #     "topic": self.current_topic + "/__suggestions__" + ("/"+input if self.mode == "topics" else ""),
                    #     "input": "" if self.mode == "topics" else input,
                    #     "output": "",
                    #     "label": "",
                    #     "labeler": "imputed",
                    #     "description": ""
                    # }
                    # for c in self.score_columns:
                    #     s[c] = ""
                    # suggestions.append(s)
                    if str_val is not None:
                        test_map_tmp[str_val] = True

            # suggestions = pd.DataFrame(suggestions, index=[uuid.uuid4().hex for _ in range(len(suggestions))], columns=self.test_tree.columns)
            self._compute_embeddings_and_scores(self.test_tree)

        # Filter invalid suggestions
        # if self.mode != "topics":
        #     suggestions = suggestions.dropna(subset=[self.score_columns[0]])

        # When we have outputs filled in by the scorer we might have more duplicates we need to remove
        # duplicates = []
        # for k,row in suggestions.iterrows():
        #     # str_val = row.topic + " " + test_type + " " + row.value1 + " " +  row.value2 + " " +  row.value3
        #     str_val = " ".join(builtins.filter(None, (row.topic, test_type, row.value1, row.value2, row.value3))) # Safely handles None
        #     if str_val in test_map:
        #         duplicates.append(k)
        #     test_map[str_val] = True
        # suggestions = suggestions.drop(duplicates)

        # if self.topic_model_scale != 0:
        #     self._add_topic_model_score(suggestions, topic_model_scale=self.topic_model_scale)
        # return suggestions

    def _get_topic_marker_id(self, topic):
        """
        Returns the id of the topic marker row for the given topic.
        Returns None if not found.
        """
        topic_marker_index_df = self.test_tree.index[(self.test_tree['topic'] == topic) & (self.test_tree['label'] == 'topic_marker')]
        topic_marker_index = topic_marker_index_df.tolist()[0] if len(topic_marker_index_df) > 0 else None
        return topic_marker_index

    def _add_topic_model_score(self, df, topic_model_scale):
        documents = []
        for k,s in df.iterrows():
            max_output = -10e8
            max_output_name = None
            for k,v in json.loads(s["score value1 outputs"]).items():
                if v > max_output:
                    max_output = v
                    max_output_name = k
            documents.append(f'"{s["value1"]}" > "{max_output_name}"')

        query = self._make_prompt(
            self.current_topic,
            prompt_size=20,
            include_value2=True
        )["prompt"]

        r = openai.Engine("davinci-instruct-beta").search(
            documents=documents,
            query=query
        )

        sim_scores = np.array([v["score"] for v in r["data"]])
        sim_scores -= np.mean(sim_scores)
        sim_scores /= np.std(sim_scores)

        for i, (k, row) in enumerate(df.iterrows()):
            row["score"] = float(row["score"]) + topic_model_scale * sim_scores[i]

    def _compute_embeddings_and_scores(self, tests, recompute=False, overwrite_outputs=False): # TODO: Rename/refactor/merge with _compute_scores?
        log.debug(f"compute_embeddings_and_scores(tests=<DataFrame shape={tests.shape}>, recompute={recompute})")

        for k in self.scorer:
            self.scorer[k](tests, k+" score", overwrite_outputs=overwrite_outputs)
        return
        for k in self.scorer:

            self.scorer[k](tests)

            # run the model on all the rows without a score
            new_ids = tests.index[tests[k+ " score"] == ""]
            outputs = self.scorer[k].outputs(tests["input"][new_ids])

        if self.scorer is not None:
            self._compute_scores(tests, recompute=recompute)

        # model outputs and embeddings
        if adatest.embedding_model is not None:
            new_embedding_ids = [k for k in tests.index if k not in adatest._embedding_cache]
            if len(new_embedding_ids) > 0:
                value1s = []
                value2s = []
                value3s = []
                for k in new_embedding_ids:
                    if tests.loc[k, "type"] == "topic_marker":
                        parts = tests.loc[k, "topic"].rsplit("/", 1)
                        value1s.append(parts[1] if len(parts) == 2 else "")
                        value2s.append("")
                        value3s.append("")
                    else:
                        value1s.append(str(tests.loc[k, "value1"]))
                        value2s.append(str(tests.loc[k, "value2"]))
                        value3s.append(str(tests.loc[k, "value3"]))
                new_value1_embeddings = adatest.embedding_model.encode(value1s, convert_to_tensor=True, show_progress_bar=False).cpu()
                new_value2_embeddings = adatest.embedding_model.encode(value2s, convert_to_tensor=True, show_progress_bar=False).cpu()
                new_value3_embeddings = adatest.embedding_model.encode(value3s, convert_to_tensor=True, show_progress_bar=False).cpu()
                for i,k in enumerate(new_embedding_ids):
                    adatest._embedding_cache[k] = np.hstack([new_value1_embeddings[i], new_value2_embeddings[i], new_value3_embeddings[i]])

    def _compute_scores(self, tests, recompute):
        """ Use the scorer(s) to fill in scores in the passed TestTree.

        Parameters
        ----------
        tests : TestTree
            The TestTree to fill in missing scores for.

        recompute : bool
            If True, recompute all scores. If False, only recompute scores that are missing.
        """
        
        log.debug(f"_compute_scores(tests=<TestTree shape={tests.shape}>, recompute={recompute})")

        # see which rows need scores computed
        if recompute or len(self.score_columns) == 0:
            new_score_mask = np.ones(tests.shape[0], dtype=np.bool)
        else:
            new_score_mask = np.array(tests[self.score_columns[0]].isnull()) | np.array(tests[self.score_columns[0]] == "")
        new_score_mask = new_score_mask & np.array(tests["label"] != "topic_marker", dtype=np.bool)

        if new_score_mask.sum() > 0:
            scores = {}
            tests_to_score = tests.loc[new_score_mask, ["topic", "input", "output", "label"]]
            
            # call the scorers
            blank_outputs = [{} for _ in range(tests_to_score.shape[0])]
            for k in self.scorer:
                scorer_output = self.scorer[k](tests_to_score)
                scores[k+" score"] = ["|".join(str(vv) for vv in v) for v in scorer_output["scores"]]
                scores[k+" value1 outputs"] = scorer_output.get("value1_outputs", blank_outputs)
                scores[k+" value2 outputs"] = scorer_output.get("value2_outputs", blank_outputs)
                scores[k+" value3 outputs"] = scorer_output.get("value3_outputs", blank_outputs)

            # copy the scores into the TestTree
            for k in scores:
                for i, j in enumerate(np.where(new_score_mask)[0]):
                    tests.loc[tests.index[j], k] = json.dumps(scores[k][i]) if isinstance(scores[k][i], dict) else scores[k][i]

            # copy outputs that may have been generated by the scorers over to the passed test tree
            for k in tests.index[new_score_mask]:
                tests.loc[k, "value1"] = tests_to_score.loc[k, "value1"]
                tests.loc[k, "value2"] = tests_to_score.loc[k, "value2"]
                tests.loc[k, "value3"] = tests_to_score.loc[k, "value3"]

    # def _load_dataset(self, time_budget=30, min_samples=100):
    #     '''Evaluate model on dataset and capture useful information.'''
    #     # TODO: Generalize to more dataset formats
    #     if self.dataset is None:
    #         return None
        
    #     model = self.scorer['model'].model

    #     # Unpack dataset object
    #     X, y = self.dataset[0], self.dataset[1]
    #     output_names = self.scorer['model'].output_names
        
    #     unknown_labels = set(y) - set(output_names)
    #     assert len(unknown_labels) == 0, f"Unknown labels found: {unknown_labels}. \
    #     Please update the label vector or output names property."

    #     # Time how long inference takes on a single sample
    #     try:
    #         start = time.time()
    #         _ = model(X[0:1])
    #         end = time.time()
    #     except Exception as e: # TODO: Improve this message
    #         raise ValueError(f"Training data cannot be evaluated by model. Error recieved: {e}.")


    #     # Ensure min_samples <= n_samples <= len(data) and computes in {time_budget} seconds
    #     n_samples = int(min(max(time_budget // (end - start), min_samples), len(X)))

    #     if n_samples < len(X):
    #         print(f"Only using {n_samples} samples to meet time budget of {time_budget} seconds.")
    #         # TODO: unify input types
    #         sample_indices = np.random.choice(np.arange(len(X)), n_samples, replace=False)
    #         X = [X[sample] for sample in sample_indices]
    #         y = [y[sample] for sample in sample_indices]

    #     # Build output frame
    #     df = pd.DataFrame(columns=['sample', 'label', 'label_proba', \
    #                                         'pred', 'pred_proba', 'largest_error', 'largest_error_proba'])
    #     df['sample'] = X
    #     df['label'] = y

    #     # model's current prediction
    #     raw_model_output = model(X)
    #     pred_indices = np.argsort(raw_model_output, axis=1)
        
    #     df['pred_proba'] = raw_model_output[range(len(pred_indices)), pred_indices[:, -1]]
    #     df['pred'] = [output_names[i] for i in pred_indices[:, -1]]

    #     label_lookup = {output:index for index, output in enumerate(output_names)}
    #     label_indices = [label_lookup[label] for label in y]
    #     df['label_proba'] = raw_model_output[range(len(label_indices)), label_indices]
        

    #     correct_predictions = df['pred'] == df['label']
    #     mispredictions = ~correct_predictions
        
    #     # For mispredicted samples, the largest error is the current prediction.
    #     df.loc[mispredictions, 'largest_error'] = df.loc[mispredictions, 'pred']
    #     df.loc[mispredictions, 'largest_error_proba'] = df.loc[mispredictions, 'pred_proba']
        
    #     # For correct samples, we use the 2nd highest class as the largest error.
    #     largest_errors = pred_indices[correct_predictions][:, -2]
    #     df.loc[correct_predictions, 'largest_error'] = [output_names[i] for i in largest_errors]
    #     df.loc[correct_predictions, 'largest_error_proba'] = raw_model_output[range(len(largest_errors)), largest_errors]

    #     df.index = [uuid.uuid4().hex for _ in range(len(df))]
    #     return df

    # def _convert_dataset_to_tests(self, dataset_frame): # TODO: Consider removing from class?
    #     '''Converts a loaded dataset into test formats.'''
        
    #     column_names = ['topic', 'type' , 'value1', 'value2', 'value3', 'author', 'description', \
    #     'model value1 outputs', 'model value2 outputs', 'model value3 outputs', 'model score']

    #     test_frame = pd.DataFrame(columns=column_names)

    #     # All tests currently formatted as not predicting the largest error.
    #     test_frame['value1'] = dataset_frame['sample']
    #     test_frame['type'] = "{} should not output {}"
    #     test_frame['value2'] = dataset_frame['largest_error']

    #     # Constants
    #     test_frame['topic'] = ''
    #     test_frame['author'] = "dataset"
    #     test_frame['description'] = ''

    #     test_frame.index = dataset_frame.index
        
    #     return test_frame # TODO: Cast this as a formal TestTree instead of dataframe

    def test_display_parts(self, test):
        
        # # find which template instantiation has the highest score (and so should be displayed)
        # score_parts = test[self.score_columns[0]].split("|")
        # if len(score_parts) == 1:
        #     max_score_ind = 0
        # else:
        #     max_score_ind = np.argmax([float(v) for v in test[self.score_columns[0]].split("|")])
        
        # if test.type == "{} should not be completed to become {}":
        #     value1_outputs = safe_json_load(test.get(self.score_columns[0][:-6] + " value1 outputs", "{}")).get("string", [""])
        #     if not isinstance(value1_outputs, list):
        #         value1_outputs = [value1_outputs]
        #     if len(value1_outputs) == 1:
        #         value1_outputs_str = value1_outputs[0]
        #     else:
        #         value1_outputs_str = "{" + "|".join(value1_outputs) + "}"

        #     # hispanic / 95f08d610df74679afc1217f03b9c1b3.template[0] "this is a test of[]" should not be completed to become "toxic"
        #     # {}

        #     out = []
        #     for i,option in enumerate(expand_template(test.value1 + "|||" + value1_outputs_str + "|||" + test.value2, keep_braces=False)):
        #         value1_disp,d_text1b,value2_disp = option.split("|||")
        #         out.append({
        #             "d_text1a": '"',
        #             "d_value1": "{}",
        #             "value1_disp": value1_disp,
        #             "d_text1b": d_text1b + '"',
        #             "d_text2a": '"',
        #             "d_value2": "{}",
        #             "value2_disp": value2_disp,
        #             "d_text2b": '"',
        #             "d_text3a": '',
        #             "d_value3": "",
        #             "d_text3b": ''
        #         })
        # else: # this is the default two-value test format that only varies in the select value
        out = [{
            "d_text1a": '"',
            "d_value1": "{}",
            "d_text1b": '"',
            "d_text2a": '"',
            "d_value2": "{}",
            "d_text2b": '"',
            "d_text3a": '',
            "d_value3": "",
            "d_text3b": ''
        }]
        
        return {
            "display_parts": out
        }

    def templatize(self, s):
        prompt = """INPUT: "Where are regular people on Twitter"
    OUTPUT: "Where are {regular|normal|sane|typical} people on {Twitter|Facebook|Reddit|Instagram}"
    ###
    INPUT: "Anyone who says this food tastes horrible is out of their mind"
    OUTPUT: "{Anyone|Someone|He|She} who says this food tastes {horrible|terrible|rotten} is out of their mind"
    ###
    INPUT: "great"
    OUTPUT: "{great|excellent|wonderful|superb|delightful}"
    ###
    INPUT: "If you haven't come here before, you probably live under a rock"
    OUTPUT: "If you haven't come here {before|in the past|before now|yet}, you probably {live under a rock|never get out|are a hermit|are isolated}"
    ###
    INPUT: "Only crazy people would say they had a lousy time"
    OUTPUT: "Only {crazy people|insane people|people with no sense|those out of their mind} would say they had a {lousy|terrible|bad|miserable} time"
    ###
    INPUT: "If I didn't come here again, I would be very happy for the rest of my life"
    OUTPUT: "If I didn't come {here|hereabouts|around here} {again|once more|all over again}, I would be very {happy|glad|pleased|elated} for the rest of my life"
    ###
    INPUT: "I don't know what James was talking about when they said they loved the food."
    OUTPUT: "I don't know what {James|John|Robert|Steve|Bob} was talking about when they {said they|stated that|claimed that|mentioned that} they {loved|liked|adored|appreciated} the food."
    ###
    INPUT: "new_input_value"
    OUTPUT: \""""
        prompt = prompt.replace("new_input_value",  s)
        response = openai.Completion.create(
            engine=self.engine, prompt=prompt, max_tokens=300,
            temperature=0.7, n=4, stop="\""
        )

        lines = [choice["text"] for choice in response["choices"]]
        options = []
        for line in lines:
            line = clean_template(line)
            valid = False
            for option in expand_template(line):
                if option == s:
                    valid = True
                    break
            if valid:
                options.append((-len(line), line))
        options.sort()
        log.debug(f"options = {options}")
        return options[0][1]
    
    def _auto_save(self):
        """ Save the current state of the model if we are auto saving.
        """
        if self.auto_save:
            self.test_tree.to_csv()

def score_max(s, nan_val=-1e3):
    if s == "" or s is None:
        return nan_val
    elif isinstance(s, str):
        return np.max([convert_float(v) for v in s.split("|")])
    elif np.isnan(s):
        return nan_val
    else:
        return np.max(s)

def score_parts(s):
    if isinstance(s, str):
        return [convert_float(v) for v in s.split("|")]
    else:
        return [s]

def convert_float(s):
    if s == "":
        return np.nan
    try:
        f = float(s)
    except ValueError:
        f = np.nan
    return f

def safe_json_load(input):
    if isinstance(input, float): # catch NaN's
        return {}
    else:
        return json.loads(input)

def split_test_type(test_type):
    part_names = ["text1", "value1", "text2", "value2", "text3", "value3", "text4"]
    parts = re.split(r"(\{\}|\[\])", test_type)
    part_values = ["" for _ in range(7)]
    for i, part in enumerate(parts):
        part_values[i] = part
    return {name: value for name,value in zip(part_names, part_values)}



def safe_mode(l):
    """ This just silences the error from a double mode from python <= 3.7.
    """
    try:
        return statistics.mode(l)
    except:
        return l[0]