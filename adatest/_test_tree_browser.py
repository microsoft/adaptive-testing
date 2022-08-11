import numpy as np
import copy
import pandas as pd
import json
import re
from tqdm import tqdm

from adatest.generators import TestTreeSource

from .comm import JupyterComm
import uuid
import pathlib
import copy
import re
import logging
import statistics
from threading import Timer
from ._scorer import expand_template, clean_template, Scorer
import adatest # Need to import like this to prevent circular dependencies
import urllib.parse
from .utils import has_tag

log = logging.getLogger(__name__)

def matches_filter(test, filter_text):
    if filter_text is None or filter_text == "":
        return True
    else:
        return filter_text in test["input"] or filter_text in test["output"]

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
        self.current_tags = starting_path
        self.score_filter = score_filter
        self.topic_model_scale = topic_model_scale
        self.filter_text = ""

        # convert single generator to the multi-generator format
        if not isinstance(self.generators, dict):
            self.generators = {'generator': self.generators}

        # merge any default generators into generators
        if adatest.default_generators is not None:
            self.generators = {**self.generators, **adatest.default_generators}

        # Find and cast any TestTrees in generators to TestTreeSource
        for generator_name, generator in self.generators.items():
            if isinstance(generator, adatest._test_tree.TestTree): # TODO: make this autoreload friendly
                self.generators[generator_name] = TestTreeSource(generator) 

        # get a reference to the active backend object
        if self.active_generator == "default":
            self._active_generator_obj = next(iter(self.generators.items()))[1]
        else:
            self._active_generator_obj = self.generators[self.active_generator]

        # if we are recomputing the scores then we erase all the old scores
        if recompute_scores is True:
            for c in self.test_tree.columns:
                if c.endswith("score"):
                    self.test_tree.drop(c, axis=1, inplace=True)

        # convert single scorer args to the multi-scorer format
        if not isinstance(self.scorer, dict):
            self.scorer = {} if self.scorer is None else {"model": self.scorer}

        # note the score column of each scorer
        self.score_columns = [k+" score" for k in self.scorer]
        for k in self.scorer:
            self.scorer[k] = Scorer(self.scorer[k])

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
                self.test_tree[c] = ["__TOEVAL__" for _ in range(self.test_tree.shape[0])]

        # a unique identifier for this test set instance (used for UI connections)
        self._id = uuid.uuid4().hex

        # these are all temporary state
        self._hidden_topics = {}
        self.comm = None

        # apply all the scorers to the test tree (this updates the test tree)
        self._compute_scores(self.test_tree, self.recompute_scores, overwrite_outputs=False, save_outputs=True)

        # save the current state of the test tree
        self._auto_save()

        # track if we failed to generate suggestions
        self._suggestions_error = ""

    def auto_optimize(self, rounds=10, tags=["/Topic", "/Expectation"]):
        """ Run the testing loop for a given set of tags without user involvement.
        
        Note that this assumes the labeling model and membership model are both always correct.
        """

        for _ in tqdm(list(range(rounds))):
    
            # create new suggestions in the topic
            self.generate_suggestions(tags)
            
            # get the ids of the on-topic and off-topic suggestions
            keep_ids = []
            drop_ids = []
            mask = self.test_tree.has_exact_tags(["/__suggestion__"] + tags)
            tag_match_ids = self.test_tree.index[mask]
            for k in tag_match_ids:
                test = self.test_tree.loc[k]
                main_score = test[self.score_columns[0]]
                if test.label != "off_topic" and not isinstance(main_score, str) and not np.isnan(main_score):
                    keep_ids.append(k)
                else:
                    drop_ids.append(k)
            
            # label and move these top suggestions to the root topic
            self.test_tree.loc[keep_ids, "labeler"] = "auto_optimize"
            self.test_tree.loc[keep_ids, "tags"] = ":".join(tags)
            self.test_tree.drop(drop_ids, inplace=True)

    def _repr_html_(self, prefix="", environment="jupyter", websocket_server=None):
        """ Returns the HTML interface for this browser.

        Parameters
        ----------
        prefix : str
            The URL prefix this test tree browser is being served from.

        environment : str
            The environment this test tree browser is being served from (jupyter or web).

        websocket_server : str
            The websocket server URL the client should connect to.
        """

        # spin up a JupyterComm object if we are called directly (which we assume is in a notebook)
        if self.comm is None and environment == "jupyter":
            self.comm = JupyterComm(f'adatest_interface_target_{self._id}', self.interface_event)

        # dump the client javascript to the interface
        file_path = pathlib.Path(__file__).parent.absolute()
        with open(file_path / "resources" / "main.js", encoding="utf-8") as f:
            js_data = f.read()
        interface_html = f"""
<div id="adatest_container_{self._id}" style="width: 100%; all: initial;"></div>
<script type='text/javascript'>
  {js_data};
  AdaTestReactDOM.render(
    AdaTestReact.createElement(AdaTest, {{
      interfaceId: "{self._id}", environment: "{environment}", startingTags: "{self.current_tags}", prefix: "{prefix}",
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
        from IPython.display import display, HTML
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

            # messages to the whole browser object
            if k == "browser":
                action = msg[k].get("action", None)
                
                # rewdraw the entire interface
                if action == "redraw":
                    self._refresh_interface()
                
                # generate a new set of suggested tests/topics
                elif action == "generate_suggestions":
                    self._clear_suggestions()
                    self.test_tree.retrain_topic_labeling_model(self.current_tags)
                    self.test_tree.retrain_topic_membership_model(self.current_tags)
                    self._generate_suggestions(filter=msg[k].get("filter", ""))
                    self._refresh_interface()
                
                # change the current set of tags
                elif action == "set_current_tags":
                    self.current_tags = msg[k]["tags"]
                    self._refresh_interface()
                
                # clear the current set of suggestions
                elif action == "clear_suggestions":
                    self._clear_suggestions()
                    self._refresh_interface()

                # add a new empty subtopic to the current topic
                elif action == "create_new_tag":
                    parent_tag = msg[k]["parent_tag"]
                    self.test_tree.loc[uuid.uuid4().hex] = {
                        "tags": parent_tag + "/New tag",
                        "label": "tag_marker",
                        "input": "",
                        "output": "",
                        "labeler": self.user,
                        "description": ""
                    }
                    self._compute_scores(self.test_tree)
                    self._auto_save()
                    self._refresh_interface()
                
                # add a new empty test to the current topic
                elif action == "create_new_test":

                    # add the new test row
                    row = {
                        "tags": ":".join(self.current_tags),
                        "input": "New test", # The special value "New test" causes the interface to auto-select the text
                        "output": "",
                        "label": "",
                        "labeler": "imputed",
                        "description": ""
                    }
                    for c in self.score_columns:
                        row[c] = "__TOEVAL__"
                        row[c[:-6] + " raw outputs"] = "{}"
                    self.test_tree.loc[uuid.uuid4().hex] = row

                    self._compute_scores(self.test_tree)
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

                # change the description string for a tag set
                elif action == 'change_description':
                    id = msg[k]['tag_marker_id']
                    if id not in self.test_tree.index:
                        self.test_tree.loc[id, 'topic'] = "" # only the root topic would be missing from the tree
                        self.test_tree.loc[id, 'input'] = ""
                        self.test_tree.loc[id, 'output'] = ""
                        self.test_tree.loc[id, 'label'] = "tag_marker"
                    self.test_tree.loc[id, 'description'] = msg[k]['description']
                    self._auto_save()

                # change the current filter string
                elif action == 'change_filter':
                    self.filter_text = msg[k]['filter_text']
                    self._refresh_interface()


            # if we are just updating a single row in tests then we only recompute the scores
            elif "tags" not in msg[k]:
                sendback_data = {}
                
                # convert template expansions into a standard value update
                if msg[k].get("action", "") == "template_expand":
                    template_value = self.templatize(self.test_tree.loc[k, msg[k]["value"]])
                    msg[k] = {msg[k]["value"]: template_value}
                    sendback_data[msg[k]["value"]] = template_value

                # update the row and recompute scores
                for k2 in msg[k]:
                    self.test_tree.loc[k, k2] = msg[k][k2]
                if "input" in msg[k] or "output" in msg[k]:
                    self.test_tree.loc[k, self.score_columns] = "__TOEVAL__"
                    self._compute_scores(self.test_tree, overwrite_outputs="output" not in msg[k])
                elif "label" in msg[k]:
                    # SML: we could recompute the scores here but then that would change the output of stochastic output models
                    #      ...unless we cache the output of the model in the row, which we might need to update the scores plots anyway
                    pass 

                # send just the data that changed back to the frontend
                sendback_data["scores"] = {c: [[k, v] for v in ui_score_parts(self.test_tree.loc[k, c], self.test_tree.loc[k, "label"])] for c in self.score_columns}
                outputs = {c: [[k, json.loads(self.test_tree.loc[k].get(c[:-6] + " raw outputs", "{}"))]] for c in self.score_columns}
                sendback_data["raw_outputs"] = outputs
                if "output" not in msg[k]: # if the output was given to us the client is managing its current state so we shouldn't send it back
                    sendback_data["output"] = self.test_tree.loc[k, "output"]
                sendback_data["label"] = self.test_tree.loc[k, "label"]
                sendback_data["labeler"] = self.test_tree.loc[k, "labeler"]
                self.comm.send({k: sendback_data})
                
                self._auto_save()

            # if we are just changing the tags
            elif "tags" in msg[k] and len(msg[k]) == 1:
                
                # move a test that is in the test tree
                if k in self.test_tree.index:
                    if msg[k]["tags"] == "_DELETE_": # this means delete the test
                        self.test_tree.drop(k, inplace=True)
                    else:
                        self.test_tree.loc[k, "tags"] = msg[k]["tags"]
                        self.test_tree.loc[k, "author"] = self.user
                
                # move a whole tag set around
                else:
                    source_tag = k
                    target_tag = msg[k]["tags"]
                    assert not source_tag.contains(":") and not target_tag.contains(":"), "Can only move one tag at a time, not sets of tags!"
                    if target_tag == "_DELETE_":
                        self.test_tree.drop(self.test_tree.index[self.test_tree.has_tag(source_tag)], inplace=True)
                    else:
                        self.test_tree["tags"].str.replace(r"(^|:)%s(/|$|:)" % re.escape(source_tag), r"\1%s\2" % re.escape(target_tag), inplace=True)

                # Recompute any missing embeddings to handle any changes
                self._compute_scores(self.test_tree)
                self._refresh_interface()
                self._auto_save()

            else:
                log.debug(f"Unable to parse the interface message: {msg[k]}")

    def _refresh_interface(self):
        """ Send our entire current state to the frontend interface.
        """

        # get the children of the current topic
        data = {}

        def create_children(data, tests, tags):
                    
            # find tests that are each in all of the given topics
            children = tests.index[tests.has_exact_tags(tests, tags)]
            for k in children:
                test = tests.loc[k]
                
                # add a test
                if test.label != "tag_marker" and matches_filter(test, self.filter_text):
                    data[k] = {
                        "input": test.input,
                        "output": test.output,
                        "label": test.label,
                        "labeler": test.labeler,
                        "description": test.description,
                        "scores": {c: [[k, v] for v in ui_score_parts(test[c], test.label)] for c in self.score_columns},
                        "editing": test.input == "New test"
                    }

                    data[k]["raw_outputs"] = {c: [[k, safe_json_load(test.get(c[:-6] + " raw outputs", "{}"))]] for c in self.score_columns}

            # sort by score and always put new topics first
            def sort_key(id):
                try:
                    total = 0
                    count = 0
                    # offset = 0 if data[id]["label"] == "fail" else -1
                    for s in data[id]["scores"][self.score_columns[0]]:
                        val = score_max(s[1], nan_val=np.nan)
                        if not np.isnan(val) and val is not None:
                            total += val #+ offset
                            count += 1
                    if count == 0:
                        return 1e3
                    else:
                        return -total / count
                except Exception as e:
                    print(e)
            sorted_children = sorted(children, key=sort_key)
            sorted_children = sorted(sorted_children, key=lambda id: 0 if data[id].get("label", "") == "tag_marker" else 1) # put folders first
            sorted_children = sorted(sorted_children, key=lambda id: 1 if data[id].get("label", "") == "off_topic" else 0) # off topic last
            sorted_children = sorted(sorted_children, key=lambda id: 0 if id.endswith("/New tag") or data[id].get("input", "") == "New test" else 1) # put new items first

            return sorted_children
        
        # get the children of the current topics
        children = create_children(data, self.test_tree, self.current_tags)
        suggestions_children = create_children(data, self.test_tree, self.current_tags + ["/__suggestions__"])

        # create tag entries
        tag_ids = self.test_tree.index[self.test_tree["label"] == "tag_marker"]
        for k in tag_ids:
            test = self.test_tree.loc[k]

            # add a tag
            data[test.tags] = {
                "label": test.label,
                "labeler": test.labeler,
                "description": "",
                "scores": {c: [] for c in self.score_columns},
                "tag_marker_id": k,
                "tag_name": test.tags.rsplit("/", 1)[-1] if test.tags != "" else "Root",
                "editing": test.tags.endswith("/New tag"),
                "child_selected": any([re.match(r"(^|:)%s(/|$|:)" % re.escape(test.tags), t) for t in self.current_tags]) and test.tags not in self.current_tags,
                "selected": test.tags in self.current_tags,
                "children": []
            }

        # fill in the children of the tags
        for k in tag_ids:
            test = self.test_tree.loc[k]

            # add ourselves to our parent tag's children list
            parts = test.tags.rsplit("/", 1)
            if len(parts) > 1:
                parent = parts[0]
                data[parent]["children"].append(test.tags)

                # create an 'Uncatagorized' tag if we don't have one
                if len(data[parent]["children"]) == 1:
                    subtag = parent + "/Uncategorized"
                    data[subtag] = {
                        "label": "tag_marker",
                        "labeler": "generated",
                        "description": "",
                        "scores": {c: [] for c in self.score_columns},
                        "tag_marker_id": subtag,
                        "tag_name": "Uncategorized",
                        "editing": False,
                        "child_selected": False, # uncategorized tags cannot have children
                        "selected": subtag in self.current_tags,
                    }
                    data[parent]["children"].append(subtag)

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

        # compile the global browser state for the frontend
        data["browser"] = {
            "suggestions": suggestions_children,
            "tests": children,
            "user": self.user,
            "tags": self.current_tags,
            "score_filter": score_filter,
            "disable_suggestions": False,
            "read_only": False,
            "score_columns": self.score_columns,
            "suggestions_error": self._suggestions_error,
            "generator_options": [str(x) for x in self.generators.keys()] if isinstance(self.generators, dict) else [self.active_generator],
            "active_generator": self.active_generator,
            "mode": self.mode,
            "mode_options": self.mode_options
        }

        self.comm.send(data)

    def _clear_suggestions(self):
        """ Clear the suggestions for the current topics.
        """
        ids = self.test_tree.index[self.test_tree.has_exact_tags(self.current_tags + ["/__suggestion__"])]
        self.test_tree.drop(ids, inplace=True)

    def generate_suggestions(self, tags=None, filter=""):
        if tags is not None:
            self.current_tags = tags
        self._clear_suggestions()
        self.test_tree.retrain_topic_labeling_model(self.current_tags)
        self.test_tree.retrain_topic_membership_model(self.current_tags)
        self._generate_suggestions(filter=filter)

    def _generate_suggestions(self, filter, mode="examples"):
        """ Generate suggestions for the current topic.

        Parameters
        ----------
        filter : str
            The filter to apply to the tests while generating suggestions.
        """

        # save a lookup we can use to detect duplicate tests
        test_map = {}
        for _, test in self.test_tree.iterrows():
            if test.label == "tag_marker":
                test_map[test.tags + " __tag_marker__"] = True
            else:
                test_map[test.tags + " __JOIN__ " + test.input] = True

        # compute the maximum number of suggestion threads we can use given our suggestion_thread_budget
        p = self.prompt_builder.prompt_size
        budget = 1 + self.suggestion_thread_budget
        suggestion_threads = max(1, int(np.floor(budget * (p/(p+1) + 1/(p+1) * self.max_suggestions) - 1/(p+1) * self.max_suggestions) / (p/(p+1))))
        
        # generate the prompts for the backend
        prompts = self.prompt_builder(
            test_tree=self.test_tree,
            tags=self.current_tags,
            score_column=self.score_columns[0],
            repetitions=suggestion_threads,
            filter=filter,
            suggest_tags=mode == "tags"
        )

        # get the current topic description
        curr_topic_mask = (self.test_tree["topic"] == self.current_tags) & (self.test_tree["label"] == "tag_marker")
        if curr_topic_mask.sum() == 0:
            desc = ""
        else:
            desc = self.test_tree.loc[(self.test_tree["topic"] == self.current_tags) & (self.test_tree["label"] == "tag_marker")]["description"][0]

        # generate the suggestions
        generators = [self._active_generator_obj] + list(self.generators.values())
        for generator in generators:
            try:
                proposals = generator(prompts, self.current_tags, desc, self.mode, self.scorer, num_samples=self.max_suggestions // len(prompts) if len(prompts) > 0 else self.max_suggestions)
                break
            except ValueError:
                pass # try the next generator
        
        # all topics should be URI encoded
        if self.mode == "topics":
            proposals = [urllib.parse.quote(x) for x in proposals]
        
        # Build up suggestions catalog, unless generating from a test tree source.
        # NOTE: Doing safe checks for TestTree type in order to prevent circular imports
        if isinstance(proposals, pd.DataFrame) or proposals.__class__.__name__ == "TestTree":
            suggestions = proposals
            suggestions['topic'] = self.current_tags + "/__suggestions__" + suggestions['topic'].apply(lambda x: x[len(self.current_tags):] if x != "" else "")
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
                    str_val = self.current_tags + "/" + input + " __tag_marker__"
                else:
                    str_val = self.current_tags + " __JOIN__ " + input
                if str_val not in test_map_tmp:
                    id = uuid.uuid4().hex
                    self.test_tree.loc[id, "topic"] = self.current_tags + "/__suggestions__" + ("/"+input if self.mode == "topics" else "")
                    self.test_tree.loc[id, "input"] = "" if self.mode == "topics" else input
                    self.test_tree.loc[id, "output"] = "__TOOVERWRITE__"
                    self.test_tree.loc[id, "label"] = "tag_marker" if self.mode == "topics" else ""
                    self.test_tree.loc[id, "labeler"] = "imputed"
                    self.test_tree.loc[id, "description"] = ""
                    for c in self.score_columns:
                        self.test_tree.loc[id, c] = "__TOEVAL__"

                    # s = {
                    #     "topic": self.current_tags + "/__suggestions__" + ("/"+input if self.mode == "topics" else ""),
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
            # make sure any duplicates we may have introduced are removed
            self.test_tree.drop_duplicates(in_place=True)
            
            # compute the scores for the new tests
            self._compute_scores(self.test_tree)

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

    def _get_tag_marker_id(self, topic):
        """
        Returns the id of the topic marker row for the given topic.
        Returns None if not found.
        """
        tag_marker_index_df = self.test_tree.index[(self.test_tree['topic'] == topic) & (self.test_tree['label'] == 'tag_marker')]
        tag_marker_index = tag_marker_index_df.tolist()[0] if len(tag_marker_index_df) > 0 else None
        return tag_marker_index

    def _add_topic_model_score(self, df, topic_model_scale):
        """ This is an old experimental funciton that is not meant to be used anymore.
        """
        import openai
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
            self.current_tags,
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

    def _compute_scores(self, tests, recompute=False, overwrite_outputs=False, save_outputs=False): # TODO: Rename/refactor/merge with _compute_scores?
        log.debug(f"compute_embeddings_and_scores(tests=<DataFrame shape={tests.shape}>, recompute={recompute})")

        for k in self.scorer:

            # determine which rows we need to evaluate
            eval_ids = tests.index[((tests[k+" score"] == "__TOEVAL__") | (tests["output"] == "__TOOVERWRITE__")) & (tests["label"] != "tag_marker") & (tests["label"] != "off_topic")]

            if len(eval_ids) > 0:

                # run the scorer
                new_outputs,scores = self.scorer[k](tests, eval_ids)

                # update the scores in the test tree
                current_outputs = tests["output"]
                for i,id in enumerate(eval_ids):

                    if not overwrite_outputs and current_outputs.loc[id] != "__TOOVERWRITE__" and current_outputs.loc[id] != new_outputs[i]:

                        # mark the current row as nan score (meaning the output does not match)
                        tests.loc[id, k+" score"] = np.nan

                        # add a new test where the model output does match if we are saving outputs
                        if save_outputs:
                            id_new = uuid.uuid4().hex
                            tests.loc[id_new, "tags"] = tests.loc[id, "tags"]
                            tests.loc[id_new, "input"] = tests.loc[id, "input"]
                            tests.loc[id_new, "output"] = new_outputs[i]
                            tests.loc[id_new, "labeler"] = "imputed"
                            tests.loc[id_new, "label"] = ""
                            tests.loc[id_new, k+" score"] = scores[i]
                    else:
                        tests.loc[id, "output"] = new_outputs[i]
                        tests.loc[id, k+" score"] = scores[i]

        # make sure any duplicates we may have introduced are removed
        tests.drop_duplicates(in_place=True)

        # reimpute missing labels
        tests.impute_labels()

    def templatize(self, s):
        """ This is an experimental function that is not meant to be used generally.
        """
        import openai
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

def ui_score_parts(s, label):
    """ Split a score into its parts and encode the label into the sign.
    
    Note this encoding is just used for passing scores to the UI (scores are not signed in the TestTree).
    """
    offset = 0
    if label == "pass":
        sign = 1
        offset = -1 - 1e-6
    elif label == "fail":
        sign = 1
        offset = 1e-6 # just so we have a positive number to encode that this was a failure
    else:
        sign = np.nan
    if isinstance(s, str):
        return [np.clip(offset + convert_float(v)*sign, -1, 1) for v in s.split("|")]
    else:
        return [np.clip(offset + s*sign, -1, 1)]

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