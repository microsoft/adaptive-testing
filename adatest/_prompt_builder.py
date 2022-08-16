import numpy as np
import logging
import re
import urllib.parse
import adatest
from .embedders import cos_sim
from .utils import has_tags, has_tag
log = logging.getLogger(__name__)


class PromptBuilder():
    """ A class to build prompts for the model.
    """
    
    def __init__(self, prompt_size=7, slot_randomization=0.25, score_randomization=0.05, skip_randomization=0.25, prompt_diversity=True,
                 subtag_diversity=True):
        """ Initialize the prompt builder.
        
        Parameters
        ----------
        prompt_size : int
            The number of test slots to include in the prompt.

        slot_randomization : float
            The proportion of slots to make fully random (within the current topic).

        score_randomization : float
            The standard deviation of an additive Gaussian randomization factor for the scores.

        skip_randomization : float
            The proportion of times we skip over top ranking tests when building the prompt.

        prompt_diversity : bool
            Whether to include a diversity term when selecting tests for the prompt. This diversity term is based
            on the embeddings of each test.

        subtag_diversity : bool
            If true, we will try and pick tests from a diverse set of subtags of the current tag (if we are
            using subtag tests and not direct child tests).
        """

        assert skip_randomization < 0.99, "skip_randomization must be less than 1, otherwise everything will always be skipped!"

        self.prompt_size = prompt_size
        self.slot_randomization = slot_randomization
        self.score_randomization = score_randomization # TODO: make this scale according to the stddev of the top 7 entries?
        self.skip_randomization = skip_randomization
        self.prompt_diversity = prompt_diversity
        self.subtag_diversity = subtag_diversity
    
    def __call__(self, test_tree, tags, score_column, repetitions=1, filter="", suggest_tagsets=False, working_set_size=100, embeddings=None):
        """ This builds a prompt for GPT3 that elicits useful input examples.

        Parameters
        ----------
        test_tree : adatest.TestTree
            The test tree to generate prompts from.
        
        tags : str
            The tag set to build a prompt for. Note that you shouldn't include /__suggestion__ as a tag normally.

        score_column : str
            The column to use for scoring the tests.

        repetitions : int
            The number of times to repeat the prompt generation process. This is how many prompots we will return.

        filter : str
            A filter to apply to the test set before selecting tests to build the prompt.

        suggest_tagsets : bool
            If true, we will create a prompt filled with tag names instead of a list of tests.

        working_set_size : int
            How many top tests to consider when doing the full iterative scoring process. Larger values may take longer.
            Note that this has no effect as long as we never go more than working_set_size tests deep during the prompt
            item selection process.

        embeddings : dict
            A dictionary of embeddings to use for the prompt. This is used to compute the prompt_diversity.
        """

        ids = np.array(test_tree.index)

        # return early for an empty test tree
        if len(ids) == 0:
            return [[] for _ in range(repetitions)]
        
        # we compute each test's distance from current topic, where distance is measured
        # by the length of the topic prefix shared between the test and the current topic
        tag_scaling = np.ones(test_tree.shape[0])
        tag_list = tags.split(":")
        for tag in tag_list:
            topic_parts = tag.split("/")
            for i in range(1, len(topic_parts)):
                prefix = "/".join(topic_parts[:i+1])
                if suggest_tagsets:
                    prefix += "/"
                tag_scaling *= 1 + 99 * test_tree.has_tag(prefix)
        
        # promote direct children over subtopic descendants and filter for topics vs tests
        if suggest_tagsets:
            # match direct children (note that subtag suggestions can't be for multiple tags)
            tag_scaling *= 1 + 99 * test_tree["tags"].str.match(r"^%s/[^/]+$" % re.escape(tags))
            tag_scaling *= np.array(test_tree["label"] == "tag_marker")
        else:
            for tag in tag_list:
                tag_scaling *= 1 + 99 * test_tree.has_exact_tag(tag)
            tag_scaling *= np.array(test_tree["label"] != "tag_marker")
        tag_scaling *= np.array(["__suggestions__" not in t for t in test_tree["topic"]])

        # return early if we have nothing to build a prompt with
        if np.sum(tag_scaling) == 0:
            return [[] for _ in range(repetitions)]

        tag_scaling /= np.max(tag_scaling)

        # hide rows that don't match the filter
        hidden_scaling = np.ones(len(ids))
        if filter != "":
            filter_compiled = re.compile(filter)
            for i,k in enumerate(ids):
                test = test_tree.loc[k]
                if filter_compiled.search(test.test_type) is not None:
                    continue
                if hasattr(test, "input") and filter_compiled.search(test.input) is not None:
                    continue
                if hasattr(test, "output") and filter_compiled.search(test.output) is not None:
                    continue
                hidden_scaling[i] = 0.0

        # filter down to a single test type (chosen to match the top scoring test)
        if suggest_tagsets:
            # scores currently do not influence topic suggestions
            # TODO: can we score topics and topic suggestions?
            scores = np.ones(len(ids))
        else:
            # compute a positive single value score for each test
            scores = np.array([score_max(test_tree.loc[k, score_column], test_tree.loc[k, "label"]) for k in ids])

        # filter down to just top rows we will use during the iterative scoring process
        rank_vals = scores * tag_scaling * hidden_scaling
        top_inds = np.argsort(-rank_vals)[:working_set_size]
        ids = ids[top_inds]
        tag_scaling = tag_scaling[top_inds]
        hidden_scaling = hidden_scaling[top_inds]
        scores = scores[top_inds] * 1.0

        # build a list of randomized prompts
        prompts = []
        for _ in range(repetitions):

            # store tmp versions of things we update during the iteration
            scores_curr = scores.copy()
            tag_scaling_curr = tag_scaling.copy()

            # score randomization
            scores_curr += self.score_randomization * np.random.rand(len(ids))

            # sim_avoidance is a vector that marks which items (and items related through similarities)
            # should be avoided (ranked lower for prompt selection)
            if self.prompt_diversity:
                sim_avoidance = np.zeros(len(ids))
                if suggest_tagsets:
                    embeddings_arr = np.vstack(adatest.embed(
                        [urllib.parse.unquote(test_tree.loc[id, "tags"].split("/")[-1]) for id in ids],
                    ))
                else:
                    embeddings_arr = np.hstack([
                        np.vstack(adatest.embed([test_tree.loc[id, "input"] for id in ids])),
                        np.vstack(adatest.embed([test_tree.loc[id, "output"] for id in ids]))
                    ])
                similarities = cos_sim(embeddings_arr, embeddings_arr)
            hard_avoidance = np.zeros(len(ids))
            diversity = np.ones(len(ids))

            # compute how many greedy and how many random positions we will have
            num_random = max(0, min(np.random.binomial(self.prompt_size, self.slot_randomization), len(ids) - self.prompt_size))
            num_greedy = max(0, min(self.prompt_size - num_random, len(ids) - num_random))
            
            # iteratively select prompt items
            prompt_ids = []
            outside_tags_used = np.ones(len(ids))
            while len(prompt_ids) < num_greedy + num_random:

                # once we get to the random part of the process we scramble the scores
                if len(prompt_ids) == num_greedy:
                    scores_curr = 1 + np.random.rand(len(ids))*0.1

                # find the next bext index
                if self.prompt_diversity:
                    diversity = 1 - (similarities * sim_avoidance).max(1)
                rank_vals = scores_curr * tag_scaling_curr * diversity * (1 - hard_avoidance) * hidden_scaling * outside_tags_used

                if np.nanmax(rank_vals) <= 0 and len(prompt_ids) > 0: # stop if we have run out of the current subtree
                    break

                new_ind = np.nanargmax(rank_vals)
                skip_rand = np.random.rand()

                # make it unlikely we will choose the same outside tag set twice
                new_ind_tags = test_tree.loc[ids[new_ind], "tags"]
                if not has_tags(tags, new_ind_tags):
                    outside_tags_used *= 1 - 0.9 * test_tree.has_tags(new_ind_tags)

                # add or skip this item
                if skip_rand >= self.skip_randomization:
                    prompt_ids.append(ids[new_ind])
                    avoidance_level = 1
                else:
                    avoidance_level = 1 - 0.1

                # avoid this IO pair as we select the next pairs
                hard_avoidance[new_ind] = avoidance_level
                if self.prompt_diversity:
                    sim_avoidance[new_ind] = avoidance_level

                # lower the weight of the subtag(s) we just picked from
                if self.subtag_diversity:
                    new_tags = test_tree.loc[ids[new_ind], "tags"]
                    if tags != new_tags and has_tags(tags, new_tags):
                        tag_scaling_curr *= 1 - 0.999 * test_tree.has_tags(new_tags)
                        # TODO: this should really penalize the direct child tag not just the full tag path like what we used to have:
                        # subtopic = topic + "/" + new_tags[(len(topic)+1):].split("/")[0]
                        # subtag_scaling = np.array([0.001 if is_subtopic(subtopic, test_tree.loc[k, "topic"]) else 1 for k in ids])

            # create the prompt as a list of tuples
            prompt = []
            for k in reversed(prompt_ids):
                row = test_tree.loc[k]
                if suggest_tagsets:
                    for tag in row["tags"].split(":"):
                        if has_tag(tag, tags):
                            parents,child = tag.rsplit("/", 1)
                            prompt.append((k, parents, urllib.parse.unquote(child)))
                            break
                    #     prompt.append((tag, row["label"]))
                    # if row["tags"] == "":
                    #     continue # we can't use the root to help suggest topic names
                    # parents,child = row["topic"].rsplit("/", 1)
                    # prompt.append((k, parents, urllib.parse.unquote(child)))
                else:
                    prompt.append((k, row["tags"], row["input"]))
            prompts.append(prompt)
        
        return prompts

def score_max(s, label):
    if s == "" or s is None:
        return 1 if label == "fail" else 0
    elif isinstance(s, str):
        return np.max([convert_float(v) for v in s.split("|")])
    elif np.isnan(s):
        return 1 if label == "fail" else 0
    else:
        return np.max(s)

def convert_float(s):
    try:
        f = float(s)
    except ValueError:
        f = np.nan
    return f