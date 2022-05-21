import numpy as np
import logging
import re
import sentence_transformers
import torch
import adatest

log = logging.getLogger(__name__)

# test_tree(scorer=model, dataset=dataset)

# test_tree()

# test_tree2 = test_tree(dataset)

# test_tree.auto_categorize(dataset)

# test_tree.browse(dataset)

# test_tree.browse(target=model, dataset=dataset)

# test_tree.browse(model)

# adatest.browse(test_tree, model)



class PromptBuilder():
    """ A class to build prompts for the model.
    """
    
    def __init__(self, prompt_size=7, slot_randomization=0.25, score_randomization=1.0, skip_randomization=0.25, prompt_diversity=True,
                 subtopic_diversity=True):
        """ Initialize the prompt builder.
        
        Parameters
        ----------
        prompt_size : int
            The number of test slots to include in the prompt.

        slot_randomization : float
            The proportion of slots to make fully random (within the current topic).

        score_randomization : float
            An additive randomization factor for the scores. 0 mean no randomization, 1 means add one standard
            deviation of Gaussian noise.

        skip_randomization : float
            The proportion of times we skip over top ranking tests when building the prompt.

        prompt_diversity : bool
            Whether to include a diversity term when selecting tests for the prompt. This diversity term is based
            on the embeddings of each test.

        subtopic_diversity : bool
            If true, we will try and pick tests from a diverse set of subtopics of the current topic (if we are
            using subtopic tests and not direct child tests).
        """

        assert skip_randomization < 0.99, "skip_randomization must be less than 1, otherwise everything will always be skipped!"

        self.prompt_size = prompt_size
        self.slot_randomization = slot_randomization
        self.score_randomization = score_randomization
        self.skip_randomization = skip_randomization
        self.prompt_diversity = prompt_diversity
        self.subtopic_diversity = subtopic_diversity
    
    def __call__(self, test_tree, topic, score_column, repetitions=1, filter="", suggest_topics=False, working_set_size=100, embeddings=None):
        """ This builds a prompt for GPT3 that elicits useful input examples.

        Parameters
        ----------
        test_tree : adatest.TestTree
            The test tree to generate prompts from.
        
        topic : str
            The topic to build a prompt for.

        score_column : str
            The column to use for scoring the tests.

        repetitions : int
            The number of times to repeat the prompt generation process. This is how many prompots we will return.

        filter : str
            A filter to apply to the test set before selecting tests to build the prompt.

        suggest_topics : bool
            If true, we will create a prompt filled with topic names instead of a list of tests.

        working_set_size : int
            How many top tests to consider when doing the full iterative scoring process. Larger values may take longer.
            Note that this has no effect as long as we never go more than working_set_size tests deep during the prompt
            item selection process.

        embeddings : dict
            A dictionary of embeddings to use for the prompt. This is used to compute the prompt_diversity.
        """

        ids = np.array(test_tree.index)

        # make sure all the embeddings are computed
        test_tree.compute_embeddings()
        
        # if suggesting topics mark only direct subtopics as "in topic"
        # if suggest_topics:
        #     topic_scaling = np.zeros(len(ids))
        #     for i, k in enumerate(ids):
        #         if test_tree.loc[k, "type"] == "topic_marker" and test_tree.loc[k, "topic"].rsplit('/', 1)[0] == topic:
        #             topic_scaling[i] = 1.0
        
        # # if suggesting tests we compute each test's distance from current topic, where distance
        # # is measured by the length of the topic prefix shared between the test and the current topic
        # else:
        parts = topic.split("/")
        topic_scaling = np.ones(test_tree.shape[0])
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i+1])
            if suggest_topics:
                prefix += "/"
            topic_scaling *= 1 + 99 * np.array([v.startswith(prefix) for v in test_tree["topic"]])
        
        # promote direct children over subtopic descendants and filter for topics vs tests
        if suggest_topics:
            topic_scaling *= 1 + 99 * np.array([v.rsplit('/', 1)[0] == topic for v in test_tree["topic"]])
            topic_scaling *= np.array(test_tree["label"] == "topic_marker")
        else:
            topic_scaling *= 1 + 99 * np.array([v == topic for v in test_tree["topic"]])
            topic_scaling *= np.array(test_tree["label"] != "topic_marker")

        topic_scaling /= np.max(topic_scaling)

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
        if suggest_topics:
            # test_type = "topic_marker"
            scores = np.ones(len(ids)) # Scores should not influence topic suggestions
        else:
            # compute a positive single value score for each test
            scores = np.array([score_max(test_tree.loc[k, score_column]) for k in ids])
            scores -= np.nanmin(scores) - 1e-8 # the 1e-8 terms causes NaN scores to be last priority
            scores = np.nan_to_num(scores)

            # rank_vals = scores * topic_scaling * hidden_scaling
            # test_type = test_tree.loc[ids[np.argmax(rank_vals)], "type"]
            # for i,k in enumerate(ids):
            #     if test_tree.loc[k, "type"] != test_type:
            #         hidden_scaling[i] = 0.0
                    

        # filter down to just top rows we will use during the iterative scoring process
        rank_vals = scores * topic_scaling * hidden_scaling
        top_inds = np.argsort(-rank_vals)[:working_set_size]
        ids = ids[top_inds]
        topic_scaling = topic_scaling[top_inds]
        hidden_scaling = hidden_scaling[top_inds]
        scores = scores[top_inds]

        # build a list of randomized prompts
        prompts = []
        for _ in range(repetitions):

            # store tmp versions of things we update during the iteration
            scores_curr = scores.copy()
            topic_scaling_curr = topic_scaling.copy()

            # score randomization TODO: Scott look into scores_curr > 1
            score_weights = topic_scaling * (scores_curr > 1)
            if np.sum(score_weights) > 0:
                std_dev = np.sqrt(np.cov(scores_curr, aweights=score_weights)) + 1e-6
            else:
                std_dev = 1e-6
            if not np.isnan(std_dev):
                scores_curr += self.score_randomization * std_dev * np.random.rand(len(ids))

            # sim_avoidance is a vector that marks which items (and items related through similarities)
            # should be avoided (ranked lower for prompt selection)
            if self.prompt_diversity:
                sim_avoidance = np.zeros(len(ids))
                if suggest_topics:
                    embeddings_arr = torch.tensor(np.vstack(adatest.embed(
                        [test_tree.loc[id, "topic"].split("/")[-1] for id in ids]
                    )))
                else:
                    embeddings_arr = torch.tensor(np.vstack([
                        np.hstack(adatest.embed([test_tree.loc[id, "input"], test_tree.loc[id, "output"]]))
                    for id in ids]))
                similarities = adatest._cos_sim(embeddings_arr, embeddings_arr).numpy()
            hard_avoidance = np.zeros(len(ids))
            diversity = np.ones(len(ids))

            # compute how many greedy and how many random positions we will have
            num_random = max(0, min(np.random.binomial(self.prompt_size, self.slot_randomization), len(ids) - self.prompt_size))
            num_greedy = max(0, min(self.prompt_size - num_random, len(ids) - num_random))
            
            # iteratively select prompt items
            prompt_ids = []
            outside_topics_used = np.ones(len(ids))
            while len(prompt_ids) < num_greedy + num_random:

                # once we get to the random part of the process we forget what topics we have visited and scramble the scores
                if len(prompt_ids) == num_greedy:
                    scores = 1 + np.random.rand(len(ids))*0.1

                # find the next bext index
                if self.prompt_diversity:
                    diversity = 1 - (similarities * sim_avoidance).max(1)
                rank_vals = scores * topic_scaling_curr * diversity * (1 - hard_avoidance) * hidden_scaling * outside_topics_used

                if np.nanmax(rank_vals) <= 0 and len(prompt_ids) > 0: # stop if we have run out of the current subtree
                    break

                new_ind = np.nanargmax(rank_vals)
                skip_rand = np.random.rand()

                # make it unlikely we will choose the same outside topic twice
                new_ind_topic = test_tree.loc[ids[new_ind], "topic"]
                if not is_subtopic(topic, new_ind_topic):
                    outside_topics_used *= 1 - 0.9 * np.array([test_tree.loc[id, "topic"] == new_ind_topic for id in ids])

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

                # lower the weight of the subtopic we just picked from
                if self.subtopic_diversity:
                    new_topic = test_tree.loc[ids[new_ind], "topic"]
                    if topic == new_topic:
                        subtopic_scaling = np.ones(len(ids))
                        subtopic_scaling[new_ind] = 0.0001
                    else:
                        subtopic = topic + "/" + new_topic[(len(topic)+1):].split("/")[0]
                        # print(subtopic)
                        # subtopic_scaling = np.array([0.0001 if test_tree.loc[k, "topic"].startswith(subtopic) else 1 for k in ids])
                        subtopic_scaling = np.array([0.0001 if is_subtopic(subtopic, test_tree.loc[k, "topic"]) else 1 for k in ids])
                    topic_scaling_curr *= subtopic_scaling

            # create the prompt as a list of tuples
            prompt = []
            for k in reversed(prompt_ids):
                row = test_tree.loc[k]
                if suggest_topics:
                    if row["topic"] == "":
                        continue # we can't use the root to help suggest topic names
                    parents,child = row["topic"].rsplit("/", 1)
                    prompt.append((k, parents, child))
                else:
                    prompt.append((k, row["topic"], row["input"]))
            prompts.append(prompt)
        
        return prompts

def is_subtopic(topic, candidate):
    """ Returns True if candidate is a subtopic of topic.
    """
    return True if re.search(r'^%s(/|$)' % topic.replace('+', r'\+'), candidate) else False

def score_max(s):
    if s == "" or s is None:
        return -1e3
    elif isinstance(s, str):
        return np.max([convert_float(v) for v in s.split("|")])
    elif np.isnan(s):
        return -1e3
    else:
        return np.max(s)

def convert_float(s):
    try:
        f = float(s)
    except ValueError:
        f = np.nan
    return f