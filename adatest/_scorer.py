import numpy as np
import re
import logging
import uuid
import itertools
import sentence_transformers
import openai
import scipy.stats
import transformers
import shap

import adatest
from ._model import Model
from .utils import isinstance_ipython
#from transformers.tokenization_utils_base import ExplicitEnum
#from ._explorer import file_log

log = logging.getLogger(__name__)

class Scorer():
    def __new__(cls, model, *args, **kwargs):
        """ If we are wrapping an object that is already a Scorer, we just return it.
        """
        if isinstance_ipython(model, Scorer):
            return model
        else:
            return super().__new__(cls)
    
    def __init__(self, model):
        """ Auto detect the model type and subclass to the right scorer object.
        """

        # do common init stuff if this is the first time we are called
        if not hasattr(self, "_id"):
            self._id = uuid.uuid4().hex

            # ensure we have a model of type Model
            if isinstance_ipython(model, Model):
                self.model = model
            else:
                self.model = Model(model)

        # If we are in the base class we need to pick the right specialized subclass to become
        if self.__class__ is Scorer:

            # finish early if we are wrapping an object that is already a Scorer (__new__ will have already done the work)
            if isinstance_ipython(model, Scorer):
                return
            
            # see if we are scoring a generator or a classifier
            out = self.model(["string 1", "string 2"])
            if isinstance(out[0], str):
                self.__class__ = GeneratorScorer
                GeneratorScorer.__init__(self, model)
            else:
                self.__class__ = ClassifierScorer
                ClassifierScorer.__init__(self, model)
            

class DummyScorer(Scorer):
    def __init__(self):
        self._id = uuid.uuid4().hex
    def __call__(self, tests):
        out = []
        for k, test in tests.iterrows():
            try:
                score = float(test.value2)
            except:
                score = np.nan
            out.append(score)
        return np.array(out)

class ClassifierScorer(Scorer):
    """ Wraps a model and defines a callable scorer that returns a score value for any input/output pair.

    Positive scores indicate test failures, positive scores indicate tests that pass. For example if we wrap
    a text sentiment classifer the `scorer(TestTree([("this is great!", "should be", "POSITIVE")]))` will return
    a large positive value indicating that the model is very likely to correctly produce that output when given
    that input.
    """

    def __init__(self, model, topk=1, output_names=None, method="dirichlet", dirichlet_concentration=10):
        """ Create a new scorer given a model that returns a probability vector for each input string.
        
        Parameters:
        -----------
        model : callable
            A model that is callable with a single argument (which is a list of strings) and returns a matrix of outputs.

        topk : int
            The number of top outputs to consider when scoring tests. For example topk=2 causes "should not be" tests to
            check the top two model outputs.

        output_names : list of strings
            A list of strings that correspond to the outputs of the model. If None, model.output_names is used.

        method : 'margin' or 'dirichlet'
            The scoring method to use. Dirichlet is preferred, but margin is available for backwards compatibility.

        dirichlet_concentration : float
            The concentration parameter for the dirichlet scoring method. It is in the units o pseudo-counts where larger
            values lead to a tighter prior centered around the model's probability outputs (so scores are more likely to
            be -1 or +1).
        """
        super().__init__(model)

        # we don't want to re-init a class if init has alrady been done (this can happen when Scorer(maybe_scorer) is called)
        if hasattr(self, "output_names"):
            return # already initialized

        # extract output names from the model if they are not provided directly
        if output_names is None:
            self.output_names = self.model.output_names
        else:
            self.output_names = output_names
        
        if not callable(self.output_names):
            self._output_name_to_index = {v: i for i, v in enumerate(self.output_names)}
            assert topk == 1, "topk must be 1 right now" # TODO: this is because we need to figure out topk and templates
        self.topk = topk
        self.method = method
        self.dirichlet_concentration = dirichlet_concentration

    def __call__(self, tests, score_column, overwrite_outputs=False, output_bias=0.5):
        """ Score a set of tests.

        Parameters
        ----------
        tests : pandas.DataFrame
            A dataframe of tests.

        output_bias : float
            How much to bias the output entries in a test towards the actual output from the model.
        """


        
        # determine which rows we need to evaluate
        eval_inputs = []
        eval_inds = []
        for i, (id, test) in enumerate(tests.iterrows()):
            if test[score_column] == "" and test.label != "topic_marker":
                template_expansions = expand_template(test.input)
                for expansion in template_expansions:
                    eval_inputs.append(expansion)
                    eval_inds.append(i)

        # run the model on the rows we need to evaluate
        try:
            model_out = self.model(eval_inputs)
        except Exception as e:
            model_out = np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
            log.error(e)
            log.error(eval_inputs)
            log.error("The model threw an exception when evaluating inputs! We are patching this disaster with np.nan for the sake of the user study!")

        # compute the output strings for each output
        out_strings = [[] for _ in range(tests.shape[0])]
        out_probs = [[] for _ in range(tests.shape[0])]
        i = 0
        while i < len(model_out):
            out_strings[eval_inds[i]].append(self.model.output_names[np.argmax(model_out[i])])
            out_probs[eval_inds[i]].append(np.max(model_out[i]))
            i += 1
        for i in eval_inds:
            out_strings[i] = "|".join(out_strings[i]) # template outputs are joined by |
            out_probs[i] = np.min(out_probs[i]) # the probability of a set of items is the prob of the min item

        # ensure the model output is represented in the tests
        current_outputs = tests["output"]
        current_labelers = tests["labeler"]
        updated_ids = []
        for i, ind in enumerate(eval_inds):
            if current_labelers.iloc[ind] == "imputed":
                id = tests.index[ind]
                tests.loc[id, "output"] = out_strings[ind]
                tests.loc[id, "label"] = ""
                tests.loc[id, score_column] = out_probs[ind]
                updated_ids.append(id)
            elif not overwrite_outputs and current_outputs.iloc[ind] != out_strings[ind]:

                # mark the current row as nan score (meaning the output does not match)
                tests.loc[tests.index[ind], score_column] = np.nan

                # add a new test where the model output does match
                id = uuid.uuid4().hex
                tests.loc[id, "topic"] = tests.loc[tests.index[ind], "topic"]
                tests.loc[id, "input"] = eval_inputs[i]
                tests.loc[id, "output"] = out_strings[ind]
                tests.loc[id, "label"] = ""
                tests.loc[id, "labeler"] = "imputed"
                tests.loc[id, score_column] = out_probs[ind]

                updated_ids.append(id)
            else:
                id = tests.index[ind]
                tests.loc[id, "output"] = out_strings[ind]
                tests.loc[id, score_column] = out_probs[ind]
                updated_ids.append(id)
        tests.deduplicate() # make sure any duplicates we may have introduced are removed

        # reimpute missing labels
        tests.impute_labels() # TODO: ensure this method caches the local models and only reimputes when needed for each topic

        # set the test score sign to match the label
        for id in updated_ids:
            if id in tests.index and tests.loc[id, "label"] == "pass":
                tests.loc[id, score_column] = -float(tests.loc[id, score_column])
        


        # out_pos = 0
        # i = 0
        # top_outputs = [{} for _ in range(tests.shape[0])]
        # value2_outputs = [{} for _ in range(tests.shape[0])]
        # while i < len(model_out):
        #     out_pos = eval_inds[i]

        #     # save the top model outputs
        #     inds = np.argsort(-model_out[i])
        #     shown_tmp = {}
        #     for j in inds[:5]:
        #         shown_tmp[self.model.output_names[j]] = float(model_out[i][j])
        #     top_outputs[out_pos] = shown_tmp

        #     out[out_pos].append(self.model.output_names[np.argmax(model_out[i])])

        #     if output_string == tests.iloc[out_pos]['output']:
        #         out[out_pos].append(float(model_out[i][top_ind]))

        #     # To make sharing test trees between models slightly easier we fall back to looking for different
        #     # capitalizations of the output if the original one doesn't exist
        #     token_to_check = tests.iloc[out_pos]['output']
        #     if token_to_check not in self._output_name_to_index:
        #         if token_to_check.capitalize() in self._output_name_to_index:
        #             token_to_check = token_to_check.capitalize()
        #         elif token_to_check.lower() in self._output_name_to_index:
        #             token_to_check = token_to_check.lower()
            
        #     # multiple tokens can be checked at the same time with templates
        #     out_val = np.nan
        #     ind = self._output_name_to_index.get(token_part, None)
        #     if ind is not None and model_out[i] is not None:
        #         sorted_values = np.argsort(model_out[i])
        #         topk = topk_threshold_ind(ind, sorted_values, self.topk)

        #         if self.method == "dirichlet":
        #             raw_score = compute_dirichlet_score(
        #                 ind, model_out[i], self.topk,
        #                 concentration=self.dirichlet_concentration,
        #                 # we treat values less than 10% of the topk value as unlikely to impact the results
        #                 # this is used to avoid unnecessary computation
        #                 domination_threshold=model_out[i][topk] / 10 
        #             )

        #             if test_type == "{} should output {}":
        #                 score = raw_score
        #             else:
        #                 score = -raw_score

        #         if np.isnan(model_out[i][ind]):
        #             score = np.nan
        #         elif model_out[i][ind] > model_out[i][topk]:
        #             if self.method == "dirichlet":
        #                 # mval = 1 / len(model_out[i]) if self.topk == 1 else 0 # minimum value possible while being at the top
        #                 # score = (raw_score - mval) / (1 - mval) # scale from 0 to 1
        #                 score = raw_score
        #             else:
        #                 score = model_out[i][ind] - model_out[i][topk]
        #         else:
        #             if self.method == "dirichlet":
        #                 # mval = 1 / (self.topk + 1) # maximum value possible while not being at the top
        #                 # score = (raw_score - 1 + mval) / (1 - mval) # scale from 0 to 1
        #                 score = raw_score - 1
        #             else:
        #                 mask = (model_out[i] <= model_out[i][topk]) & (model_out[i] > model_out[i][ind])
        #                 score = (model_out[i][ind] - model_out[i][mask]).sum()
        #         if test_type == "{} should output {}":
        #             score *= -1
        #         # out_val = max(score, out_val)
        #         out[out_pos].append(score)
        #     # out[out_pos] = max(out[out_pos], out_val)
        #     i += 1


        #     # out_pos += 1
        # return {
        #     "scores": out,
        #     "outputs": outputs,
        #     "value2_outputs": value2_outputs
        # }

    def suggest_outputs(self, current, num_suggestions=20):
        prompt = ""
        for c in current:
            prompt += '"'+c+'"\n'
        prompt += '"{output}'
        response = openai.Completion.create(
            engine='curie-instruct-beta', prompt=[prompt.format(output=o) for o in self.output_names], max_tokens=0, # self.engine
            temperature=0, n=1, stop='\"', logprobs=0, echo=True
        )
        lines = [sum(choice["logprobs"]["token_logprobs"][11:]) for choice in response["choices"]]
        pairs = list([v for v in zip(lines, self.output_names) if v[1] not in current])
        pairs.sort()
        return [v[1] for v in list(reversed(pairs))[:num_suggestions]]


class ClassifierScorerOld(Scorer):
    """ Wraps a model and defines a callable scorer that returns a score value for any input/output pair.

    Positive scores indicate test failures, positive scores indicate tests that pass. For example if we wrap
    a text sentiment classifer the `scorer(TestTree([("this is great!", "should be", "POSITIVE")]))` will return
    a large positive value indicating that the model is very likely to correctly produce that output when given
    that input.
    """

    def __init__(self, model, topk=1, output_names=None, method="dirichlet", dirichlet_concentration=10):
        """ Create a new scorer given a model that returns a probability vector for each input string.
        
        Parameters:
        -----------
        model : callable
            A model that is callable with a single argument (which is a list of strings) and returns a matrix of outputs.

        topk : int
            The number of top outputs to consider when scoring tests. For example topk=2 causes "should not be" tests to
            check the top two model outputs.

        output_names : list of strings
            A list of strings that correspond to the outputs of the model. If None, model.output_names is used.

        method : 'margin' or 'dirichlet'
            The scoring method to use. Dirichlet is preferred, but margin is available for backwards compatibility.

        dirichlet_concentration : float
            The concentration parameter for the dirichlet scoring method. It is in the units o pseudo-counts where larger
            values lead to a tighter prior centered around the model's probability outputs (so scores are more likely to
            be -1 or +1).
        """
        super().__init__(model)

        # we don't want to re-init a class if init has alrady been done (this can happen when Scorer(maybe_scorer) is called)
        if hasattr(self, "supported_test_types"):
            return # already initialized

        self.supported_test_types = [
            "{} should output {}",
            "{} should not output {}",
            "{} should have the same output as {}",
            "{} should not be more {} than {}",
            #"{} should be invertable. [] is the output.",
            #"{} should not output text containing {}",
            #"{} should output toxic text. [] is the output.",
            #"{} should not output toxic text. [] is the output."
        ]
        # {} should output {}
        # {} should not output {}
        # {} should have the same output as {}
        # {} should not be more {} than {}
        # {} should be invertable. [] is the output.
        # {} should not output text containing {}
        # {} should output toxic text. [] is the output.
        # {} should not output toxic text. [] is the output.

        # extract output names from the model if they are not provided directly
        if output_names is None:
            self.output_names = self.model.output_names
        else:
            self.output_names = output_names
        
        if not callable(self.output_names):
            self._output_name_to_index = {v: i for i, v in enumerate(self.output_names)}
        self.topk = topk
        self.output_type = "classification"
        self.method = method
        self.dirichlet_concentration = dirichlet_concentration

    def __call__(self, tests, output_bias=0.5):
        """ Score a set of tests.

        Parameters
        ----------
        tests : pandas.DataFrame
            A dataframe of tests.

        output_bias : float
            How much to bias the output entries in a test towards the actual output from the model.
        """
        if self.output_type == "classification":
            eval_inputs = []
            eval_inds = []
            variations1 = []
            variations2 = []
            for i, (k, test) in enumerate(tests.iterrows()):
                if test.type == "{} should not output {}" or test.type == "{} should output {}":
                    v1 = expand_template(test.value1)
                    for s1 in v1:
                        eval_inputs.append(s1)
                        eval_inds.append(i)
                    variations1.append(v1)
                    variations2.append(None)
                elif test.type == "{} should have the same output as {}":
                    # eval_inputs.append(test.value1)
                    # eval_inputs.append(test.value2)
                    v1 = expand_template(test.value1)
                    v2 = expand_template(test.value2)
                    for s1 in v1:
                        for s2 in v2:
                            eval_inputs.append(s1)
                            eval_inputs.append(s2)
                            eval_inds.append(i)
                            eval_inds.append(i)
                    variations1.append(v1)
                    variations2.append(v2)

            try:
                model_out = self.model(eval_inputs)
            except Exception as e:
                model_out = np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
                log.error(e)
                log.error(eval_inputs)
                log.error("The model threw an exception when evaluating inputs! We are patching this disaster with np.nan for the sake of the user study!")

            out = [[] for _ in range(tests.shape[0])]
            out_pos = 0
            i = 0
            value1_outputs = [{} for _ in range(tests.shape[0])]
            value2_outputs = [{} for _ in range(tests.shape[0])]
            while i < len(model_out):
                out_pos = eval_inds[i]

                test_type = tests.iloc[out_pos]["type"]
                if test_type == "{} should not output {}" or test_type == "{} should output {}":

                    # save the top model outputs
                    inds = np.argsort(-model_out[i])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i][j])
                    value1_outputs[out_pos] = shown_tmp

                    # TODO: here we need to not assume the LM genreates the output, but that it just generates the inputs
                    # then we can embed the tests for each of the top 10? 100? outputs and rank them by how well their
                    # embeddings match the test embeddings for the prompt
                    token_to_check = tests.iloc[out_pos]['value2']

                    # To make sharing test trees between models slightly easier we fall back to looking for different
                    # capitalizations of the output if the original one doesn't exist
                    if token_to_check not in self._output_name_to_index:
                        if token_to_check.capitalize() in self._output_name_to_index:
                            token_to_check = token_to_check.capitalize()
                        elif token_to_check.lower() in self._output_name_to_index:
                            token_to_check = token_to_check.lower()
                    
                    # multiple tokens can be checked at the same time with templates
                    out_val = np.nan
                    for token_part in expand_template(token_to_check):
                        ind = self._output_name_to_index.get(token_part, None)
                        if ind is not None and model_out[i] is not None:
                            sorted_values = np.argsort(model_out[i])
                            topk = topk_threshold_ind(ind, sorted_values, self.topk)

                            if self.method == "dirichlet":
                                raw_score = compute_dirichlet_score(
                                    ind, model_out[i], self.topk,
                                    concentration=self.dirichlet_concentration,
                                    # we treat values less than 10% of the topk value as unlikely to impact the results
                                    # this is used to avoid unnecessary computation
                                    domination_threshold=model_out[i][topk] / 10 
                                )

                                if test_type == "{} should output {}":
                                    score = raw_score
                                else:
                                    score = -raw_score

                            if np.isnan(model_out[i][ind]):
                                score = np.nan
                            elif model_out[i][ind] > model_out[i][topk]:
                                if self.method == "dirichlet":
                                    # mval = 1 / len(model_out[i]) if self.topk == 1 else 0 # minimum value possible while being at the top
                                    # score = (raw_score - mval) / (1 - mval) # scale from 0 to 1
                                    score = raw_score
                                else:
                                    score = model_out[i][ind] - model_out[i][topk]
                            else:
                                if self.method == "dirichlet":
                                    # mval = 1 / (self.topk + 1) # maximum value possible while not being at the top
                                    # score = (raw_score - 1 + mval) / (1 - mval) # scale from 0 to 1
                                    score = raw_score - 1
                                else:
                                    mask = (model_out[i] <= model_out[i][topk]) & (model_out[i] > model_out[i][ind])
                                    score = (model_out[i][ind] - model_out[i][mask]).sum()
                            if test_type == "{} should output {}":
                                score *= -1
                            # out_val = max(score, out_val)
                            out[out_pos].append(score)
                    # out[out_pos] = max(out[out_pos], out_val)
                    i += 1
                elif test_type == "{} should have the same output as {}":

                    # save the top model outputs
                    inds = np.argsort(-model_out[i])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i][j])
                    value1_outputs[out_pos] = shown_tmp
                    inds = np.argsort(-model_out[i+1])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i+1][j])
                    value2_outputs[out_pos] = shown_tmp
                    
                    if np.isnan(model_out[i][0]):
                        score = np.nan
                    elif self.method == "dirichlet":
                        score = compute_dirichlet_equality_score(
                            model_out[i], model_out[i+1], self.topk, concentration=self.dirichlet_concentration,
                            # we treat values less than 1% of the top value as unlikely to impact the results
                            # this is used to avoid unnecessary computation (note this is used more aggressivly than for the non-equality score)
                            domination_threshold=min(np.max(model_out[i]), np.max(model_out[i+1])) / 100
                        )
                    else:
                        score = equality_score(model_out[i], model_out[i+1])
                    # out[out_pos] = max(out[out_pos], score)
                    out[out_pos].append(score)
                    i += 2
                else:
                    raise Exception(f"Test type '{test_type}' not yet supported!")

                # out_pos += 1
            return {
                "scores": out,
                "value1_outputs": value1_outputs,
                "value2_outputs": value2_outputs
            }
        else:
            raise Exception(f"Output type {self.output_type} not yet supported!")

    def suggest_outputs(self, current, num_suggestions=20):
        prompt = ""
        for c in current:
            prompt += '"'+c+'"\n'
        prompt += '"{output}'
        response = openai.Completion.create(
            engine='curie-instruct-beta', prompt=[prompt.format(output=o) for o in self.output_names], max_tokens=0, # self.engine
            temperature=0, n=1, stop='\"', logprobs=0, echo=True
        )
        lines = [sum(choice["logprobs"]["token_logprobs"][11:]) for choice in response["choices"]]
        pairs = list([v for v in zip(lines, self.output_names) if v[1] not in current])
        pairs.sort()
        return [v[1] for v in list(reversed(pairs))[:num_suggestions]]

class GeneratorScorer(Scorer):
    """ Wraps a text generation model in a scorer that can score the target model against tests.
    """

    def __init__(self, model, completions=1, reverse_model=None, feature_models={}, similarity_threshold=0.9):
        """ Initializes a new scorer for a given target model.

        Parameters
        ----------
        model : callable
            The model we are scoring against the tests. It is expected to be a function that takes a list of strings as
            input and returns a list of strings as output.

        completions : int
            The number of completions to generate for each model input.

        reverse_model : callable
            The inverse of model, such that x = reverse_model(model(x)). Note that this is optional and only required
            to use the 'should be invertable' test type. If for example model is an EN to ES translation model, then
            reverse_model should be an ES to EN translation model. Just like model, reverse_model is expected to take a
            list of strings as input and return a list of strings as output.

        feature_models : dict
            A dictionary of classifiers that can be used to score the generated output. The keys are the names of the
            classifier label that is predicted and the values are the classifiers themselves. Each classifer is expected
            to take a list of strings as input and return a list of floats between 0 and 1.

        similarity_threshold : float
            The threshold for the similarity between the generated output and the expected output. If set to 1.0, then
            exact string matching is performed. If < 1.0 then this is the threshold on the sementantic similarity (a cos
            similarity score between 0 and 1).
            TODO: this was just used for the 'should be invertable' test type. Should we use it for other test types as well?
        """
        super().__init__(model)

        # we don't want to re-init a class if init has alrady been done (this can happen when Scorer(maybe_scorer) is called)
        if hasattr(self, "supported_test_types"):
            return # already initialized

        self.completions = completions
        self.reverse_model = reverse_model
        self.similarity_threshold = similarity_threshold
        self.feature_models = feature_models

        self.supported_test_types = [
            "{} can output {}",
            "{} can only output {}",
            "{} cannot output {}", # TODO: should this use semantic similarity instead of exact string match?
            "{} can have the same output as {}",
            "{} can output text containing {}",
            "{} cannot output text containing {}",
            "{} can output text starting with {}",
            "{} cannot output text starting with {}",
            "{} can output text that is {}",
            "{} cannot output text that is {}",
            "{} cannot be completed to become {}"
        ]
        # '"{}[]" should not become "{}" when completed. 

        # see if we can support the inversion test
        if self.reverse_model is not None and adatest.embedding_model is not None: # TODO: do we need an embeddig model or just use the backend?
            self.supported_test_types.append("{} should be invertable. [] is the output.") # Note that {} means user-editable, [] means read-only

        # # see if we have user-provided classifier tests
        # for k in self.feature_models:
        #     self.supported_test_types.extend([
        #         "{} should output "+k+" text. [] is the output.",
        #         "{} should not output "+k+" text. [] is the output."
        #     ])

        

    def __call__(self, tests):

        # run the model on the inputs
        eval_inputs = []
        eval_inds = []
        eval_reverse_pos = []
        variations1 = []
        variations2 = []
        eval_output_feature_pos = {k: [] for k in self.feature_models}
        eval_input_feature_pos = {k: [] for k in self.feature_models}
        eval_io_feature_pos = {k: [] for k in self.feature_models}
        for i, (k, test) in enumerate(tests.iterrows()):
            if test.type in ["{} should not output {}", "{} should output {}",
                             "{} should not output text containing {}", "{} should output text containing {}",
                             "{} should not output text starting with {}", "{} should output text starting with {}"]:
                v1 = expand_template(test.value1)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                variations1.append(v1)
                variations2.append(None)
            elif test.type == "{} should not output text that is {}" or test.type == "{} should output text that is {}":
                v1 = expand_template(test.value1)
                v2 = expand_template(test.value2)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                    for s2 in v2: # mark each feature in the template for evaluation
                        if s2 in eval_output_feature_pos:
                            eval_output_feature_pos[s2].append(len(eval_inputs)-1)
                variations1.append(v1)
                variations2.append(None)
            elif test.type == "{} should not be completed to become {}":
                v1 = expand_template(test.value1)
                v2 = expand_template(test.value2)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                    for s2 in v2: # mark each feature in the template for evaluation
                        if s2 in eval_input_feature_pos:
                            eval_io_feature_pos[s2].append(len(eval_inputs)-1)
                            eval_input_feature_pos[s2].append(len(eval_inputs)-1)
                variations1.append(v1)
                variations2.append(None)
            elif test.type == "{} should be invertable.":
                v1 = expand_template(test.value1)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                    eval_reverse_pos.append(len(eval_inputs) - 1)
            elif test.type == "{} should have the same output as {}":
                v1 = expand_template(test.value1)
                v2 = expand_template(test.value2)
                for s1 in v1:
                    for s2 in v2:
                        eval_inputs.append(s1)
                        eval_inputs.append(s2)
                        eval_inds.append(i)
                        eval_inds.append(i)
                variations1.append(v1)
                variations2.append(v2)
        # try:
            # model_out = self.model(eval_inputs) {} should {} {}
        model_out = self.model(eval_inputs, completions=self.completions) # now a list of lists
        # except Exception as e:
        #     model_out = [["__ERROR__"]*self.completions for _ in range(len(eval_inputs))]#np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
        #     log.error(e)
        #     log.error(eval_inputs)
        #     log.error("The model threw an exception when evaluating inputs! We are patching this disaster with '__ERROR__'!")

        # run feature models when we need to
        model_output_feature = {}
        model_input_feature = {}
        model_io_feature = {}
        for k in self.feature_models:
            if len(eval_output_feature_pos[k]) > 0:
                feature_out_flat = self.feature_models[k]([v for ind in eval_output_feature_pos[k] for v in model_out[ind]])
                feature_out = np.reshape(feature_out_flat, (-1, 2))
                model_output_feature[k] = [None for _ in model_out]
                for i, ind in enumerate(eval_output_feature_pos[k]):
                    model_output_feature[k][ind] = feature_out[i]
            if len(eval_input_feature_pos[k]) > 0:
                feature_out = self.feature_models[k]([eval_inputs[ind] for ind in eval_input_feature_pos[k]])
                model_input_feature[k] = [None for _ in model_out]
                for i, ind in enumerate(eval_input_feature_pos[k]):
                    model_input_feature[k][ind] = feature_out[i]
            if len(eval_io_feature_pos[k]) > 0:
                feature_out_flat = self.feature_models[k]([eval_inputs[ind] + v for ind in eval_io_feature_pos[k] for v in model_out[ind]])
                feature_out = np.reshape(feature_out_flat, (-1, self.completions))
                model_io_feature[k] = [None for _ in model_out]
                for i, ind in enumerate(eval_io_feature_pos[k]):
                    model_io_feature[k][ind] = feature_out[i]

        # feature_model_inputs = {k: [] for k in self.feature_models}
        # i = 0
        # while i < len(model_out):
        #     if test.type == "{} should not output text that is {}" or test.type == "{} should output text that is {}":
        #         if test.value1 in self.feature_models:
        #             feature_model_inputs[test.value1].append((i, model_out[i]))
        #     i += 1

        # run the reverse model on any outputs we need to
        # eval_reverse_inputs = []
        
        # for i, (k, test) in enumerate(tests.iterrows()):
        #     if test.comparator == "should be invertable.":
        #         v1 = expand_template(test.value1)
        #         for s1 in v1:
        #             eval_reverse_inputs.append(s1)
        #             eval_reverse_inds.append(i)
        if len(eval_reverse_pos) > 0:
            model_reverse_out = [None for _ in model_out]
            input_embed = [None for _ in model_out]
            round_trip_embed = [None for _ in model_out]
            try:
                # compute input embedding
                tmp = adatest.embedding_model.encode([eval_inputs[ind] for ind in eval_reverse_pos], convert_to_tensor=True, show_progress_bar=False).cpu()
                for i, ind in enumerate(eval_reverse_pos):
                    input_embed[ind] = tmp[i]

                # compute reverse model output
                reverse_out = self.reverse_model([model_out[ind] for ind in eval_reverse_pos])
                for i, ind in enumerate(eval_reverse_pos):
                    model_reverse_out[ind] = str(reverse_out[i])

                # compute round trip embedding
                tmp = adatest.embedding_model.encode(reverse_out, convert_to_tensor=True, show_progress_bar=False).cpu()
                for i, ind in enumerate(eval_reverse_pos):
                    round_trip_embed[ind] = tmp[i]

            except Exception as e:
                model_reverse_out = ["ERROR" for _ in range(len(model_out))]
                log.error(e)
                log.error("The reverse model threw an exception when evaluating inputs! We are patching this disaster with 'ERROR' for the sake of the user study!")
        else:
            model_reverse_out = []

        out = [[] for _ in range(tests.shape[0])]
        out_pos = 0
        i = 0
        value1_outputs = [{} for _ in range(tests.shape[0])]
        value2_outputs = [{} for _ in range(tests.shape[0])]
        while i < len(model_out):
            out_pos = eval_inds[i]

            test_type = tests.iloc[out_pos]["type"]
            if test_type in ["{} should not output {}", "{} should output {}",
                             "{} should output text containing {}", "{} should not output text containing {}",
                             "{} should output text starting with {}", "{} should not output text starting with {}"]:
                invert = -1 if test_type in ["{} should not output {}", "{} should not output text containing {}", "{} should not output text starting with {}"] else 1
                contain_check = test_type == "{} should not output text containing {}" or test_type == "{} should output text containing {}"
                starting_with_check = test_type == "{} should not output text starting with {}" or test_type == "{} should output text starting with {}"
                
                # auto fill missing outputs TODO: this is a hack
                if tests.iloc[out_pos]['value2'] is None:
                    tests.loc[tests.index[out_pos], 'value2'] = str(model_out[i][0])
                
                # save the model output
                value1_outputs[out_pos] = {}
                value1_outputs[out_pos]["string"] = list(model_out[i])

                # multiple tokens can be checked at the same time with templates
                for token_part in expand_template(tests.iloc[out_pos]['value2']):
                    for c in range(self.completions):
                        if contain_check:
                            out[out_pos].append(-invert if token_part in model_out[i][c] else invert)
                        elif starting_with_check:
                            out[out_pos].append(-invert if token_part.startswith(model_out[i][c]) else invert)
                        else:
                            out[out_pos].append(-invert if model_out[i][c] == token_part else invert)
                i += 1
            elif test_type == "{} should output text that is {}" or test_type == "{} should not output text that is {}":
                should_be = test_type == "{} should output text that is {}"

                # multiple features can be checked at the same time with templates
                for feature in expand_template(tests.iloc[out_pos]['value2']):
                    if feature in model_output_feature:
                        feature_prob = model_output_feature[feature][i]
                        if should_be:
                            score = (0.5 - feature_prob) * 2
                        else:
                            score = (feature_prob - 0.5) * 2
                    else:
                        score = np.nan
                    out[out_pos].append(score)
                
                # save the model round trip output
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][model_out[i]] = 1

                i += 1
            
            elif test_type == "{} should not be completed to become {}":

                # multiple features can be checked at the same time with templates
                io_feature_probs = []
                input_feature_probs = []
                for feature in expand_template(tests.iloc[out_pos]['value2']):
                    for c in range(self.completions):
                        if feature in model_io_feature:# and feature in model_input_feature is implied
                            io_feature_prob = model_io_feature[feature][i][c]
                            input_feature_prob = model_input_feature[feature][i]
                            if input_feature_prob <= 0.5 and io_feature_prob <= 0.5 or input_feature_prob >= 0.5 and io_feature_prob >= 0.5:
                                score = io_feature_prob - input_feature_prob - 0.5
                            else: # if input_feature_prob >= 0.5 and io_feature_prob <= 0.5 or (input_feature_prob <= 0.5 and io_feature_prob >= 0.5)
                                score = io_feature_prob - input_feature_prob
                        else:
                            input_feature_prob = io_feature_prob = 0
                            score = np.nan
                        out[out_pos].append(score)
                        input_feature_probs.append(input_feature_prob)
                        io_feature_probs.append(io_feature_prob)
                
                # save the model output
                value1_outputs[out_pos] = {}
                value1_outputs[out_pos]["string"] = list(model_out[i])

                max_score_ind = np.argmax(out[out_pos])
                value2_outputs[out_pos] = {}
                value2_outputs[out_pos]["input prob"] = input_feature_probs[max_score_ind]
                value2_outputs[out_pos]["input+output prob"] = io_feature_probs[max_score_ind]

                i += 1
            
            elif test_type == "{} should be invertable.":
                
                # compare embedding distances
                score = sentence_transformers.util.pytorch_cos_sim(input_embed[i], round_trip_embed[i]).numpy()[0][0]
                out[out_pos].append(self.similarity_threshold-score)

                # update the output since it is always computed in inversion tests
                tests.loc[tests.index[out_pos], 'value2'] = str(model_reverse_out[i])
                
                # save the model round trip output
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][str(model_out[i])] = 1

                i += 1
            elif test_type == "{} should have the same output as {}":

                # save the model outputs
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][model_out[i]] = 1
                value2_outputs[out_pos]  = {}
                value2_outputs[out_pos][model_out[i+1]] = 1
                
                # save the score
                out[out_pos].append(1 if model_out[i] == model_out[i+1] else -1)
                i += 2
            else:
                raise Exception(f"Test type type '{test_type}' not yet supported!")

            # out_pos += 1
        return {
            "scores": out,
            "value1_outputs": value1_outputs,
            "value2_outputs": value2_outputs
        }


def expand_template(s, keep_braces=False):
    """ Expand a template string into a list of strings.
    """
    # parts = []
    # for s in strings:
    matches = re.findall("{[^}]*}", s)
    s = re.sub("{[^}]*}", "{}", s)
    template_groups = [str(m)[1:-1].split("|") for m in matches]
    try:
        if keep_braces:
            return [s.format(*['{{{p}}}' for p in parts]) for parts in itertools.product(*template_groups)]
        else:
            return [s.format(*parts) for parts in itertools.product(*template_groups)]
    except ValueError:
        return [s] # we return the template not filled in if it is invalid

def clean_template(s):
    """ This removes duplicate template entries.
    """
    matches = re.findall("{[^}]*}", s)
    s = re.sub("{[^}]*}", "{}", s)
    template_groups = [str(m)[1:-1].split("|") for m in matches]
    clean_groups = ["{"+"|".join(list({v: None for v in g}.keys()))+"}" for g in template_groups]
    try:
        return s.format(*clean_groups)
    except ValueError:
        return s # we return the template not cleaned in if it is invalid

def topk_threshold_ind(ind, sorted_values, k):
    """ Return the threshold value for which if ind dropped below it would not be in the top k (without other scores changing).
    """
    if ind in sorted_values[-k:]:
        topk = sorted_values[-k - 1]
    else:
        topk = sorted_values[-k]
    if topk == ind:
        topk = sorted_values[-k - 1]
    return topk

def compute_dirichlet_score(ind, model_output, k=1, concentration=10, empirical_samples=1000, domination_threshold=1e-5):
    """ Compute the probability that ind is in the top k set of probabilities.

    This is done by sampling from a Dirichlet distribution with concentration parameter centered around the model output
    (by default assuming a concentration weight of 10 pseudo-counts).

    Parameters
    ----------
    ind : int
        The index to compute the score for.

    model_output : np.array
        The model output probabilities.

    k : int
        The number of top k probabilities to consider ind in.

    concentration : float
        The concentration parameter for the Dirichlet distribution. Larger values make the distribution have lower variance.

    empirical_samples : int
        We can't calculate the probability of ind being in the top k set of probabilities exactly, so we sample from the
        distribution to get an empirical estimate. This controls the number of samples to take.

    domination_threshold : float
        Below this value we assume that these output dims can be safely ignored. This is used to avoid unnessecary computation.
    """

    # a small heuristic tie breaker to allow better ranking of small effects
    # this helps when the effect sizes are so small the empirical_samples can't capture the differences
    tie_breaker = model_output[ind]*1e-4
    
    # if our output of interest is dominated then we just skip the work and return essentially 0
    if model_output[ind] < domination_threshold:
        return tie_breaker
    
    # shrink the number of dims we have to deal with by collapsing low probability dims
    bundles = []
    bundle = []
    bundle_sizes = []
    inds = np.argsort(model_output)
    bundle_size = 0
    new_ind = -1
    for i,sind in enumerate(inds):
        if bundle_size + model_output[sind] < domination_threshold:
            bundle.append(sind)
            bundle_size += model_output[sind]
        else:
            if len(bundle) > 0:
                bundles.append(bundle)
                bundle_sizes.append(bundle_size)

            if sind == ind:
                new_ind = len(bundles)
            bundle = [sind]
            bundle_size = model_output[sind]
    bundles.append(bundle)
    bundle_sizes.append(bundle_size)
    
    # normalize the scores for the Dirichlet parameter
    normed_output = np.array(bundle_sizes) + 1e-6
    normed_output /= normed_output.sum()
    normed_output *= concentration
    
    if k == 1:
        sort_inds = np.argmax(scipy.stats.dirichlet.rvs(normed_output, empirical_samples, random_state=0), 1)
        return min(1, (sort_inds == new_ind).mean() + tie_breaker)
    else:
        sort_inds = np.argsort(-scipy.stats.dirichlet.rvs(normed_output, empirical_samples, random_state=0), 1)
        return min(1, ((sort_inds[:,:k] - new_ind) == 0).sum() / sort_inds.shape[0] + tie_breaker)

def compute_dirichlet_equality_score(model_output1, model_output2, k=1, concentration=10, empirical_samples=1000, domination_threshold=1e-5):
    """ Compute the probability that ind is in the top k set of probabilities.

    This is done by sampling from a Dirichlet distribution with concentration parameter centered around the model output
    and assuming a concentration weight of 10 pseudo-counts.

    Parameters
    ----------
    ind : int
        The index to compute the score for.

    model_output : np.array
        The model output probabilities.

    k : int
        The number of top k probabilities to consider ind in.

    concentration : float
        The concentration parameter for the Dirichlet distribution. Larger values make the distribution have lower variance.

    empirical_samples : int
        We can't calculate the probability of ind being in the top k set of probabilities exactly, so we sample from the
        distribution to get an empirical estimate. This controls the number of samples to take.

    domination_threshold : float
        Below this value we assume that these output dims can be safely ignored. This is used to avoid unnessecary computation.
    """

    assert len(model_output1) == len(model_output2)

    # shrink the number of dims we have to deal with by collapsing low probability dims
    used_inds = [i for i in range(len(model_output1)) if model_output1[i] > domination_threshold or model_output2[i] > domination_threshold]
    # model_output1 = model_output1[used_inds]
    # model_output2 = model_output2[used_inds]
    model_output1_padded = np.zeros(len(used_inds) + 1)
    model_output1_padded[1:] = model_output1[used_inds]
    model_output1_padded[0] = 1 - np.sum(model_output1)
    model_output2_padded = np.zeros(len(used_inds) + 1)
    model_output2_padded[1:] = model_output2[used_inds]
    model_output2_padded[0] = 1 - np.sum(model_output2)

    assert model_output1_padded[0] >= -1e-6 and model_output2_padded[0] >= -1e-6, "The given model output probabilities do not sum to 1!"
    model_output1_padded[0] = max(model_output1_padded[0], 0)
    model_output2_padded[0] = max(model_output2_padded[0], 0)

    # normalize the scores for the Dirichlet parameter
    normed_output1 = np.array(model_output1_padded) + 1e-6
    normed_output1 /= normed_output1.sum()
    normed_output1 *= concentration
    normed_output2 = np.array(model_output2_padded) + 1e-6
    normed_output2 /= normed_output2.sum()
    normed_output2 *= concentration
    
    if k == 1:
        sort_inds1 = np.argmax(scipy.stats.dirichlet.rvs(normed_output1, empirical_samples, random_state=0), 1)
        sort_inds2 = np.argmax(scipy.stats.dirichlet.rvs(normed_output2, empirical_samples, random_state=0), 1)

        # the average number of matches, excluding the first position (which is a bucket for all dominated, low prob, dims)
        match_rate = ((sort_inds1 - sort_inds2 == 0) * (sort_inds1 != 0)).mean()

        if np.argmax(model_output1) == np.argmax(model_output2):
            return -match_rate
        else:
            return 1 - match_rate
    else:
        raise Exception("The 'should be the same as for' is not implemented for topk > 1!")

def equality_score(output_values1, output_values2, topk=1):
    assert topk == 1
    ind1 = np.argmax(output_values1)
    ind2 = np.argmax(output_values2)
    max1 = output_values1[ind1]
    max2 = output_values2[ind2]
    margins = np.zeros(len(output_values1))

    if ind1 != ind2:
        min_margin = 1e6
        for i in range(len(output_values1)):
            score1 = max(0, max1 - output_values1[i])
            score2 = max(0, max2 - output_values2[i])
            margin = score1 + score2
            if margin < min_margin:
                min_margin = margin
        return min_margin
    else:
        val1 = output_values1[ind1]
        output_values1[ind1] = np.nan
        score1 = val1 - np.nanmax(output_values1)
        output_values1[ind1] = val1

        val2 = output_values2[ind2]
        output_values2[ind2] = np.nan
        score2 = val2 - np.nanmax(output_values2)
        output_values2[ind2] = val2
        return -min(score1, score2)
