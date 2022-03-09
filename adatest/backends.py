""" A set of backends for AdaTest.
"""
import asyncio
import aiohttp
import transformers
import openai
import numpy as np
import copy
from ._filters import clean_string

class Backend():
    """ Abstract class for language model backends.
    """
    
    def __init__(self, models, sep, subsep, quote, filter_profanity):
        """ Create a new backend with the given separators and max length.
        """
        if not isinstance(models, list) and not isinstance(models, tuple):
            models = [models]
        self.model = models[0]
        self.models = models
        self.subsep = subsep
        self.sep = sep
        self.quote = quote
        self.filter_profanity = filter_profanity

        # value1_format = '"{value1}"'
        # value2_format = '"{value1}"'
        # value1_value2_format = '"{value1}" {comparator} "{value2}"'
        # value1_comparator_value2_format = '"{value1}" {comparator} "{value2}"'
        # value1_comparator_format = '"{value1}" {comparator} "{value2}"'
        # comparator_value2_format = '"{value1}" {comparator} "{value2}"'


    
    def __call__(self, prompts, max_length=None):
        """ This should be overridden by concrete subclasses.

        Parameters:
        -----------
        prompts: list of tuples, or list of lists of tuples
        """
        pass
    
    def _validate_prompts(self, prompts):
        """ Ensure the passed prompts are a list of prompt lists.
        """
        if isinstance(prompts[0][0], str):
            prompts = [prompts]
        return prompts
    
    def _varying_values(self, prompts, topic):
        """ Marks with values vary in the set of prompts.
        """

        show_topics = False
        gen_value1 = False
        gen_value2 = False
        gen_value3 = False
        for prompt in prompts:
            topics, value1s, value2s, value3s = zip(*prompt)
            show_topics = show_topics or len(set(list(topics) + [topic])) > 1
            gen_value1 = gen_value1 or len(set(value1s)) > 1
            gen_value2 = gen_value2 or len(set(value2s)) > 1
            gen_value3 = gen_value3 or len(set(value3s)) > 1
        return show_topics, gen_value1, gen_value2, gen_value3
    
    def _create_prompt_strings(self, prompts, topic):
        """ Convert prompts that are lists of tuples into strings for the LM to complete.
        """

        show_topics, gen_value1, gen_value2, gen_value3 = self._varying_values(prompts, topic)

        prompt_strings = []
        for prompt in prompts:
            topics, value1s, value2s, value3s = zip(*prompt)
            prompt_string = ""
            for p_topic, value1, value2, value3 in prompt:
                if show_topics:
                    prompt_string += self.sep + p_topic + ":" + self.sep + self.quote
                else:
                    prompt_string += self.quote
                
                if gen_value1:
                    prompt_string += value1 + self.quote
                if gen_value1 and (gen_value2 or gen_value3):
                    prompt_string += self.subsep + self.quote
                if gen_value2:
                    prompt_string += value2 + self.quote
                if gen_value2 and gen_value3:
                    prompt_string += self.subsep + self.quote
                if gen_value3:
                    prompt_string += value3 + self.quote
                prompt_string += self.sep
            if show_topics:
                prompt_strings.append(prompt_string + self.sep + topic + ":" + self.sep + self.quote)
            else:
                prompt_strings.append(prompt_string + self.quote)
        return prompt_strings
    
    def _parse_suggestion_texts(self, suggestion_texts, prompts):
        """ Parse the suggestion texts into tuples.
        """
        assert len(suggestion_texts) % len(prompts) == 0, "Missing prompt completions!"

        _, gen_value1, gen_value2, gen_value3 = self._varying_values(prompts, "") # note that "" is an unused topic argument
        
        num_samples = len(suggestion_texts) // len(prompts)
        samples = []
        for i, suggestion_text in enumerate(suggestion_texts):
            if self.filter_profanity:
                suggestion_text = clean_string(suggestion_text)
            prompt_ind = i // num_samples
            prompt = prompts[prompt_ind]
            suggestion = suggestion_text.split(self.quote+self.subsep+self.quote)
            # if len(self.quote) > 0: # strip any dangling quote
            #     suggestion[-1] = suggestion[-1][:-len(self.quote)]
            if not gen_value1:
                suggestion = [prompt[0][0]] + suggestion
            if not gen_value2:
                suggestion = suggestion[:1] + [prompt[0][2]] + suggestion[1:]
            if not gen_value3:
                suggestion = suggestion[:2] + [prompt[0][3]]

            if len(suggestion) == 3:
                samples.append(tuple(suggestion))
        return samples

           
class Transformers(Backend):
    def __init__(self, model, tokenizer, sep="\n", subsep=" ", quote="\"", filter_profanity=True):
        super().__init__(models=model, sep=sep, subsep=subsep, quote=quote)
        self.tokenizer = tokenizer
        self.device = self.model.device

        class StopAtSequence(transformers.StoppingCriteria):
            def __init__(self, stop_string, tokenizer, window_size=10):
                self.stop_string = stop_string
                self.tokenizer = tokenizer
                self.window_size = 10
                self.max_length = None
                self.prompt_length = 0
                
            def __call__(self, input_ids, scores):
                if len(input_ids[0]) > self.max_length + self.prompt_length:
                    return True

                # we need to decode rather than check the ids directly because the stop_string may get enocded differently in different contexts
                return self.tokenizer.decode(input_ids[0][-self.window_size:])[-len(self.stop_string):] == self.stop_string

        self._sep_stopper = StopAtSequence(self.quote+self.sep, self.tokenizer)
    
    def __call__(self, prompts, topic, num_samples=1, max_length=100):
        
        prompts = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts, topic)
        
        # monkey-patch a method that prevents the use of past_key_values
        saved_func = self.model.prepare_inputs_for_generation
        def prepare_inputs_for_generation(input_ids, **kwargs):
            if "past_key_values" in kwargs:
                return {"input_ids": input_ids, "past_key_values": kwargs["past_key_values"]}
            else:
                return {"input_ids": input_ids}
        self.model.prepare_inputs_for_generation = prepare_inputs_for_generation
        
        # run the generative LM for each prompt
        suggestion_texts = []
        for prompt_string in prompt_strings:
            input_ids = self.tokenizer.encode(prompt_string, return_tensors='pt').to(self.device)
            cache_out = self.model(input_ids[:, :-1], use_cache=True)

            for _ in range(num_samples):
                self._sep_stopper.prompt_length = 1
                self._sep_stopper.max_length = max_length
                out = self.model.sample(
                    input_ids[:, -1:], pad_token_id=self.model.config.eos_token_id,
                    stopping_criteria=self._sep_stopper,
                    past_key_values=cache_out.past_key_values # TODO: enable re-using during sample unrolling as well
                )

                # we ignore first token because it is part of the prompt
                suggestion_text = self.tokenizer.decode(out[0][1:])
                
                # we ignore the stop string to match other backends
                if suggestion_text[-len(self._sep_stopper.stop_string):] == self._sep_stopper.stop_string:
                    suggestion_text = suggestion_text[:-len(self._sep_stopper.stop_string)]
                
                suggestion_texts.append(suggestion_text)

        # restore the old function that prevents the past_key_values argument from getting passed
        self.model.prepare_inputs_for_generation = saved_func
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)


class OpenAI(Backend):
    """ Backend wrapper for the OpenAI API that exposes GPT-3.
    """
    
    def __init__(self, models, api_key=None, sep="\n", subsep=" ", quote="\"", temperature=1.0, top_p=0.95, filter_profanity=True):
        super().__init__(models, sep=sep, subsep=subsep, quote=quote, filter_profanity=filter_profanity)
        self.temperature = temperature
        self.top_p = top_p
        if api_key is not None:
            openai.api_key = api_key

    def __call__(self, prompts, topic, test_type, scorer, num_samples=1, max_length=100):
        prompts = self._validate_prompts(prompts)
        # prompt_strings = self._create_prompt_strings(prompts, topic)

        # find out which values in the prompt have multiple values and so should be generated
        topics_vary, gen_value1, gen_value2, gen_value3 = self._varying_values(prompts, topic)

        # TODO: bias generation towards model failures
        # custom generation process for the "should not output" test
        # if test_type == "{} should not output {}":

        #     # create prompts to generate the inputs to the model
        #     input_prompt_strings, _, _, _ = self._create_prompt_strings(prompts, topic, prefix=[], generate=[1])

        #     # call the OpenAI API to complete the input generation prompts
        #     response = openai.Completion.create(
        #         engine=self.model, prompt=input_prompt_strings, max_tokens=max_length,
        #         temperature=self.temperature, top_p=self.top_p, n=num_samples, stop=self.quote
        #     )
        #     input_suggestions = [choice["text"] for choice in response["choices"]]

        #     output_scores = scorer.model(input_suggestions)
        #     per_completion_token_bias_values = self._compute_bias_values(output_scores, scorer.model.output_names)

        #     # create prompts to generate the outputs to the model
        #     output_prompt_strings, _, _, _ = self._create_prompt_strings(prompts, topic, prefix=[1], generate=[2])

        #     # call the OpenAI API to complete the output generation prompts
        #     response = openai.Completion.create(
        #         engine=self.model, prompt=output_prompt_strings, max_tokens=max_length,
        #         temperature=self.temperature, top_p=self.top_p, n=num_samples, stop=self.quote,
        #         per_completion_token_bias_values=per_completion_token_bias_values
        #     )
        #     output_suggestions = [choice["text"] for choice in response["choices"]]

        #     # then we build the final suggestions as the combination of the input and output suggestions



        # create prompts to generate the model input parameters of the tests
        prompt_strings = self._create_prompt_strings(prompts, topic)

        # see if we can stop at the first quote or need to wait for the seperator
        if gen_value1 + gen_value2 + gen_value3 == 1:
            stop_string = self.quote
        else:
            stop_string = self.quote+self.sep
        
        # call the OpenAI API to complete the prompts
        response = openai.Completion.create(
            engine=self.model, prompt=prompt_strings, max_tokens=max_length,
            temperature=self.temperature, top_p=self.top_p, n=num_samples, stop=stop_string
        )
        suggestion_texts = [choice["text"] for choice in response["choices"]]
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)

        # if self._should_generate_outputs(test_type, scorer):
        #     model_inputs = self._create_model_inputs(test_type, inputs_filled)
        #     outputs = self._generate_outputs(model_inputs, scorer)
        #     output_prompt_strings = self._create_output_prompt_strings(prompts, topic)

        #     for i in range(len(outputs)):
        #         logit_bias = {self._quote_token: 1} # we also upweight the quote token so we don't decrease the chance of ending the output
        #         for id in self._tokenizer(outputs[i])["input_ids"]:
        #             logit_bias[id] = logit_bias.get(id, 0) + 1
        #             response = openai.Completion.create(
        #                 engine=self.model, prompt=output_prompt_strings, max_tokens=max_length,
        #                 temperature=self.temperature, top_p=self.top_p, n=num_samples, stop=self.quote, logit_bias=logit_bias
        #             )
        #             suggestion_texts = [choice["text"] for choice in response["choices"]]
                






class AI21(Backend):
    """ Backend wrapper for the OpenAI API that exposes GPT-3.
    """
    
    def __init__(self, model, api_key, sep="\n", subsep=" ", quote="\"", temperature=0.95, filter_profanity=True):
        super().__init__(model, sep=sep, subsep=subsep, quote=quote, filter_profanity=filter_profanity)
        self.api_key = api_key
        self.temperature = temperature
        self.event_loop = asyncio.get_event_loop()
    
    def __call__(self, prompts, topic, num_samples=1, max_length=100):
        prompts = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts, topic)
        
        # define an async call to the API
        async def http_call(prompt_string):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"https://api.ai21.com/studio/v1/{self.model}/complete",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "prompt": prompt_string, 
                            "numResults": num_samples, 
                            "maxTokens": max_length,
                            "stopSequences": [self.quote+self.sep],
                            "topKReturn": 0,
                            "temperature": self.temperature
                        }) as resp:
                    result = await resp.json()
                    return [c["data"]["text"] for c in result["completions"]]
        
        # call the AI21 API asyncronously to complete the prompts
        results = self.event_loop.run_until_complete(asyncio.gather(*[http_call(s) for s in prompt_strings]))
        suggestion_texts = []
        for result in results:
            suggestion_texts.extend(result)
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)