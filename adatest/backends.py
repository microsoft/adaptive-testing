""" A set of backends for AdaTest.
"""
import asyncio
import aiohttp
import shap
import transformers
import openai

class Backend():
    """ Abstract class for language model backends.
    """
    
    def __init__(self, models, sep, subsep, quote):
        """ Create a new backend with the given separators and max length.
        """
        if not isinstance(models, list) and not isinstance(models, tuple):
            models = [models]
        self.model = models[0]
        self.models = models
        self.subsep = subsep
        self.sep = sep
        self.quote = quote

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
    
    def _create_prompt_strings(self, prompts):
        """ Convert prompts that are lists of tuples into strings for the LM to complete.
        """
        prompt_strings = []
        self._gen_value1s = []
        self._gen_comparators = []
        self._gen_value2s = []
        for prompt in prompts:
            value1s, comparators, value2s = zip(*prompt)
            self._gen_value1s.append(len(set(value1s)) > 1)
            self._gen_comparators.append(len(set(comparators)) > 1)
            self._gen_value2s.append(len(set(value2s)) > 1)
            prompt_string = "\""
            for value1, comparator, value2 in prompt:
                if self._gen_value1s[-1]:
                    prompt_string += value1 + self.quote
                if self._gen_value1s[-1] and (self._gen_comparators[-1] or self._gen_value2s[-1]):
                    prompt_string += self.subsep + self.quote
                if self._gen_comparators[-1]:
                    prompt_string += comparator + self.quote
                if self._gen_comparators[-1] and self._gen_value2s[-1]:
                    prompt_string += self.subsep + self.quote
                if self._gen_value2s[-1]:
                    prompt_string += value2 + self.quote
                prompt_string += self.sep + self.quote
            prompt_strings.append(prompt_string)
        return prompt_strings
    
    def _parse_suggestion_texts(self, suggestion_texts, prompts):
        """ Parse the suggestion texts into tuples.
        """
        assert len(suggestion_texts) % len(prompts) == 0, "Missing prompt completions!"
        
        num_samples = len(suggestion_texts) // len(prompts)
        samples = []
        for i, suggestion_text in enumerate(suggestion_texts):
            prompt_ind = i // num_samples
            prompt = prompts[prompt_ind]
            suggestion = suggestion_text.split(self.quote+self.subsep+self.quote)
            # if len(self.quote) > 0: # strip any dangling quote
            #     suggestion[-1] = suggestion[-1][:-len(self.quote)]
            if not self._gen_value1s[prompt_ind]:
                suggestion = [prompt[0][0]] + suggestion
            if not self._gen_comparators[prompt_ind]:
                suggestion = suggestion[:1] + [prompt[0][1]] + suggestion[1:]
            if not self._gen_value2s[prompt_ind]:
                suggestion = suggestion[:2] + [prompt[0][2]]

            if len(suggestion) == 3:
                samples.append(tuple(suggestion))
        return samples

           
class Transformers(Backend):
    def __init__(self, model, tokenizer, sep="\n", subsep=" ", quote="\""):
        super().__init__(model=model, sep=sep, subsep=subsep, quote=quote)
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
    
    def __call__(self, prompts, num_samples=1, max_length=100):
        
        prompts = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts)
        
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
    
    def __init__(self, models, api_key=None, sep="\n", subsep=" ", quote="\"", temperature=0.95):
        super().__init__(models, sep=sep, subsep=subsep, quote=quote)
        self.temperature = temperature
        if api_key is not None:
            openai.api_key = api_key
    
    def __call__(self, prompts, num_samples=1, max_length=100):
        prompts = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts)
        
        # call the OpenAI API to complete the prompts
        response = openai.Completion.create(
            engine=self.model, prompt=prompt_strings, max_tokens=max_length,
            temperature=self.temperature, n=num_samples, stop=self.quote+self.sep
        )
        suggestion_texts = [choice["text"] for choice in response["choices"]]
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)


class AI21(Backend):
    """ Backend wrapper for the OpenAI API that exposes GPT-3.
    """
    
    def __init__(self, model, api_key, sep="\n", subsep=" ", quote="\"", temperature=0.95):
        super().__init__(model, sep=sep, subsep=subsep, quote=quote)
        self.api_key = api_key
        self.temperature = temperature
        self.event_loop = asyncio.get_event_loop()
    
    def __call__(self, prompts, num_samples=1, max_length=100):
        prompts = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts)
        
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