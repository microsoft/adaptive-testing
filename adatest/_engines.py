import openai

class OpenAI():
    def __init__(self, engine="davinci-msft", max_tokens=50):
        self.engine = engine
        self.max_tokens = max_tokens

    def __call__(self, prompts, num_reps=1, temperature=0.9):
        response = openai.Completion.create(
            engine=self.engine, prompt=[p["prompt"] for p in prompts], max_tokens=self.max_tokens,
            temperature=temperature, n=num_reps, stop="\n"
        )
        return [choice["text"] for choice in response["choices"]]