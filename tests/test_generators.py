import pandas as pd

from transformers import pipeline

from adatest import generators


class TestTransformers:
    def test_smoke(self):
        hf_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
        target = generators.Transformers(hf_model.model, hf_model.tokenizer)

        prompts = [
            ("id A", "Great hotel", "positive"),
            ("id B", "Bathroom too small", "negative"),
        ]

        results = target(prompts, "Rooms")
        assert results is not None
