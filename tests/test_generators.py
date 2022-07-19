import pandas as pd

from transformers import pipeline

from adatest import generators


class TestTransformers:
    def test_smoke(self):
        hf_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
        target = generators.Transformers(hf_model.model, hf_model.tokenizer)

        prompts = [
            ("id A", "", "Great hotel"),
            ("id B", "", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(prompts, "", num_samples=desired_result_count)
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)
