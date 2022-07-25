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

        results = target(prompts, "", mode="tests", num_samples=desired_result_count)
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)


class TestOpenAI:
    def test_smoke(self, mocker):
        OPENAI_API_KEY = "Not for you, CredScan"

        openai_completion = mocker.patch("openai.Completion", auto_spec=True)
        patched_response = {"choices": [{"text": "Ret 1"}, {"text": "Ret 2"}]}
        openai_completion.create.return_value = patched_response

        target = generators.OpenAI("curie", api_key=OPENAI_API_KEY)

        prompts = [
            ("id A", "", "Great hotel"),
            ("id B", "", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(prompts, "", mode="tests", num_samples=desired_result_count)
        assert results is not None
        assert len(results) == desired_result_count
        assert "Ret 1" in results
        assert "Ret 2" in results
        openai_completion.create.assert_called_with(
            engine="curie",
            prompt=['"Great hotel"\n"Bathroom too small"\n"'],
            max_tokens=100,
            temperature=1.0,
            top_p=0.95,
            n=2,
            stop='"',
        )
