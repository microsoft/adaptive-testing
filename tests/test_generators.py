import pandas as pd
import pytest

from transformers import pipeline

from adaptivetesting import generators

TRANSFORMER_PIPELINE_MODELS = ["EleutherAI/gpt-neo-125M"]


class TestTransformers:
    @pytest.mark.parametrize("model_name", TRANSFORMER_PIPELINE_MODELS)
    def test_smoke(self, model_name):
        hf_model = pipeline("text-generation", model=model_name)
        target = generators.Transformers(hf_model.model, hf_model.tokenizer)

        prompts = [
            ("id A", "", "Great hotel"),
            ("id B", "", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(
            prompts=prompts,
            topic="",
            mode="tests",
            topic_description="",
            num_samples=desired_result_count,
        )
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)

    @pytest.mark.parametrize("model_name", TRANSFORMER_PIPELINE_MODELS)
    def test_with_topics(self, model_name):
        hf_model = pipeline("text-generation", model=model_name)
        target = generators.Transformers(hf_model.model, hf_model.tokenizer)

        prompts = [
            ("id A", "some string", "Great hotel"),
            ("id B", "some_string", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(
            prompts=prompts,
            topic="",
            mode="tests",
            topic_description="",
            num_samples=desired_result_count,
        )
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)


GENERATOR_PIPELINE_MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "EleutherAI/gpt-neo-125M",
    "gpt2",
    "bigscience/bloom-560m",
]


class TestPipelines:
    @pytest.mark.parametrize(
        "model_name",
        GENERATOR_PIPELINE_MODELS,
    )
    def test_smoke(self, model_name):
        hf_pipeline = pipeline("text-generation", model=model_name)
        target = generators.Pipelines(hf_pipeline)

        prompts = [
            ("id A", "", "Great hotel"),
            ("id B", "", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(
            prompts=prompts,
            topic="some topic",
            mode="tests",
            topic_description="",
            num_samples=desired_result_count,
        )
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)

    @pytest.mark.parametrize(
        "model_name",
        GENERATOR_PIPELINE_MODELS,
    )
    def test_with_topics(self, model_name):
        hf_pipeline = pipeline("text-generation", model=model_name)
        target = generators.Pipelines(hf_pipeline)

        prompts = [
            ("id A", "some_string", "Great hotel"),
            ("id B", "some_string", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(
            prompts=prompts,
            topic="some topic",
            mode="tests",
            topic_description="",
            num_samples=desired_result_count,
        )
        assert results is not None
        assert len(results) == desired_result_count
        for item in results:
            assert isinstance(item, str)


class TestOpenAI:
    def test_smoke(self, mocker):
        OPENAI_API_KEY = "Not for you, CredScan"

        openai_completion = mocker.patch("openai.Completion", autospec=True)
        patched_response = {"choices": [{"text": "Ret 1"}, {"text": "Ret 2"}]}
        openai_completion.create.return_value = patched_response

        target = generators.OpenAI("curie", api_key=OPENAI_API_KEY)

        prompts = [
            ("id A", "", "Great hotel"),
            ("id B", "", "Bathroom too small"),
        ]

        desired_result_count = 2

        results = target(
            prompts=prompts,
            topic="",
            topic_description="",
            mode="tests",
            num_samples=desired_result_count,
            scorer=None,
        )
        assert results is not None
        assert len(results) == desired_result_count
        assert "Ret 1" in results
        assert "Ret 2" in results
        openai_completion.create.assert_called_with(
            model="curie",
            prompt=['"Great hotel"\n"Bathroom too small"\n"'],
            user="adaptivetesting",
            max_tokens=100,
            temperature=1.0,
            top_p=0.95,
            n=2,
            stop='"',
        )
