import adatest

def test_test_suggestions_smoke():
    test_tree = adatest.TestTree("imdb_hotel_conversion.csv")

    prompt_size=5
    target = adatest._prompt_builder.PromptBuilder(prompt_size=prompt_size)

    n_reps = 8
    suggestions = target(test_tree, topic='/Price', score_column='model score', repetitions=n_reps)
    assert suggestions is not None
    assert isinstance(suggestions, list)
    assert len(suggestions) == n_reps
    for s in suggestions:
        assert isinstance(s, list)
        assert len(s) == prompt_size