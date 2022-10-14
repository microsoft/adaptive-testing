import uuid

import adatest


def test_test_suggestions_smoke():
    test_tree = adatest.TestTree("imdb_hotel_conversion.csv")

    prompt_size = 5
    target = adatest._prompt_builder.PromptBuilder(prompt_size=prompt_size)

    n_reps = 8
    my_topic = "/Price"
    suggestions = target(
        test_tree, topic=my_topic, score_column="model score", repetitions=n_reps
    )
    assert suggestions is not None
    assert isinstance(suggestions, list)
    assert len(suggestions) == n_reps
    for s in suggestions:
        assert isinstance(s, list)
        assert len(s) == prompt_size
        for p in s:
            assert isinstance(p, tuple)
            # Each tuple should reference a row in the test_true
            tt_row = test_tree.loc[p[0]]
            assert tt_row["topic"] == p[1]
            assert tt_row["input"] == p[2]
