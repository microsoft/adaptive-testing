from operator import truediv
import os
import pathlib
import tempfile

import numpy as np

import adaptivetesting


def test_simple_init():
    tree = adaptivetesting.TestTree()
    assert len(tree) == 0


def test_simple_init_with_file():
    tree = adaptivetesting.TestTree("temp_test_tree.csv")
    assert len(tree) == 0


def test_simple_init_with_list():
    tree = adaptivetesting.TestTree(["The food was nice!", "The location is excellent."])
    assert len(tree) == 3
    assert tree.columns.to_list() == [
        "topic",
        "input",
        "output",
        "label",
        "labeler",
        "description",
    ]
    assert "The food was nice!" in tree["input"].to_list()
    assert "The location is excellent." in tree["input"].to_list()
    outputs = np.unique(tree["output"].to_list(), return_counts=True)
    assert outputs[0][0] == ""
    assert outputs[1][0] == 1
    assert outputs[0][1] == "[no output]"
    assert outputs[1][1] == 2


def test_to_csv():
    tree = adaptivetesting.TestTree(
        [
            {
                "topic": "",
                "type": "{} should output {}",
                "input": "This is good",
                "output": "NEGATIVE",
                "label": "fail",
            }
        ]
    )
    with tempfile.TemporaryDirectory() as td:
        target_file = os.path.join(td, "adaptivetesting_out.csv")
        tree.to_csv(target_file)
        assert os.path.exists(target_file)


def test_has_subtopic_or_tests():
    curr_file = pathlib.Path(__file__)
    curr_dir = curr_file.parent
    input_csv = curr_dir / "simple_test_tree.csv"
    assert input_csv.exists()
    tree = adaptivetesting.TestTree(str(input_csv))
    # The top level topic appears to be an empty string, which is odd
    assert tree.topic_has_subtopics("") == True
    assert tree.topic_has_direct_tests("") == True
    assert tree.topic_has_direct_tests("/A") == True
    assert tree.topic_has_subtopics("/A") == True
    assert tree.topic_has_direct_tests("/A/B") == True
    assert tree.topic_has_subtopics("/A/B") == False
    assert tree.topic_has_direct_tests("/A/C") == False
    assert tree.topic_has_subtopics("/A/C") == False
