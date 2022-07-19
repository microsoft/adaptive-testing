import os
import tempfile

import adatest


def test_simple_init():
    tree = adatest.TestTree()
    assert len(tree) == 0


def test_simple_init_with_file():
    tree = adatest.TestTree("temp_test_tree.csv")
    assert len(tree) == 0


def test_simple_init_with_list():
    tree = adatest.TestTree(
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
    assert len(tree) == 2
    assert tree["topic"][0] == ""
    assert tree["topic"][1] == ""
    if tree['type'][0] is None:
        assert tree["type"][1] == "{} should output {}"
        assert tree["input"][0] == ""
        assert tree["input"][1] == "This is good"
        assert tree["output"][0] == ""
        assert tree["output"][1] == "NEGATIVE"
        assert tree["label"][0] == "topic_marker"
        assert tree["label"][1] == "fail"
    else:
        assert tree["type"][0] == "{} should output {}"
        assert tree["type"][1] is None
        assert tree["input"][1] == ""
        assert tree["input"][0] == "This is good"
        assert tree["output"][1] == ""
        assert tree["output"][0] == "NEGATIVE"
        assert tree["label"][1] == "topic_marker"
        assert tree["label"][0] == "fail"


def test_to_csv():
    tree = adatest.TestTree(
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
        target_file = os.path.join(td, "adatest_out.csv")
        tree.to_csv(target_file)
        assert os.path.exists(target_file)
