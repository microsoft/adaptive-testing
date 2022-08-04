import os
import tempfile

import numpy as np

import adatest


def test_simple_init():
    tree = adatest.TestTree()
    assert len(tree) == 0


def test_simple_init_with_file():
    tree = adatest.TestTree("temp_test_tree.csv")
    assert len(tree) == 0


def test_simple_init_with_list():
    tree = adatest.TestTree(
        ["The food was nice!", "The location is excellent."]
    )
    assert len(tree) == 3
    assert tree.columns.to_list() == ['topic', 'input', 'output', 'label', 'labeler', 'description']
    assert "The food was nice!" in tree['input'].to_list()
    assert "The location is excellent." in tree['input'].to_list()
    outputs = np.unique(tree['output'].to_list(), return_counts=True)
    assert outputs[0][0] == ''
    assert outputs[1][0] == 1
    assert outputs[0][1] == '__TOOVERWRITE__'
    assert outputs[1][1] ==2



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
