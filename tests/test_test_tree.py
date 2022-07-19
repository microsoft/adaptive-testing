import adatest

def test_simple_init():
    tree = adatest.TestTree()
    assert len(tree) == 0

def test_simple_init_with_file():
    tree = adatest.TestTree("temp_test_tree.csv")
    assert len(tree) == 0

def test_simple_init_with_list():
    tree = adatest.TestTree([
        {"topic": "", "type": "{} should output {}", "input": "This is good", "output": "NEGATIVE", "label": "fail"}
    ])
    assert len(tree) == 2