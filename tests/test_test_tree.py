import adatest

def test_simple_init():
    tree = adatest.TestTree()

def test_simple_init_with_file():
    tree = adatest.TestTree("temp_test_tree.csv")

def test_simple_init_with_list():
    tree = adatest.TestTree([
        {"topic": "", "type": "{} should output {}", "input": "This is good", "output": "NEGATIVE", "label": "fail"}
    ])