import adatest

def test_simple_init():
    tree = adatest.TestTree()

def test_simple_init_with_file():
    tree = adatest.TestTree("temp_test_tree.csv")

def test_simple_init_with_list():
    tree = adatest.TestTree([
        {"topic": "", "type": "{} should output {}", "value1": "This is good", "value2": "NEGATIVE"}
    ])