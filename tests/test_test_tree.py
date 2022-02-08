import adatest

def test_simple_init():
    tree = adatest.TestTree()

def test_simple_init_with_auto_save():
    tree = adatest.TestTree("temp_test_tree.csv", auto_save=True)

def test_simple_init_with_list():
    tree = adatest.TestTree([("This is good", "should be", "POSITIVE"), ("This is bad", "should be", "NEGATIVE")])