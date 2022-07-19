import  adatest._test_tree_browser as ttb

class TestIsSubTopic:
    def test_topic_is_subtopic(self):
        assert ttb.is_subtopic('/A', '/A/B')
        assert ttb.is_subtopic('/A', '/A/B/C')
        assert ttb.is_subtopic('/A', '/A/B/C/')

    def test_topic_is_not_subtopic(self):
        assert not ttb.is_subtopic('/A/B', '/A/C')