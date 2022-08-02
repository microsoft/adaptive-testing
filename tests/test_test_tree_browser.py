import pytest

import adatest._test_tree_browser as ttb


class TestIsSubTopic:
    @pytest.mark.parametrize(
        ["topic", "sub_topic"], [("/A", "/A/B"), ("/A", "/A/B/C"), ("/A", "/A/B/C/")]
    )
    def test_topic_is_subtopic(self, topic, sub_topic):
        assert ttb.is_subtopic(topic, sub_topic)

    @pytest.mark.parametrize(
        ["topic", "not_sub_topic"], [("/A/B", "/A/C"), ("/A", "/AB")]
    )
    def test_topic_is_not_subtopic(self, topic, not_sub_topic):
        assert not ttb.is_subtopic(topic, not_sub_topic)
