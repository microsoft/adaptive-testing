import pytest

import adaptivetesting.utils as utils


class TestIsSubTopic:
    @pytest.mark.parametrize(
        ["topic", "sub_topic"],
        [
            ("/A", "/A/B"),
            ("/A", "/A/B/C"),
            ("/A", "/A/B/C/"),
            ("/A ", "/A /B"),
            ("/A ", "/A /B "),
            ("/A ", "/A /B /C"),
        ],
    )
    def test_topic_is_subtopic(self, topic, sub_topic):
        assert utils.is_subtopic(topic, sub_topic)

    @pytest.mark.parametrize(
        ["topic", "not_sub_topic"], [("/A/B", "/A/C"), ("/A", "/AB")]
    )
    def test_topic_is_not_subtopic(self, topic, not_sub_topic):
        assert not utils.is_subtopic(topic, not_sub_topic)

    @pytest.mark.parametrize(
        ["topic"],
        # Extra ',' in tuples to resolve ambiguity between string and list of characters
        [("/A",), ("/A/B",), ("/A /B",), ("/A /B ",), ("/A/B/C",), ("/A /B /C",)],
    )
    def test_topic_own_subtopic(self, topic):
        assert utils.is_subtopic(topic, topic)
