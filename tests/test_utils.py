import numpy as np
import pytest

import adatest.utils as utils


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


class TestConvertFloat:
    @pytest.mark.parametrize(
        "v", [0.1, 0, 1, -1, 1.1e18, -6.123e25, 5.7e-14, -8.834e-18, 6, 6.0, 6, 900]
    )
    def test_conversion_expected(self, v):
        v_str = str(v)

        assert v == utils.convert_float(v_str)

    def test_empty_string(self):
        assert np.isnan(utils.convert_float(""))

    def test_nonnumeric_string(self):
        assert np.isnan(utils.convert_float("one"))
