import pytest

from agenticlab.core.llm.utils import parse_json_response


def test_parse_json_plain():
    result = parse_json_response('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_markdown_block():
    result = parse_json_response('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_parse_json_embedded_in_prose():
    result = parse_json_response('Sure! Here is the result: {"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_raises_on_invalid():
    with pytest.raises(ValueError, match="No JSON"):
        parse_json_response("this is not json at all")
