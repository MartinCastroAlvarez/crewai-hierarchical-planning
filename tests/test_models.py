from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from hierarchical_crew.models import LaunchBrief
from hierarchical_crew.models import extract_json_object
from hierarchical_crew.models import parse_brief
from tests.conftest import brief_dict


def test_extract_json_object_tolerates_wrapping() -> None:
    raw = json.dumps(brief_dict())
    assert extract_json_object(f"<think>{{'x'}}</think>Here you go:\n```json\n{raw}\n```")
    assert extract_json_object(f"Sure! {raw} Hope that helps.") == brief_dict()
    assert extract_json_object('``` not json ``` {"a": 1}') == {"a": 1}
    assert extract_json_object("no json at all {oops") is None
    assert extract_json_object("[1, 2]") is None


def test_parse_brief_from_many_shapes() -> None:
    model = LaunchBrief.model_validate(brief_dict())
    assert parse_brief(model) is model
    assert parse_brief(brief_dict()).product == "Aurora"
    assert parse_brief("Final: " + json.dumps(brief_dict())).price_usd == 131.99

    class Other(BaseModel):
        product: str = "Tidal"

    assert parse_brief(Other()).product == "Tidal"
    with pytest.raises(ValueError, match="no JSON"):
        parse_brief("nothing here")
    with pytest.raises(ValidationError):
        parse_brief(brief_dict(price_usd=-1))
    with pytest.raises(ValidationError):
        parse_brief(brief_dict(key_messages=["m"] * 6))
