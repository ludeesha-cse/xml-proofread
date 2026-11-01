"""Tests for schema.py: the shape the LLM must return."""

import pytest
from pydantic import ValidationError

from proofread.models import ERROR_TYPES
from proofread.schema import ProofreadResponse


def test_valid_response_parses():
    data = {
        "paragraphs": [
            {
                "id": 0,
                "errors": [
                    {
                        "first_token": 2,
                        "last_token": 2,
                        "original": "companys",
                        "type": "punctuation",
                        "correction": "company's",
                    }
                ],
            },
            {"id": 1, "errors": []},
        ]
    }
    response = ProofreadResponse.model_validate(data)
    assert response.paragraphs[0].errors[0].correction == "company's"


def test_missing_errors_list_defaults_to_empty():
    response = ProofreadResponse.model_validate({"paragraphs": [{"id": 0}]})
    assert response.paragraphs[0].errors == []


def test_unknown_error_type_is_rejected():
    bad = {"first_token": 0, "last_token": 0, "original": "a", "type": "style", "correction": "A"}
    with pytest.raises(ValidationError):
        ProofreadResponse.model_validate({"paragraphs": [{"id": 0, "errors": [bad]}]})


def test_schema_lists_all_error_types():
    schema = ProofreadResponse.model_json_schema()
    enum = schema["$defs"]["FoundError"]["properties"]["type"]["enum"]
    assert tuple(enum) == ERROR_TYPES
