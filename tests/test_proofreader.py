"""Tests for proofreader.py, using the FakeLLM from conftest.py."""

import asyncio

import pytest
from langchain_core.exceptions import ModelRateLimitError
from lxml import etree

from proofread.exceptions import LLMError
from proofread.proofreader import ParagraphJob, Proofreader, make_job, short_error, to_error_span
from proofread.schema import FoundError
from proofread.tokenizer import tokenize


def job(text: str, index: int = 0) -> ParagraphJob:
    return ParagraphJob(index=index, text=text, tokens=tokenize([text]))


def found(first, last, original, correction="X", type="grammar") -> FoundError:
    return FoundError(
        first_token=first, last_token=last, original=original, type=type, correction=correction
    )


def proofread(llm, jobs, batch_size=10, concurrency=4, strict=False):
    """Run Proofreader.proofread (async) and return its output."""
    proofreader = Proofreader(
        llm, language="English", lang="en", batch_size=batch_size,
        concurrency=concurrency, strict=strict,
    )  # fmt: skip
    return asyncio.run(proofreader.proofread(jobs))


# --- make_job -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        ("<p>Hello world.</p>", "Hello world."),
        ("<p>Hello <b>big</b> world.</p>", "Hello big world."),
        ("<p/>", None),
        ("<p>   </p>", None),
        ("<p>42 - 7</p>", None),  # no letters
        ("<p><![CDATA[teh text]]></p>", None),
    ],
)
def test_make_job(xml, expected):
    p = etree.fromstring(xml, etree.XMLParser(strip_cdata=False))
    result = make_job(0, p)
    assert (result.text if result else None) == expected


# --- to_error_span --------------------------------------------------------------------


def test_to_error_span_converts_tokens_to_characters():
    span = to_error_span(job("My name is john."), found(6, 6, "john", "John"))
    assert (span.start, span.end, span.correction) == (11, 15, "John")


def test_to_error_span_repairs_an_off_by_one_token_number():
    # Token 5 is " ", but the model meant "john" (token 6).
    span = to_error_span(job("My name is john."), found(5, 5, "john", "John"))
    assert (span.start, span.end) == (11, 15)


def test_to_error_span_rejects_ambiguous_original():
    # "the" appears twice, so a wrong token number can't be repaired safely.
    with pytest.raises(ValueError, match="unambiguously"):
        to_error_span(job("the cat and the dog"), found(1, 1, "the"))


def test_to_error_span_rejects_text_that_is_not_there():
    with pytest.raises(ValueError):
        to_error_span(job("a b c"), found(0, 0, "zebra"))


def test_to_error_span_rejects_out_of_range_tokens():
    with pytest.raises(ValueError):
        to_error_span(job("a b"), found(7, 9, "a b c"))


@pytest.mark.parametrize(
    ("text", "first", "last", "original", "correction", "fixed"),
    [
        # comma put on the next word: "galleries , and" -> "galleries, and"
        ("the galleries and the food", 4, 4, "and", ", and", "the galleries, and the food"),
        # space selected, but the correction keeps a space before the comma
        ("work hard but sad", 3, 3, " ", " , ", "work hard, but sad"),
        # combined fix put on the second word: "Paris , France" -> "Paris, France"
        ("Paris france.", 2, 2, "france", ", France", "Paris, France."),
        # correct selections are left alone
        ("eggs toast", 1, 1, " ", ", ", "eggs, toast"),
        ("ok?", 1, 1, "?", ".", "ok."),
        ("However it", 0, 0, "However", "However,", "However, it"),
    ],
)
def test_no_space_is_left_before_a_comma_or_period(text, first, last, original, correction, fixed):
    span = to_error_span(job(text), found(first, last, original, correction))
    assert text[: span.start] + span.correction + text[span.end :] == fixed


# --- Proofreader ----------------------------------------------------------------------


def test_finds_errors_and_maps_them_to_paragraph_indexes(fake_llm):
    llm = fake_llm({"My name is john.": [(6, 6, "capitalization", "John")]})
    jobs = [job("Fine text.", index=0), job("My name is john.", index=3)]
    output = proofread(llm, jobs)
    assert list(output.spans) == [3]
    assert output.spans[3][0].correction == "John"
    assert output.failed == [] and output.dropped == 0


def test_paragraphs_are_sent_in_batches(fake_llm):
    llm = fake_llm()
    jobs = [job(f"Text {i}.", index=i) for i in range(5)]
    proofread(llm, jobs, batch_size=2)
    assert [len(call) for call in llm.calls] == [2, 2, 1]


def test_failed_request_leaves_paragraphs_unchanged(fake_llm):
    output = proofread(fake_llm(fail=True), [job("a b", index=4)])
    assert output.failed == [4]
    assert output.spans == {}


def test_failed_request_in_strict_mode_raises(fake_llm):
    with pytest.raises(LLMError):
        proofread(fake_llm(fail=True), [job("a b")], strict=True)


def test_unusable_suggestions_are_counted(fake_llm):
    llm = fake_llm({"a b": [(0, 99, "grammar", "X")]})  # token 99 doesn't exist
    output = proofread(llm, [job("a b")])
    assert output.dropped == 1
    assert output.spans == {}


def test_works_with_a_langchain_runnable(fake_llm):
    llm = fake_llm({"i agree.": [(0, 0, "capitalization", "I")]})
    output = proofread(llm.as_runnable(), [job("i agree.")])
    assert output.spans[0][0].correction == "I"


def test_skipped_paragraph_is_marked_failed(fake_llm):
    llm = fake_llm({"i agree.": [(0, 0, "capitalization", "I")]}, skip={"Other."})
    output = proofread(llm, [job("i agree.", index=0), job("Other.", index=1)])
    assert output.failed == [1]  # the skipped one
    assert output.spans[0][0].correction == "I"  # the returned one is still used


def test_skipped_paragraph_in_strict_mode_raises(fake_llm):
    llm = fake_llm(skip={"Other."})
    with pytest.raises(LLMError, match="skipped paragraphs"):
        proofread(llm, [job("Fine.", index=0), job("Other.", index=1)], strict=True)


def test_requests_run_in_parallel_up_to_the_limit(fake_llm):
    llm = fake_llm(delay=0.05)
    jobs = [job(f"Text {i}.", index=i) for i in range(10)]
    output = proofread(llm, jobs, batch_size=1, concurrency=3)
    assert output.requests == 10
    assert llm.max_in_flight == 3  # parallel, but never more than --concurrency


def test_concurrency_one_runs_requests_one_at_a_time(fake_llm):
    llm = fake_llm(delay=0.01)
    proofread(llm, [job(f"T{i}.", index=i) for i in range(4)], batch_size=1, concurrency=1)
    assert llm.max_in_flight == 1


def test_identical_paragraphs_are_sent_once(fake_llm):
    llm = fake_llm({"i agree.": [(0, 0, "capitalization", "I")]})
    jobs = [job("i agree.", index=0), job("Other.", index=1), job("i agree.", index=2)]
    output = proofread(llm, jobs)
    assert llm.calls == [["i agree.", "Other."]]  # the duplicate was not sent
    assert output.duplicates == 1
    assert output.spans[0] == output.spans[2]  # but both got the answer


def test_failure_is_copied_to_duplicates(fake_llm):
    output = proofread(fake_llm(fail=True), [job("a b", index=0), job("a b", index=5)])
    assert sorted(output.failed) == [0, 5]


def test_short_error_keeps_logs_readable():
    huge = ModelRateLimitError("429 RESOURCE_EXHAUSTED. " + "{'details': 'x'} " * 200)
    text = short_error(huge)
    assert text.startswith("ModelRateLimitError: 429 RESOURCE_EXHAUSTED.")
    assert len(text) <= 160 and text.endswith("…")
    assert short_error(ValueError("short")) == "ValueError: short"


def test_rate_limited_requests_are_counted(fake_llm):
    llm = fake_llm(fail=True, error=ModelRateLimitError("429"))
    output = proofread(llm, [job("a b", index=0), job("c d", index=1)], batch_size=1)
    assert output.rate_limited == 2
    assert output.first_error == "ModelRateLimitError: 429"
