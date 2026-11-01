"""Tests for proofreader.py, using the FakeLLM from conftest.py."""

import asyncio

import pytest
from lxml import etree

from proofread.exceptions import LLMError
from proofread.proofreader import ParagraphJob, Proofreader, make_job, to_error_span
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
