"""Shared test helpers. pytest loads this file automatically."""

import asyncio
import json
import re

import pytest
from langchain_core.runnables import RunnableLambda

from proofread.schema import FoundError, ParagraphErrors, ProofreadResponse
from proofread.tokenizer import tokenize


class FakeLLM:
    """Stands in for a real structured-output LLM in tests.

    It reads the paragraphs out of the prompt, looks up prepared answers by paragraph
    text, and returns a ProofreadResponse, just like the real chain does.
    `answers` maps paragraph text -> list of (first, last, type, correction).
    """

    _PARAGRAPH_RE = re.compile(r'<paragraph id="(\d+)">\ntext: (".*")\n')

    def __init__(
        self,
        answers: dict[str, list[tuple]] | None = None,
        fail: bool = False,
        delay: float = 0.0,
        skip: set[str] | None = None,
        error: Exception | None = None,
    ):
        self.answers = answers or {}
        self.fail = fail
        self.skip = skip or set()  # paragraph texts to leave out of the answer
        self.delay = delay  # seconds each async call takes, to test parallel requests
        self.calls: list[list[str]] = []  # paragraph texts sent in each call
        self.in_flight = 0  # calls running right now
        self.max_in_flight = 0  # most calls that ever ran at the same time
        self.error = error  # what to raise when fail=True (default: ConnectionError)

    def invoke(self, messages) -> ProofreadResponse:
        found = [
            (int(pid), json.loads(text))
            for pid, text in self._PARAGRAPH_RE.findall(messages[-1].content)
        ]
        self.calls.append([text for _, text in found])
        if self.fail:
            raise self.error or ConnectionError("model unreachable")

        paragraphs = []
        for pid, text in found:
            if text in self.skip:
                continue  # simulate a model that forgets a paragraph
            tokens = tokenize([text])
            errors = [
                FoundError(
                    first_token=first,
                    last_token=last,
                    original="".join(t.text for t in tokens[first : last + 1]),
                    type=error_type,
                    correction=correction,
                )
                for first, last, error_type, correction in self.answers.get(text, [])
            ]
            paragraphs.append(ParagraphErrors(id=pid, errors=errors))
        return ProofreadResponse(paragraphs=paragraphs)

    async def ainvoke(self, messages, config=None) -> ProofreadResponse:
        """Async version, used by the Proofreader. Tracks how many calls overlap."""
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return self.invoke(messages)
        finally:
            self.in_flight -= 1

    def as_runnable(self):
        """Wrap as a LangChain Runnable, like the real chain."""
        return RunnableLambda(self.invoke)


@pytest.fixture
def fake_llm():
    """Factory fixture: fake_llm(answers, fail=False) -> FakeLLM."""
    return FakeLLM
