"""Unusual but realistic inputs, run through the whole pipeline with a fake model."""

from pathlib import Path

import pytest

from proofread.config import build_settings
from proofread.paragraph import paragraph_text
from proofread.pipeline import run
from proofread.xml_io import find_paragraphs, load_xml


def proofread_bytes(tmp_path: Path, data: bytes, llm, **overrides):
    """Write `data` to a file, run the pipeline on it, return (result, output bytes)."""
    src = tmp_path / "in.xml"
    src.write_bytes(data)
    values = dict(
        input_path=src, lang="en", model="ollama:fake", output_path=None, batch_size=10,
        concurrency=4, max_retries=0, timeout_s=10.0, strict=False,
    )  # fmt: skip
    values.update(overrides)
    result = run(build_settings(**values), llm=llm)
    return result, result.output_path.read_bytes()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n<r><p>Café crème.</p></r>\n'.encode(
                "latin-1"
            ),
            id="latin-1",
        ),
        pytest.param(
            '\ufeff<?xml version="1.0" encoding="UTF-8"?>\n<r><p>Hello.</p></r>\n'.encode(),
            id="utf-8-with-bom",
        ),
        pytest.param(b"<r><title>No paragraphs</title></r>", id="no-paragraphs"),
    ],
)
def test_unchanged_files_round_trip_exactly(tmp_path, fake_llm, data):
    _, out = proofread_bytes(tmp_path, data, fake_llm({}))
    assert out == data


def test_latin1_file_gets_errors_in_its_own_encoding(tmp_path, fake_llm):
    data = '<?xml version="1.0" encoding="ISO-8859-1"?>\n<r><p>Café teh crème.</p></r>'
    llm = fake_llm({"Café teh crème.": [(2, 2, "spelling", "the")]})
    _, out = proofread_bytes(tmp_path, data.encode("latin-1"), llm)
    assert out.decode("latin-1").endswith(
        '<p>Café <error type="spelling" correction="the">teh</error> crème.</p></r>'
    )


def test_entity_references_survive(tmp_path, fake_llm):
    data = b'<?xml version="1.0"?>\n<!DOCTYPE r [<!ENTITY co "Acme">]>\n<r><p>&co; is grate.</p></r>'
    llm = fake_llm({" is grate.": [(3, 3, "spelling", "great")]})
    _, out = proofread_bytes(tmp_path, data, llm)
    assert b'<p>&co; is <error type="spelling" correction="great">grate</error>.</p>' in out


def test_utf16_file_keeps_its_text(tmp_path, fake_llm):
    # Known limitation: the <?xml?> declaration is not kept for UTF-16 files.
    # The output is still valid UTF-16 XML (its byte-order mark names the encoding).
    data = '<?xml version="1.0" encoding="UTF-16"?>\n<r><p>Hello wrld.</p></r>'.encode("utf-16")
    llm = fake_llm({"Hello wrld.": [(2, 2, "spelling", "world")]})
    result, _ = proofread_bytes(tmp_path, data, llm)
    paragraph = find_paragraphs(load_xml(result.output_path).root)[0]
    assert paragraph_text(paragraph) == "Hello wrld."
    assert result.errors_applied == 1


def test_large_file_in_parallel(tmp_path, fake_llm):
    count = 500
    body = "".join(f"<p>Paragraph {i} has teh error.</p>" for i in range(count))
    answers = {f"Paragraph {i} has teh error.": [(4, 4, "spelling", "the")] for i in range(count)}
    llm = fake_llm(answers, delay=0.02)
    result, out = proofread_bytes(
        tmp_path, f"<r>{body}</r>".encode(), llm, batch_size=10, concurrency=8
    )
    assert (result.errors_applied, result.requests, result.failed) == (count, 50, 0)
    assert llm.max_in_flight == 8
    assert out.count(b"<error ") == count


def test_repeated_paragraphs_are_sent_once(tmp_path, fake_llm):
    body = "<p>Same teh text.</p>" * 20 + "<p>Other text.</p>"
    llm = fake_llm({"Same teh text.": [(2, 2, "spelling", "the")]})
    result, out = proofread_bytes(tmp_path, f"<r>{body}</r>".encode(), llm)
    assert (result.duplicates, result.errors_applied) == (19, 20)
    assert sum(len(call) for call in llm.calls) == 2  # only two paragraphs were sent
