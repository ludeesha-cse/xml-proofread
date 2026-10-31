"""Tests for xml_io.py."""

from pathlib import Path

import pytest

from proofread.exceptions import InputFileError
from proofread.xml_io import load_xml

from proofread.exceptions import InputFileError, OutputFileError
from proofread.xml_io import load_xml, save_xml

def write(tmp_path: Path, content: bytes, name: str = "in.xml") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_load_reads_details(tmp_path):
    path = write(tmp_path, b"<?xml version='1.0' encoding='utf-8'?>\r\n<r><p>Hi</p></r>\r\n")
    doc = load_xml(path)
    assert doc.root.tag == "r"
    assert doc.encoding == "utf-8"
    assert doc.header == b"<?xml version='1.0' encoding='utf-8'?>\r\n"
    assert doc.footer == b"\r\n"
    assert doc.newline == b"\r\n"


def test_load_without_declaration(tmp_path):
    doc = load_xml(write(tmp_path, b"<r/>"))
    assert doc.header == b""
    assert doc.footer == b""
    assert doc.newline == b"\n"


@pytest.mark.parametrize("content", [b"", b"<r>", b"<r></x>", b"not xml at all"])
def test_load_rejects_malformed_xml(tmp_path, content):
    with pytest.raises(InputFileError, match="not well-formed"):
        load_xml(write(tmp_path, content))


def test_load_missing_file(tmp_path):
    with pytest.raises(InputFileError, match="Cannot read"):
        load_xml(tmp_path / "missing.xml")

# --- round trip ---------------------------------------------------------------

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            b"<?xml version='1.0' encoding='utf-8'?>\r\n<r>\r\n  <p>a</p>\r\n</r>", id="crlf"
        ),
        pytest.param(
            b'<?xml version="1.0" encoding="UTF-8"?>\n<r><p>a</p></r>\n', id="final-newline"
        ),
        pytest.param(b"<r><p>a</p></r>", id="no-declaration"),
        pytest.param(
            b'<r xmlns="http://x" xmlns:h="http://h"><h:p>x</h:p><p>y</p></r>', id="namespaces"
        ),
        pytest.param(
            b"<r><p>x <![CDATA[a < b]]> y</p><!-- note --><?pi z?></r>", id="cdata-comment-pi"
        ),
        pytest.param(b"<r><p>Tom &amp; Jerry &lt;3</p></r>", id="escaped-chars"),
        pytest.param(
            b'<r><p id="1" class="c">x <b>bold</b> and <i>it</i>.</p></r>', id="mixed-content"
        ),
        pytest.param("<r><p>Café naïve 日本語</p></r>".encode(), id="unicode"),
    ],
)
def test_round_trip_is_byte_identical(tmp_path, content):
    src = write(tmp_path, content)
    out = tmp_path / "out.xml"
    save_xml(load_xml(src), out)
    assert out.read_bytes() == content


def test_round_trip_assignment_sample(tmp_path):
    out = tmp_path / "out.xml"
    save_xml(load_xml(SAMPLES / "input.xml"), out)
    assert out.read_bytes() == (SAMPLES / "input.xml").read_bytes()


def test_save_to_missing_folder_fails(tmp_path):
    doc = load_xml(write(tmp_path, b"<r/>"))
    with pytest.raises(OutputFileError):
        save_xml(doc, tmp_path / "no-such-dir" / "out.xml")


def test_save_leaves_no_temp_files(tmp_path):
    src = write(tmp_path, b"<r/>")
    save_xml(load_xml(src), tmp_path / "out.xml")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.xml", "out.xml"]