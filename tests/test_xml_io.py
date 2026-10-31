"""Tests for xml_io.py."""

from pathlib import Path

import pytest

from proofread.exceptions import InputFileError
from proofread.xml_io import load_xml


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