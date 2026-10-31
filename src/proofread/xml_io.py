"""Reading and writing XML files without changing anything we did not touch.

lxml parses the document into a tree we can edit. A few details of the original file
are not kept by lxml, so we save them ourselves and put them back when writing:

* the XML declaration exactly as written (quotes, case, no added standalone='no')
* the line-ending style (lxml turns Windows "\\r\\n" into "\\n")
* whitespace after the last tag (e.g. a final newline)
"""

import io
import logging
import re
from dataclasses import dataclass

from lxml import etree

from proofread.exceptions import InputFileError, OutputFileError
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional UTF-8 byte-order mark, the <?xml ...?> declaration, and the whitespace after it.
_DECLARATION_RE = re.compile(rb"^(?:\xef\xbb\xbf)?<\?xml[^>]*\?>\s*")
# Whitespace at the very end of the file.
_TRAILING_WS_RE = re.compile(rb"\s*$")


@dataclass
class XmlDocument:
    """A parsed XML file plus the details lxml would otherwise lose."""

    tree: etree._ElementTree
    encoding: str  # e.g. "utf-8"
    header: bytes  # raw declaration + whitespace after it
    footer: bytes  # raw whitespace after the last tag
    newline: bytes  # b"\r\n" for Windows-style files, otherwise b"\n"

    @property
    def root(self) -> etree._Element:
        return self.tree.getroot()

def _make_parser() -> etree.XMLParser:
    """An lxml parser that keeps everything and is safe against hostile files."""
    return etree.XMLParser(
        strip_cdata=False, 
        remove_blank_text=False,  # keep whitespace between tags (indentation)
        remove_comments=False,
        remove_pis=False,  # keep processing instructions like <?pi ...?>
        resolve_entities=False,  # don't expand entities (blocks XXE attacks, keeps &custom;)
        no_network=True, 
    )

def load_xml(path: Path) -> XmlDocument:
    """Read and parse an XML file. Raises InputFileError with a clear message."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputFileError(f"Cannot read {path}: {exc.strerror or exc}") from exc

    try:
        tree = etree.parse(io.BytesIO(raw), _make_parser())
    except etree.XMLSyntaxError as exc:
        raise InputFileError(f"{path} is not well-formed XML: {exc}") from exc

    declaration = _DECLARATION_RE.match(raw)
    doc = XmlDocument(
        tree=tree,
        encoding=tree.docinfo.encoding or "utf-8",
        header=declaration.group(0) if declaration else b"",
        footer=_TRAILING_WS_RE.search(raw).group(0),
        newline=b"\r\n" if b"\r\n" in raw else b"\n",
    )
    logger.debug(
        "Loaded %s (%d bytes, encoding=%s, newline=%r)",
        path,
        len(raw),
        doc.encoding,
        doc.newline,
    )
    return doc
