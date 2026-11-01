"""Score a model's output against a reference (expected) output.

Two outputs can mark the same fix differently: "paris france" as one <error>, or
"paris" and "france" separately. Comparing tags would call one of them wrong. So we
compare what matters: the corrected text you get by applying every correction.

Each paragraph's corrected text is compared with its original text, token by token,
to get a set of edits. An edit the model made that the reference also made is a hit.
  precision = hits / edits the model made       (were its changes right?)
  recall    = hits / edits the reference made   (did it find the real errors?)
"""

import difflib
from dataclasses import dataclass

from lxml import etree

from proofread.paragraph import is_paragraph, paragraph_text
from proofread.tokenizer import tokenize

Edit = tuple[int, int, str]  # (first original token, end token, replacement text)


def corrected_text(p: etree._Element) -> str:
    """The paragraph's text with every <error> replaced by its correction."""
    parts: list[str] = []

    def walk(el: etree._Element) -> None:
        parts.append(el.text or "")
        for child in el:
            if isinstance(child.tag, str) and etree.QName(child).localname == "error":
                parts.append(child.get("correction", ""))
            elif isinstance(child.tag, str) and not is_paragraph(child):
                walk(child)
            parts.append(child.tail or "")

    walk(p)
    return "".join(parts)


def edits(original: str, corrected: str) -> set[Edit]:
    """The changes that turn `original` into `corrected`, at token level."""
    before = [t.text for t in tokenize([original])]
    after = [t.text for t in tokenize([corrected])]
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return {
        (i1, i2, "".join(after[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    }


@dataclass
class Score:
    hits: int = 0  # edits both made
    extra: int = 0  # edits only the model made (false positives)
    missed: int = 0  # edits only the reference made (false negatives)
    exact: int = 0  # paragraphs whose corrected text matches the reference exactly
    paragraphs: int = 0

    @property
    def precision(self) -> float:
        made = self.hits + self.extra
        return self.hits / made if made else 1.0

    @property
    def recall(self) -> float:
        needed = self.hits + self.missed
        return self.hits / needed if needed else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(reference: etree._Element, output: etree._Element) -> Score:
    """Compare two annotated versions of the same document, paragraph by paragraph."""
    ref_paragraphs = list(reference.iter("{*}p"))
    out_paragraphs = list(output.iter("{*}p"))
    if len(ref_paragraphs) != len(out_paragraphs):
        raise ValueError("The two documents have a different number of <p> elements.")

    result = Score()
    for ref, out in zip(ref_paragraphs, out_paragraphs, strict=True):
        original = paragraph_text(out)  # identical in both, thanks to the invariant
        expected, actual = corrected_text(ref), corrected_text(out)
        wanted, made = edits(original, expected), edits(original, actual)
        result.hits += len(wanted & made)
        result.extra += len(made - wanted)
        result.missed += len(wanted - made)
        result.exact += expected == actual
        result.paragraphs += 1
    return result
