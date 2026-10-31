"""How a <p> element's text is stored in lxml, as a list of "text slots".

lxml does not keep a paragraph's text in one string. In

    <p>Hello <b>big</b> world</p>

the text lives in three places: p.text = "Hello ", b.text = "big", b.tail = " world".
Each place is a TextSlot. Joining every slot gives the paragraph's text, and every
<error> tag we add must sit inside exactly one slot.
"""

from dataclasses import dataclass

from lxml import etree

# encapsulated: read-only computed property
@dataclass(frozen=True)
class TextSlot:
    """One piece of a paragraph's text: `owner.text` or `owner.tail`."""

    owner: etree._Element
    attr: str  # "text" or "tail"
    start: int  # position of this slot's first character in the paragraph text
    end: int  # position just after its last character

    @property
    def value(self) -> str:
        return getattr(self.owner, self.attr) or ""


def is_paragraph(el: etree._Element) -> bool:
    """True for <p> in any namespace. Comments and PIs have a non-string tag."""
    return isinstance(el.tag, str) and etree.QName(el).localname == "p"