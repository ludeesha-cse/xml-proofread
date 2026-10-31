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


def text_slots(p: etree._Element) -> list[TextSlot]:
    """List the non-empty text slots of a <p>, in reading order.

    Included: the <p>'s own text, and the text inside inline children such as <b>.
    Excluded: the content of comments, processing instructions, entities and nested
    <p> elements (because nested <p> is proofread on its own). The text *after* them (their
    tail) still belongs to this paragraph.
    """
    slots: list[TextSlot] = []
    position = 0

    def add(owner: etree._Element, attr: str) -> None:
        nonlocal position  # lets this inner function update the outer variable
        value = getattr(owner, attr) or ""
        if value:
            slots.append(TextSlot(owner, attr, position, position + len(value)))
            position += len(value)

    def walk(el: etree._Element) -> None:
        add(el, "text")
        for child in el:
            if isinstance(child.tag, str) and not is_paragraph(child):
                walk(child)  # inline element like <b>: its text is part of the sentence
            add(child, "tail")

    walk(p)
    return slots


def paragraph_text(p: etree._Element) -> str:
    """The paragraph's text exactly as a reader sees it, without any tags."""
    return "".join(slot.value for slot in text_slots(p))


def contains_cdata(p: etree._Element) -> bool:
    """True if the paragraph contains a CDATA section.

    lxml merges CDATA into normal text, so editing such a paragraph would silently
    turn <![CDATA[a<b]]> into a&lt;b. Tags also can't go inside CDATA at all, so these
    paragraphs are left unchanged.
    """
    return b"<![CDATA[" in etree.tostring(p, with_tail=False)
