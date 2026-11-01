"""Print each paragraph's <error> tags on short lines, for easy review.

Usage: uv run python scripts/show_errors.py samples/all_error_types.corrected.xml
"""

import sys
from pathlib import Path

from proofread.paragraph import is_paragraph
from proofread.xml_io import find_paragraphs, load_xml

doc = load_xml(Path(sys.argv[1]))
for n, p in enumerate(find_paragraphs(doc.root)):
    # Only this paragraph's own <error> tags, not those of a nested <p>.
    errors = [
        e for e in p.iter("{*}error") if next(a for a in e.iterancestors() if is_paragraph(a)) is p
    ]
    print(f"[{p.get('id') or n}]", "no errors" if not errors else "")
    for e in errors:
        print(f"    {e.get('type'):15} {e.text!r:22} -> {e.get('correction')!r}")
