"""XML proofreader: proofreads <p> elements with a GenAI model and injects <error> tags.

The text inside each <p> is never changed. Errors are marked by wrapping the original
text in <error> elements whose attributes carry the suggested correction.
"""

__version__ = "0.1.0"
