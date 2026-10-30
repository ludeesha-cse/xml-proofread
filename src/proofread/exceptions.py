"""Error types for the proofreader.

Every error the user can cause (bad flag, missing file, ...) is one of these.
Each carries an exit code, so the CLI can print a clear message and exit with
a predictable status instead of crashing with a stack trace.
"""


class ProofreadError(Exception):
    """Base class for all expected failures. Default exit code is 1."""

    exit_code = 1


class ConfigError(ProofreadError):
    """Invalid settings: bad --lang, bad --model, missing API key, bad numbers."""

    exit_code = 2


class InputFileError(ProofreadError):
    """The input file is missing, unreadable, or not valid XML."""

    exit_code = 3


class OutputFileError(ProofreadError):
    """The output file could not be written."""

    exit_code = 4