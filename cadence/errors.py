"""Errors this package raises on its own behalf.

Two of these replace things that used to end the process or surface as a bare
RuntimeError. A library that calls `SystemExit` on a missing model cannot be
embedded in a web server, which is the one thing this package now has to do.
"""

from __future__ import annotations


class CadenceError(Exception):
    """Base class for every error this package raises deliberately."""


class ModelNotInstalled(CadenceError):
    """The spaCy model is not installed. Nothing can be parsed without it."""


class MissingAPIKey(CadenceError, RuntimeError):
    """No Anthropic credentials were supplied and none are in the environment.

    Also a RuntimeError, because callers written against the earlier API catch
    that and a rename should not silently stop their fallback from working.
    """
