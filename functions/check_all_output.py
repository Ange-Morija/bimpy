"""DEPRECATED MODULE

This module was removed from active use. The functionality was deprecated
when the package was reduced to a smaller set of core features.

If you need the previous implementation, recover it from version control.
"""

def _deprecated(*args, **kwargs):
    raise RuntimeError("functions.check_all_output is deprecated and intentionally disabled.")

__all__ = ['_deprecated']
