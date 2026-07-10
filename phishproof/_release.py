"""Release gating for the public PhishProof skeleton.

The public repository ships the full package structure, the class and function
signatures, and the documentation of the PhishProof pipeline, but withholds the
executable body of every method until the paper is accepted. Each stubbed
callable delegates to :func:`pending`, which surfaces a readable notice instead
of running the (withheld) implementation.
"""

from __future__ import annotations

import sys
from typing import NoReturn

RELEASE_NOTICE = (
    "PhishProof reference implementation is not public yet. The full source of "
    "this component will be released in this repository once the paper is "
    "accepted; the public release currently ships the interface and "
    "documentation only."
)


def pending(component: str = "") -> NoReturn:
    """Announce a withheld implementation and stop execution.

    Every function body in this skeleton calls ``pending(...)``. It prints the
    release notice -- so running the code surfaces a human-readable message --
    and then raises :class:`NotImplementedError` so that no caller silently
    receives an empty or wrong result.

    Parameters
    ----------
    component:
        Dotted name of the withheld callable, echoed in the message to make
        tracebacks self-explanatory.
    """
    where = f" [{component}]" if component else ""
    print(f"{RELEASE_NOTICE}{where}", file=sys.stderr)
    raise NotImplementedError(f"{RELEASE_NOTICE}{where}")
