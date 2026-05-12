"""Session-state purge helpers for the « Nouvelle entrée » Streamlit tab.

This module avoids importing ``streamlit`` so ``src.auth`` can drop in-browser
draft keys on logout without circular imports with ``src.ui_components``.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

_NEW_ENTRY_PREFIX = "new_entry_"
_PENDING_CLEAR_PREFIX = "_pending_clear_new_entry_"


def purge_all_new_entry_session_state(session: MutableMapping[str, Any]) -> None:
    """Remove every new-entry draft key and pending-clear flag from ``session``.

    Used on logout and when the authenticated account changes so browser
    session state cannot attach another user's legacy buffers to the current
    account.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
    """
    for key in list(session.keys()):
        ks = str(key)
        if ks.startswith(_NEW_ENTRY_PREFIX) or ks.startswith(_PENDING_CLEAR_PREFIX):
            session.pop(key, None)
