"""Operator notification pipelines (Discord DM, etc.).

Failure-isolated, non-blocking trade alerts delivered to the operator's
personal Discord DMs via the standard REST API. Importing this package must
never block or raise even when credentials are absent — every dispatcher
degrades to a silent no-op.
"""

from .discord_dm import DiscordDMNotifier, get_notifier

__all__ = ["DiscordDMNotifier", "get_notifier"]