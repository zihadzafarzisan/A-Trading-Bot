"""Local analytics: append-only forensic trade recording & reflection."""

from .ledger import TradeLedger

# ReflectionEngine is imported lazily (PEP 562 __getattr__) rather than eagerly
# here. An eager top-level `from .reflect import ReflectionEngine` pre-populates
# sys.modules["...reflect"] before `python -m ...reflect` gets a chance to run
# that module, which triggers runpy's "found in sys.modules after import ... but
# prior to execution" RuntimeWarning. Loading it on attribute access keeps
# `from crypto_quant.analytics import ReflectionEngine` working unchanged.

__all__ = ["TradeLedger", "ReflectionEngine"]


def __getattr__(name: str):
    if name == "ReflectionEngine":
        from .reflect import ReflectionEngine

        return ReflectionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")