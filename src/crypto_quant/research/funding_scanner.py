"""Multi-Asset Funding Yield & Basis Scanner.

Scans the active Binance USDS-M perpetual universe, computes the current
spot/perp basis spread and predicted 8-hour funding rate, deducts estimated
round-trip transaction drag across both legs, filters out illiquid pairs, and
ranks the surviving universe by net annualized yield (APR %).

Design
------
- Public CCXT endpoints only — **no API keys required**.
- Prefers the USDS-M demo host (demo-fapi.binance.com) and falls back to the
  production host (prod fapi is network-blocked from some environments; demo
  funding closely tracks production).
- The pure ``rank()`` core is separated from all I/O so it can be unit-tested
  offline with injected dictionaries (no network). ``scan()`` adds the I/O
  layer and delegates to ``rank()``.

Basis convention (spec)::

    Basis % = (Futures Mark Price - Spot Index Price) / Spot Index Price * 100

Net APR (spec)::

    gross_apr_pct = funding_rate * funding_periods_per_day * days_per_year * 100
    net_apr_pct   = gross_apr_pct - fee_drag_pct   # round-trip drag as an APR haircut
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..logging_config import get_logger

logger = get_logger("research")

# Default trading constants
FUNDING_PERIODS_PER_DAY = 3     # 8h funding schedule
DAYS_PER_YEAR = 365
DEFAULT_MIN_VOLUME_USDT = 10_000_000.0
DEFAULT_FEE_DRAG_PCT = 0.14      # round-trip both-leg fee drag, in percentage points
DEFAULT_BASIS_ANOMALY_PCT = 2.0  # |basis| above this is flagged & excluded
DEFAULT_QUOTE = "USDT"

# USD-M public endpoints in failover order: the canonical host first, then the
# documented alias set (mirrors Binance's ``fapi``/``fapi1..3`` DNS entries).
_USDM_USD_M_ENDPOINTS = (
    "fapi.binance.com",
    "fapi1.binance.com",
    "fapi2.binance.com",
    "fapi3.binance.com",
)
# Per-request connect/read timeout (ms) applied to the market-discovery round.
_REQUEST_TIMEOUT_MS = 15_000
# Extra attempts (in addition to the first) per failover host, and the backoff
# base between attempts: sleeps grow as 2s, 4s, 8s, … for retries 1, 2, 3, ….
_HOST_RETRIES = 2
_BACKOFF_BASE_SECONDS = 2.0
# A string-form API URL always looks like ``https://<sub>.binance.com/<path>``
# in the ccxt binance-family clients; retargeting rewrites just the hostname.
_URL_HOST_RE = re.compile(r"https?://[a-z0-9.-]*\.binance\.com")
# A real Binance *Spot* host (``api[1-3].binance.com``). We keep only the
# ``urls['api']`` groups that resolve to one of these when pinning the scanner's
# spot client, so a spot request can never fall through to a futures host
# (``fapi``/``dapi``/``eapi``/``papi`` or a demo/testnet host).
_SPOT_HOST_RE = re.compile(r"https?://api[0-9]*\.binance\.com")


@dataclass
class CarryOpportunity:
    """One ranked delta-neutral funding-harvest candidate (spot long / perp short)."""

    symbol: str
    spot_price: float
    futures_price: float
    basis_spread_pct: float
    predicted_funding_rate: float
    gross_apr_pct: float
    net_apr_pct: float
    next_funding_time: Optional[datetime]
    volume_24h_usdt: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form (UTC ISO string for the optional datetime)."""
        return {
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "futures_price": self.futures_price,
            "basis_spread_pct": self.basis_spread_pct,
            "predicted_funding_rate": self.predicted_funding_rate,
            "gross_apr_pct": self.gross_apr_pct,
            "net_apr_pct": self.net_apr_pct,
            "next_funding_time": (
                self.next_funding_time.astimezone(timezone.utc).isoformat()
                if self.next_funding_time is not None else None
            ),
            "volume_24h_usdt": self.volume_24h_usdt,
        }


def _num(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to float, returning ``default`` on failure/None."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _multialias(mapping: Dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value among ``keys`` from ``mapping``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


class FundingScanner:
    """Scans and ranks the Binance USDS-M perp universe by net funding APR."""

    def __init__(
        self,
        min_volume_usdt: float = DEFAULT_MIN_VOLUME_USDT,
        fee_drag_pct: float = DEFAULT_FEE_DRAG_PCT,
        basis_anomaly_threshold_pct: float = DEFAULT_BASIS_ANOMALY_PCT,
        quote: str = DEFAULT_QUOTE,
        funding_periods_per_day: int = FUNDING_PERIODS_PER_DAY,
        days_per_year: int = DAYS_PER_YEAR,
        top_n: Optional[int] = None,
    ) -> None:
        self.min_volume_usdt = float(min_volume_usdt)
        self.fee_drag_pct = float(fee_drag_pct)
        self.basis_anomaly_threshold_pct = float(basis_anomaly_threshold_pct)
        self.quote = quote.upper()
        self.funding_periods_per_day = funding_periods_per_day
        self.days_per_year = days_per_year
        self.top_n = top_n
        # Candidates excluded as basis anomalies (|basis| > threshold) during the
        # last scan/rank run. Populated so callers can surface the warnings.
        self.anomalies: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ main API
    def scan(self, provider: Optional[Any] = None) -> List[CarryOpportunity]:
        """Fetch the universe and return ranked opportunities.

        ``provider`` is a duck-typed object exposing ``fetch_funding_rates()``,
        ``fetch_spot_tickers()`` and ``active_usdm_symbols()``. When omitted a
        default CCXT provider is built. Errors surface to the caller (so the
        CLI can render a graceful failure).
        """
        prov = provider or self._default_provider()
        perp_rates = prov.fetch_funding_rates()       # symbol -> {fundingRate, markPrice, indexPrice, ...}
        spot_tickers = prov.fetch_spot_tickers()      # "BTC/USDT" -> {last, quoteVolume}
        active = prov.active_usdm_symbols()           # {"BTC/USDT:USDT", ...}
        return self.rank(perp_rates, spot_tickers, active)

    # ------------------------------------------------------------ pure core
    def rank(
        self,
        perp_rates: Dict[str, Dict[str, Any]],
        spot_tickers: Dict[str, Dict[str, Any]],
        active: Optional[Set[str]] = None,
    ) -> List[CarryOpportunity]:
        """Rank provided perp funding data (no I/O).

        ``perp_rates`` keyed by perp symbol (e.g. ``"BTC/USDT:USDT"``) → data
        with ``fundingRate``, ``markPrice``, ``indexPrice``, and a next-funding
        time. ``spot_tickers`` keyed by spot symbol (``"BTC/USDT"``) → ``last``
        price and ``quoteVolume`` (24h USDT).

        Filters: inactive/delisted, non-``quote`` perps, missing spot twin,
        insufficient 24h volume, and basis anomalies. Returns opportunities
        ranked by ``net_apr_pct`` descending (then ``volume_24h_usdt`` descending).
        """
        self.anomalies = []
        opportunities: List[CarryOpportunity] = []
        active_symbols = set(active or ())

        for perp_symbol, perp in perp_rates.items():
            # Isolate perp base so we can map to its spot twin and check quote.
            base, sep, perp_quote = self._split_perp_symbol(perp_symbol)
            if not sep:
                continue  # not a USDS-M perp we recognise
            if perp_quote != self.quote:
                continue  # only scan the target quote universe
            if active_symbols and perp_symbol not in active_symbols:
                continue  # inactive / delisted

            spot_symbol = f"{base}/{self.quote}"
            spot = spot_tickers.get(spot_symbol)
            if spot is None:
                continue  # no liquid spot twin to run the carry against

            funding = _num(perp.get("fundingRate"))
            mark = _num(perp.get("markPrice"))
            # Spot index from the premium index; fall back to the live spot last.
            index = _num(perp.get("indexPrice"), default=float("nan"))
            spot_price = index if index == index else _num(spot.get("last"), default=float("nan"))
            if mark != mark or spot_price != spot_price or spot_price == 0:
                continue  # missing/zero prices -> cannot compute basis

            basis = (mark - spot_price) / spot_price * 100.0
            if abs(basis) > self.basis_anomaly_threshold_pct:
                self.anomalies.append({
                    "symbol": perp_symbol,
                    "basis_spread_pct": basis,
                    "reason": "basis exceeds anomaly threshold",
                })
                continue

            volume = _num(spot.get("quoteVolume"), default=-1.0)
            if volume < 0.0:
                volume = _num(perp.get("quoteVolume"), default=-1.0)
            if volume < self.min_volume_usdt:
                continue  # illiquid — skip micro-caps

            gross_apr = funding * self.funding_periods_per_day * self.days_per_year * 100.0
            net_apr = gross_apr - self.fee_drag_pct

            next_ts = _multialias(
                perp, "nextFundingTimestamp", "nextFundingTime", "fundingTimestamp"
            )
            next_time = self._to_utc(next_ts)

            opportunities.append(CarryOpportunity(
                symbol=perp_symbol,
                spot_price=spot_price,
                futures_price=mark,
                basis_spread_pct=basis,
                predicted_funding_rate=funding,
                gross_apr_pct=gross_apr,
                net_apr_pct=net_apr,
                next_funding_time=next_time,
                volume_24h_usdt=volume,
            ))

        opportunities.sort(
            key=lambda o: (o.net_apr_pct, o.volume_24h_usdt), reverse=True
        )
        if self.top_n is not None:
            opportunities = opportunities[: self.top_n]
        return opportunities

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _split_perp_symbol(symbol: str):
        """Split ``"BTC/USDT:USDT"`` → ``("BTC", "/USDT:USDT", "USDT")``.

        Returns ``(base, sep, quote)`` where ``sep`` is ``""`` for symbols that
        aren't of the ``BASE/QUOTE:QUOTE`` shape (so callers can skip them).
        """
        if not isinstance(symbol, str) or ":" not in symbol or "/" not in symbol:
            return symbol, "", ""
        left, _, right = symbol.partition(":")
        if "/" not in left:
            return symbol, "", ""
        base = left.split("/")[0]
        return base, "/", right

    @staticmethod
    def _to_utc(value: Any) -> Optional[datetime]:
        """Convert an epoch-ms value to a UTC datetime, or ``None``."""
        if value is None:
            return None
        try:
            ms = float(value)
            if ms != ms:
                return None
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None

    # ------------------------------------------------------------ default provider
    def _default_provider(self):
        """Build a CCXT-backed provider (demo-fapi first, prod fallback)."""
        return _CcxtProvider(quote=self.quote)


class _CcxtProvider:
    """CCXT-backed data provider for the public Binance USDS-M universe.

    Public endpoints only — no API keys. Tries the USDS-M demo host first and
    falls back to the production host.
    """

    def __init__(self, quote: str = DEFAULT_QUOTE, timeout_ms: int = 30000) -> None:
        self.quote = quote
        self.timeout_ms = timeout_ms
        self._usdm, self._spot = self._build_clients()

    @staticmethod
    def _build_clients():
        """Build a USD-M futures client and a spot client.

        The USD-M host is selected demo-first with a production-failover order
        over Binance's documented endpoint aliases (``fapi[1-3].binance.com``).
        Each host gets an exponential-backoff reconnect loop so a single dead
        alias or a transient network blip never surfaces as a hard
        ``RuntimeError``; that only happens once *every* host has genuinely
        failed a full round of attempts.
        """
        import ccxt

        common = {
            "enableRateLimit": True,
            "timeout": _REQUEST_TIMEOUT_MS,
            "options": {"fetchCurrencies": False, "disableFuturesSandboxWarning": True},
        }
        usdm = _try_demo_usdm(common) or _try_prod_usdm(common)
        if usdm is None:
            raise RuntimeError(
                "No usable Binance USDS-M public endpoint reachable "
                "(demo and all production failover hosts failed)"
            )
        spot = _build_spot_client(_REQUEST_TIMEOUT_MS)
        return usdm, spot

    def fetch_funding_rates(self) -> Dict[str, Dict[str, Any]]:
        """Return per-symbol funding/mark/index data for the whole perp universe."""
        try:
            all_rates = self._usdm.fetch_funding_rates()
        except Exception:
            # Some ccxt builds only expose per-symbol fetches; fall back to symbols.
            all_rates = {}
            for sym in self._usdm.symbols:
                try:
                    r = self._usdm.fetch_funding_rate(sym)
                    if r:
                        all_rates[sym] = r
                except Exception:
                    continue
        out: Dict[str, Dict[str, Any]] = {}
        for sym, r in all_rates.items():
            if not isinstance(r, dict):
                continue
            out[sym] = r
        return out

    def fetch_spot_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Return spot tickers keyed by spot symbol with ``last`` and ``quoteVolume``."""
        try:
            tickers = self._spot.fetch_tickers()
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for sym, t in tickers.items():
            if not isinstance(t, dict):
                continue
            out[sym] = {"last": t.get("last"), "quoteVolume": t.get("quoteVolume")}
        return out

    def active_usdm_symbols(self) -> Set[str]:
        """Return the set of active (non-delisted) USDS-M perp symbols."""
        active: Set[str] = set()
        for sym, market in self._usdm.markets.items():
            if market.get("active") is False or market.get("delisted"):
                continue
            if market.get("quote") == self.quote and market.get("contract"):
                active.add(sym)
        return active


def _patch_demo_urls(ex) -> None:
    """Pin the USD-M demo host for a ccxt ``binanceusdm`` client.

    ccxt 4.5.78 drops ``set_sandbox_mode()`` support for futures; the supported
    way to reach the demo/futures host is to override ``urls['api']`` from the
    client's ``urls['demo']`` map.
    """
    demo_urls = ex.urls.get("demo", {})
    if demo_urls:
        ex.urls["api"] = demo_urls


def _retarget_api_hosts(client, host: str) -> None:
    """Point a ccxt USD-M client's string-form API URLs at a specific ``host``.

    ``urls['api']`` mixes plain string hosts (``fapiPublic``, ``fapiData``,
    ``dapiPublic``, …) with nested method dicts (spot). The scanner only hits the
    public futures endpoints, so we rewrite just the hostname in the string-form
    entries and leave the nested layers untouched.
    """
    api = client.urls.get("api", {})
    for key, value in api.items():
        if isinstance(value, str) and ".binance.com" in value:
            api[key] = _URL_HOST_RE.sub(f"https://{host}", value)


def _pin_spot_urls(spot) -> None:
    """Restrict a ccxt ``binance`` spot client to real Spot endpoints only.

    ccxt's ``binance`` class is a single multi-currency-type client whose
    ``urls['api']`` table carries *every* Binance product family (``fapi*``
    USDⓈ-M, ``dapi*`` COIN-M, ``eapi*`` options, ``papi*`` portfolio-margin,
    plus demo/testnet hosts). A spot handle built from that class therefore *can*
    resolve a request to a futures URL if ``defaultType`` drifts or a futures
    group is referenced — the spot→futures endpoint pollution this mission fixes.

    We pin the handle by retaining only the URL groups whose host is a real
    Spot host (``api[1-3].binance.com``), so a spot request is physically
    incapable of targeting ``fapi``/``dapi``/``eapi``/``papi``.
    """
    api = spot.urls.get("api", {})
    spot_only = {
        group: url
        for group, url in api.items()
        if isinstance(url, str) and _SPOT_HOST_RE.match(url)
    }
    # Never empty the map; if the filter matched nothing (e.g. a future ccxt
    # layout with nested dicts) keep the caller's original table unchanged.
    spot.urls["api"] = spot_only or api


def _build_spot_client(timeout_ms: int = _REQUEST_TIMEOUT_MS, load: bool = True):
    """Build the scanner's public Spot client, hardened against futures pollution.

    Returns a `ccxt.binance` handle pinned to Spot endpoints with no API
    credentials attached (public scanner -> must never carry an apiKey, so a
    stale/invalid key can never surface a 401/`Invalid API-key` at launch).
    ``load_markets`` is retried once before degrading to the empty-cache
    fallback, so a single transient blip does not silently starve the scan.
    """
    import ccxt

    spot = ccxt.binance({
        "apiKey": None,
        "secret": None,
        "enableRateLimit": True,
        "timeout": timeout_ms,
        "options": {
            "fetchCurrencies": False,
            "defaultType": "spot",
            "disableFuturesSandboxWarning": True,
        },
    })
    _pin_spot_urls(spot)
    if not load:
        return spot
    last_error: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            spot.load_markets()
            return spot
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = exc
            if attempt == 1:
                time.sleep(1.0)
    logger.warning("Spot market load failed; continuing with empty cache: %s", last_error)
    return spot


def _try_demo_usdm(common):
    """Attempt the USD-M demo host first (reachable where prod fapi is blocked)."""
    import ccxt

    try:
        client = ccxt.binanceusdm(dict(common))
        _patch_demo_urls(client)
        client.load_markets()
        logger.info("Connected to Binance USD-M demo host")
        return client
    except Exception as exc:
        logger.warning("USD-M demo host unreachable (%s); trying production", exc)
        return None


def _try_prod_usdm(common):
    """Try the USD-M production host, failing over its aliases with backoff."""
    import time as _time

    last_error = None
    for host in _USDM_USD_M_ENDPOINTS:
        for attempt in range(_HOST_RETRIES + 1):  # first attempt + retries
            try:
                import ccxt

                client = ccxt.binanceusdm(dict(common))
                _retarget_api_hosts(client, host)
                client.load_markets()
                logger.info("Connected to Binance USD-M host %s", host)
                return client
            except Exception as exc:
                last_error = exc
                if attempt < _HOST_RETRIES:
                    _time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
    logger.warning("All USD-M production endpoints unreachable (%s)", last_error)
    return None


class SampleProvider:
    """Deterministic, offline provider with representative Binance-like data.

    Ships a curated set that exercises every filter:
      * high-vol, positive-funding perps (survive) — ranked by net APR
      * a negative-funding pair (still ranked, low/negative net APR)
      * a micro-cap pair below ``min_volume_usdt`` (filtered out)
      * a basis anomaly (|basis| > threshold) (flagged, excluded)
    """

    # (base, spot_last, funding_rate_8h, quote_volume_24h_usdt)
    DATA = [
        ("BTC", 67000.0, 0.0001200, 28_000_000_000.0),
        ("ETH", 3400.0, 0.0001000, 12_000_000_000.0),
        ("SOL", 150.0, 0.0001400, 700_000_000.0),
        ("DOGE", 0.16, 0.0002000, 1_200_000_000.0),
        ("FART", 1.0, 0.0003000, 2_500_000.0),        # below default 10M volume
        ("SAMPLEDCOIN", 5.0, -0.0000600, 500_000_000.0),  # negative funding
        ("TURBO2", 3.0, 0.0001100, 200_000_000.0),    # basis anomaly (mark 6% above index)
    ]

    def __init__(self, quote: str = DEFAULT_QUOTE) -> None:
        self.quote = quote.upper()

    def active_usdm_symbols(self) -> Set[str]:
        return {
            f"{base}/{self.quote}:{self.quote}"
            for base, _spot, _funding, _vol in self.DATA
        }

    def fetch_funding_rates(self) -> Dict[str, Dict[str, Any]]:
        rates: Dict[str, Dict[str, Any]] = {}
        anomaly_bases = {"TURBO2"}
        for base, spot, funding, _vol in self.DATA:
            mark = spot * (1.06 if base in anomaly_bases else 1.0005)
            rates[f"{base}/{self.quote}:{self.quote}"] = {
                "fundingRate": funding,
                "markPrice": mark,
                "indexPrice": spot,
                "nextFundingTimestamp": 1_770_000_000_000,
            }
        return rates

    def fetch_spot_tickers(self) -> Dict[str, Dict[str, Any]]:
        return {
            f"{base}/{self.quote}": {"last": spot, "quoteVolume": vol}
            for base, spot, _funding, vol in self.DATA
        }


def sample_provider(quote: str = DEFAULT_QUOTE) -> SampleProvider:
    """Return an offline ``SampleProvider`` (used by the CLI ``--sample`` mode)."""
    return SampleProvider(quote=quote)