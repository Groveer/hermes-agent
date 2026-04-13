"""URL safety checks — blocks requests to private/internal network addresses.

Prevents SSRF (Server-Side Request Forgery) where a malicious prompt or
skill could trick the agent into fetching internal resources like cloud
metadata endpoints (169.254.169.254), localhost services, or private
network hosts.

The check can be globally disabled via ``security.allow_private_urls: true``
in config.yaml for environments where DNS resolves external domains to
private/benchmark-range IPs (OpenWrt routers, corporate proxies, VPNs
that use 198.18.0.0/15 or 100.64.0.0/10).  Even when disabled, cloud
metadata hostnames (metadata.google.internal, 169.254.169.254) are
**always** blocked — those are never legitimate agent targets.

Limitations (documented, not fixable at pre-flight level):
  - DNS rebinding (TOCTOU): an attacker-controlled DNS server with TTL=0
    can return a public IP for the check, then a private IP for the actual
    connection. Fixing this requires connection-level validation (e.g.
    Python's Champion library or an egress proxy like Stripe's Smokescreen).
  - Redirect-based bypass is mitigated by httpx event hooks that re-validate
    each redirect target in vision_tools, gateway platform adapters, and
    media cache helpers. Web tools use third-party SDKs (Firecrawl/Tavily)
    where redirect handling is on their servers.
"""

import ipaddress
import logging
import os
import socket
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Cache for SSRF whitelist config (similar to website_policy.py pattern)
_WHITELIST_CACHE_TTL = 30.0
_whitelist_lock = threading.Lock()
_cached_whitelist: Optional[List[ipaddress.IPv4Network | ipaddress.IPv6Network]] = None
_whitelist_cache_time: float = 0.0

# Hostnames that should always be blocked regardless of IP resolution
# or any config toggle.  These are cloud metadata endpoints that an
# attacker could use to steal instance credentials.
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# IPs and networks that should always be blocked regardless of the
# allow_private_urls toggle.  These are cloud metadata / credential
# endpoints — the #1 SSRF target — and the link-local range where
# they all live.
_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle metadata
    ipaddress.ip_address("169.254.170.2"),     # AWS ECS task metadata (task IAM creds)
    ipaddress.ip_address("169.254.169.253"),   # Azure IMDS wire server
    ipaddress.ip_address("fd00:ec2::254"),     # AWS metadata (IPv6)
    ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud metadata
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),    # Entire link-local range (no legit agent target)
)

# Exact HTTPS hostnames allowed to resolve to private/benchmark-space IPs.
# This is intentionally narrow: QQ media downloads can legitimately resolve
# to 198.18.0.0/15 behind local proxy/benchmark infrastructure.
_TRUSTED_PRIVATE_IP_HOSTS = frozenset({
    "multimedia.nt.qq.com.cn",
})

# 100.64.0.0/10 (CGNAT / Shared Address Space, RFC 6598) is NOT covered by
# ipaddress.is_private — it returns False for both is_private and is_global.
# Must be blocked explicitly. Used by carrier-grade NAT, Tailscale/WireGuard
# VPNs, and some cloud internal networks.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# ---------------------------------------------------------------------------
# Global toggle: allow private/internal IP resolution
# ---------------------------------------------------------------------------
# Cached after first read so we don't hit the filesystem on every URL check.
_allow_private_resolved = False
_cached_allow_private: bool = False


def _global_allow_private_urls() -> bool:
    """Return True when the user has opted out of private-IP blocking.

    Checks (in priority order):
    1. ``HERMES_ALLOW_PRIVATE_URLS`` env var  (``true``/``1``/``yes``)
    2. ``security.allow_private_urls`` in config.yaml
    3. ``browser.allow_private_urls`` in config.yaml  (legacy / backward compat)

    Result is cached for the process lifetime.
    """
    global _allow_private_resolved, _cached_allow_private
    if _allow_private_resolved:
        return _cached_allow_private

    _allow_private_resolved = True
    _cached_allow_private = False  # safe default

    # 1. Env var override (highest priority)
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in ("true", "1", "yes"):
        _cached_allow_private = True
        return _cached_allow_private
    if env_val in ("false", "0", "no"):
        # Explicit false — don't fall through to config
        return _cached_allow_private

    # 2. Config file
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        # security.allow_private_urls (preferred)
        sec = cfg.get("security", {})
        if isinstance(sec, dict) and sec.get("allow_private_urls"):
            _cached_allow_private = True
            return _cached_allow_private
        # browser.allow_private_urls (legacy fallback)
        browser = cfg.get("browser", {})
        if isinstance(browser, dict) and browser.get("allow_private_urls"):
            _cached_allow_private = True
            return _cached_allow_private
    except Exception:
        # Config unavailable (e.g. tests, early import) — keep default
        pass

    return _cached_allow_private


def _reset_allow_private_cache() -> None:
    """Reset the cached toggle — only for tests."""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = False
    _cached_allow_private = False


def _load_ssrf_whitelist() -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Load SSRF whitelist from config.yaml.

    Returns a list of IP networks that should be exempt from private/internal
    checks. Caches results for 30 seconds to avoid repeated YAML parsing.
    """
    global _cached_whitelist, _whitelist_cache_time

    now = time.monotonic()
    with _whitelist_lock:
        if _cached_whitelist is not None and (now - _whitelist_cache_time) < _WHITELIST_CACHE_TTL:
            return _cached_whitelist

    config_path = get_hermes_home() / "config.yaml"
    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []

    if not config_path.exists():
        with _whitelist_lock:
            _cached_whitelist = networks
            _whitelist_cache_time = now
        return networks

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("Failed to load config for SSRF whitelist: %s", exc)
        with _whitelist_lock:
            _cached_whitelist = networks
            _whitelist_cache_time = now
        return networks

    security = config.get("security", {}) or {}
    whitelist_raw = security.get("ssrf_whitelist", []) or []

    if not isinstance(whitelist_raw, list):
        logger.warning("security.ssrf_whitelist must be a list, got %s", type(whitelist_raw).__name__)
        with _whitelist_lock:
            _cached_whitelist = networks
            _whitelist_cache_time = now
        return networks

    for entry in whitelist_raw:
        if not isinstance(entry, str):
            continue
        try:
            network = ipaddress.ip_network(entry.strip(), strict=False)
            networks.append(network)
            logger.debug("SSRF whitelist: added network %s", network)
        except ValueError as exc:
            logger.warning("Invalid SSRF whitelist entry '%s': %s", entry, exc)

    with _whitelist_lock:
        _cached_whitelist = networks
        _whitelist_cache_time = now

    return networks


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection.

    Checks against private/reserved ranges, then exempts IPs that match
    the user-configured ssrf_whitelist.
    """
    # First check if IP is in the whitelist (exempt from blocking)
    whitelist = _load_ssrf_whitelist()
    for network in whitelist:
        if ip in network:
            logger.debug("IP %s exempted by SSRF whitelist (in %s)", ip, network)
            return False

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    # CGNAT range not covered by is_private
    if ip in _CGNAT_NETWORK:
        return True
    return False


def _allows_private_ip_resolution(hostname: str, scheme: str) -> bool:
    """Return True when a trusted HTTPS hostname may bypass IP-class blocking."""
    return scheme == "https" and hostname in _TRUSTED_PRIVATE_IP_HOSTS


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname to an IP and checks against private ranges.
    Fails closed: DNS errors and unexpected exceptions block the request.

    When ``security.allow_private_urls`` is enabled (or the env var
    ``HERMES_ALLOW_PRIVATE_URLS=true``), private-IP blocking is skipped.
    Cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    remain blocked regardless — they are never legitimate agent targets.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        scheme = (parsed.scheme or "").strip().lower()
        if not hostname:
            return False

        # Block known internal hostnames — ALWAYS, even with toggle on
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        # Check the global toggle AFTER blocking metadata hostnames
        allow_all_private = _global_allow_private_urls()

        allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

        # Try to resolve and check IP
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            # DNS resolution failed — fail closed. If DNS can't resolve it,
            # the HTTP client will also fail, so blocking loses nothing.
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            # Always block cloud metadata IPs and link-local, even with toggle on
            if ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS):
                logger.warning(
                    "Blocked request to cloud metadata address: %s -> %s",
                    hostname, ip_str,
                )
                return False

            if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        if allow_all_private:
            logger.debug(
                "Allowing private/internal resolution (security.allow_private_urls=true): %s",
                hostname,
            )
        elif allow_private_ip:
            logger.debug(
                "Allowing trusted hostname despite private/internal resolution: %s",
                hostname,
            )

        return True

    except Exception as exc:
        # Fail closed on unexpected errors — don't let parsing edge cases
        # become SSRF bypass vectors
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False
