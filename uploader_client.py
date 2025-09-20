# backend/uploader_client.py
"""
Helpers used by backend/app.py to:
- query discovery for peers
- measure RTT to peers
- pick nearest peers
- upload chunk files to chosen peers (with retries/backoff + concurrency)
- fetch chunk bytes from peers (for download/reassemble)

Public functions:
- get_peers(discovery_url, limit=CANDIDATE_LIMIT)
- distribute_chunk(discovery_url, chunk_path, chunk_hash, replication=DEFAULT_REPLICATION)
- fetch_chunk_from_peer(peer, chunk_hash, timeout=10)
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import List, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger("uploader_client")
LOG.setLevel(logging.INFO)

DEFAULT_REPLICATION = 3
CANDIDATE_LIMIT = 40
HEAD_TIMEOUT = 1.5
MEASURE_MAX_WORKERS = 16
UPLOAD_MAX_WORKERS = 8
UPLOAD_RETRIES = 2
UPLOAD_TIMEOUT = 10
BACKOFF_BASE = 0.5  # seconds


def _make_session(retries: int = 2, backoff_factor: int = 1, status_forcelist=(500, 502, 503, 504)):
    """
    Create a requests.Session with a Retry policy.
    Note: some urllib3 type stubs expect backoff_factor as int; we pass an int to satisfy linters.
    If your urllib3 accepts floats and you want fractional backoff, change the parameter type.
    """
    s = requests.Session()
    # allowed_methods is the newer name in urllib3; if your stub complains change to method_whitelist
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def get_peers(discovery_url: str, limit: int = CANDIDATE_LIMIT, timeout: float = 4.0) -> List[Dict]:
    """
    Query discovery service and return a list of peer dicts.
    Expected peer dict keys: node_id, ip, port, meta, last_heartbeat
    """
    if not discovery_url:
        return []
    s = _make_session()
    url = f"{discovery_url.rstrip('/')}/peers"
    try:
        r = s.get(url, params={"limit": limit}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("peers", []) or []
    except Exception as e:
        LOG.warning("get_peers failed: %s", e)
        return []


def measure_rtt(peer: dict, head_timeout: float = HEAD_TIMEOUT) -> float:
    """
    Measure RTT to peer by issuing HEAD to /store endpoint.
    Returns RTT seconds or float('inf') on failure.
    """
    if not isinstance(peer, dict):
        return float("inf")
    ip = peer.get("ip")
    port = peer.get("port")
    if not ip or not port:
        return float("inf")
    url = f"http://{ip}:{port}/store"
    start = time.time()
    s = _make_session(retries=0)
    try:
        # HEAD may 405; treat any response as success for timing
        s.head(url, timeout=head_timeout, allow_redirects=False)
        return time.time() - start
    except Exception:
        return float("inf")


def pick_nearest(peers: List[dict], r: int = DEFAULT_REPLICATION, max_workers: int = MEASURE_MAX_WORKERS, timeout: float = 6.0) -> List[dict]:
    """
    Measure RTTs in parallel and return the r nearest peers (by RTT).
    Uses wait(...) to collect completed measurements within timeout and treats others as infinite RTT.
    """
    if not peers:
        return []
    max_workers = min(max_workers, len(peers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(measure_rtt, p): p for p in peers}
        done, not_done = wait(futures.keys(), timeout=timeout)
        results = []
        # collect results for done futures
        for fut in done:
            p = futures.get(fut)
            try:
                rtt = fut.result()
            except Exception:
                rtt = float("inf")
            results.append((rtt, p))
        # not finished -> treat as inf (but include in list for fallback)
        for fut in not_done:
            p = futures.get(fut)
            results.append((float("inf"), p))

    results.sort(key=lambda x: x[0])
    chosen = [p for (lat, p) in results if lat < float("inf")][:r]
    if len(chosen) < r:
        chosen = [p for (_, p) in results][:r]
    return chosen


def upload_chunk_to_peer(session: requests.Session, peer: dict, chunk_path: str, chunk_hash: str, timeout: int = UPLOAD_TIMEOUT, retries: int = UPLOAD_RETRIES) -> Dict:
    """
    Upload a chunk file to a peer. Returns assignment dict with node_id, ip, port, status, error, duration.
    """
    node_id = peer.get("node_id")
    ip = peer.get("ip")
    port = peer.get("port")
    rec = {"node_id": node_id, "ip": ip, "port": port}
    if not ip or not port:
        rec["status"] = "fail"
        rec["last_error"] = "missing ip/port"
        return rec

    url = f"http://{ip}:{port}/store"
    attempt = 0
    start_total = time.time()
    while attempt <= retries:
        try:
            with open(chunk_path, "rb") as fh:
                files = {"file": (chunk_hash, fh)}
                data = {"file_hash": chunk_hash}
                resp = session.post(url, files=files, data=data, timeout=timeout)
                resp.raise_for_status()
                try:
                    j = resp.json()
                    if isinstance(j, dict):
                        rec.update(j)
                except Exception:
                    # ignore JSON parse errors
                    pass
                rec["status"] = "ok"
                rec["duration"] = time.time() - start_total
                return rec
        except Exception as e:
            rec["last_error"] = str(e)
            attempt += 1
            if attempt > retries:
                rec["status"] = "fail"
                rec["duration"] = time.time() - start_total
                return rec
            backoff = BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(backoff)
    rec["status"] = "fail"
    rec["duration"] = time.time() - start_total
    return rec


def distribute_chunk(discovery_url: str, chunk_path: str, chunk_hash: str, replication: int = DEFAULT_REPLICATION, max_workers: int = UPLOAD_MAX_WORKERS) -> List[dict]:
    """
    Pick nearest peers via discovery and upload the chunk to them in parallel.
    Returns list of assignment dicts:
      {"node_id":..., "ip":..., "port":..., "status":"ok"|"fail", "last_error":..., "duration":...}
    """
    peers = get_peers(discovery_url)
    if not peers:
        LOG.warning("distribute_chunk: no peers from discovery")
        return []

    chosen = pick_nearest(peers, r=replication)
    if not chosen:
        LOG.warning("distribute_chunk: no chosen peers")
        return []

    session = _make_session(retries=1, backoff_factor=1)
    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chosen))) as ex:
        futs = {ex.submit(upload_chunk_to_peer, session, p, chunk_path, chunk_hash): p for p in chosen}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:
                p = futs[f]
                rec = {"node_id": p.get("node_id"), "ip": p.get("ip"), "port": p.get("port"), "status": "fail", "last_error": str(e)}
            results.append(rec)
    return results


def fetch_chunk_from_peer(peer: dict, chunk_hash: str, timeout: int = UPLOAD_TIMEOUT) -> bytes:
    """
    Download chunk bytes from a peer descriptor: expected keys ip, port.
    Raises requests exceptions on failure (caller should handle).
    """
    if not isinstance(peer, dict):
        raise ValueError("peer must be a dict with ip and port")
    ip = peer.get("ip")
    port = peer.get("port")
    if not ip or not port:
        raise ValueError("peer must contain ip and port")
    url = f"http://{ip}:{port}/retrieve/{chunk_hash}"
    session = _make_session(retries=1, backoff_factor=1)
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content
