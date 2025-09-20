# app.py
import os
import hashlib
import time
import logging
from pathlib import Path
from typing import List

import requests
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    abort,
    Response,
)
from werkzeug.utils import secure_filename

from blockchain import SimpleChain
from chunker import chunk_file_and_save, compute_merkle_root, CHUNK_SIZE

# try to import uploader_client if available (discovery-aware placement)
try:
    import backend.uploader_client as uploader_client
except Exception:
    uploader_client = None

# ---------- Configuration ----------
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)
CHAIN_PATH = STORAGE_DIR / "chain.json"
ALLOWED_EXTENSIONS = None  # set a set(...) to restrict types if desired

# Legacy: Peers mapping for single-machine/testing mode
PEER_PORTS = {"node1": 5001, "node2": 5002, "node3": 5003}
N = len(PEER_PORTS)

# fallback local storage for chunks if peer unreachable
FALLBACK_CHUNKS = STORAGE_DIR / "fallback_chunks"
FALLBACK_CHUNKS.mkdir(parents=True, exist_ok=True)

# Discovery configuration (if empty -> legacy fallback)
DISCOVERY_URL = os.environ.get("DISCOVERY_URL", "").strip()
REPLICATION = int(os.environ.get("REPLICATION", "3"))

# Flask app — serve frontend
app = Flask(
    __name__,
    static_folder=str(BASE_DIR.parent / "frontend"),
    template_folder=str(BASE_DIR.parent / "frontend"),
)

# Blockchain (simple JSON-persisted chain)
chain = SimpleChain(path=str(CHAIN_PATH))


# ---------- Helpers ----------
def allowed_file(filename: str) -> bool:
    if ALLOWED_EXTENSIONS is None:
        return True
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# legacy/local single-machine placement (hash%N)
def send_chunk_to_peer_legacy(chunk_hash: str, chunk_path: Path, retries: int = 2, timeout: int = 6) -> str:
    idx = int(hashlib.sha256(chunk_hash.encode()).hexdigest(), 16) % N
    node_id = list(PEER_PORTS.keys())[idx]
    port = PEER_PORTS[node_id]
    url = f"http://127.0.0.1:{port}/store"
    for attempt in range(retries + 1):
        try:
            with chunk_path.open("rb") as fh:
                files = {"file": (chunk_path.name, fh)}
                data = {"file_hash": chunk_hash}
                resp = requests.post(url, files=files, data=data, timeout=timeout)
                resp.raise_for_status()
                try:
                    j = resp.json()
                    returned = j.get("node_id")
                    if returned:
                        LOG.info("Peer %s acknowledged store of %s", returned, chunk_hash)
                except Exception:
                    pass
                return node_id
        except Exception as e:
            LOG.warning("Failed to store chunk %s to %s (attempt %d/%d): %s", chunk_hash, node_id, attempt, retries, e)
            time.sleep(0.5 * (attempt + 1))
    # fallback: copy to local fallback folder
    fallback_path = FALLBACK_CHUNKS / chunk_path.name
    try:
        with chunk_path.open("rb") as src, fallback_path.open("wb") as dst:
            dst.write(src.read())
        LOG.warning("Falling back to local storage for chunk %s -> %s", chunk_hash, fallback_path)
        return "local_fallback"
    except Exception as e:
        LOG.error("Failed to fallback-store chunk %s: %s", chunk_hash, e)
        return "local_fallback"


def fetch_chunk_bytes_from_peer(peer_ip: str, peer_port: int, chunk_hash: str, timeout: int = 10) -> bytes:
    url = f"http://{peer_ip}:{peer_port}/retrieve/{chunk_hash}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_chunk_bytes(chunk_record: dict, chunks_dir: Path) -> bytes:
    """
    Attempts to fetch chunk bytes using multiple possible assignment formats:
    - new style: 'assignments' list (each assignment has node_id, ip, port, status)
    - legacy style: 'node_id' string that maps to PEER_PORTS (single-machine)
    Falls back to FALLBACK_CHUNKS and local chunks_dir.
    """
    chunk_hash = chunk_record["hash"]

    # 1) New-style 'assignments' produced by uploader_client.distribute_chunk
    if "assignments" in chunk_record and isinstance(chunk_record["assignments"], list):
        for a in chunk_record["assignments"]:
            # only try successful uploads
            if a.get("status") != "ok":
                continue
            # if uploader_client stored the ip/port in the assignment, use that
            if a.get("ip") and a.get("port"):
                try:
                    return fetch_chunk_bytes_from_peer(a["ip"], int(a["port"]), chunk_hash)
                except Exception as e:
                    LOG.warning("Failed to fetch chunk %s from assignment %s:%s -> %s", chunk_hash, a.get("ip"), a.get("port"), e)
                    continue
            # otherwise, if only node_id present and DISCOVERY_URL available, query discovery for node info
            if a.get("node_id") and DISCOVERY_URL:
                try:
                    r = requests.get(f"{DISCOVERY_URL.rstrip('/')}/peer/{a['node_id']}", timeout=4)
                    if r.status_code == 200:
                        peer = r.json()
                        try:
                            return fetch_chunk_bytes_from_peer(peer["ip"], int(peer["port"]), chunk_hash)
                        except Exception as e:
                            LOG.warning("Failed to fetch chunk %s from peer %s via discovery -> %s", chunk_hash, a["node_id"], e)
                            continue
                except Exception as e:
                    LOG.warning("Discovery lookup failed for node %s: %s", a.get("node_id"), e)
                    continue

    # 2) Legacy 'node_id' field mapping to PEER_PORTS (single-machine)
    node_id = chunk_record.get("node_id")
    if node_id and node_id in PEER_PORTS and node_id != "local_fallback":
        port = PEER_PORTS[node_id]
        try:
            return fetch_chunk_bytes_from_peer("127.0.0.1", port, chunk_hash)
        except Exception as e:
            LOG.warning("Failed to fetch chunk %s from legacy peer %s:%s -> %s", chunk_hash, "127.0.0.1", port, e)

    # 3) Try fallback stored chunk file (backend fallback store)
    fallback_path = FALLBACK_CHUNKS / f"chunk_{chunk_record['index']:06d}_{chunk_hash}.bin"
    if fallback_path.exists():
        return fallback_path.read_bytes()

    # 4) Try local chunks dir
    local_path = chunks_dir / f"chunk_{chunk_record['index']:06d}_{chunk_hash}.bin"
    if local_path.exists():
        return local_path.read_bytes()

    raise FileNotFoundError(f"Chunk {chunk_hash} not found on any peer or local storage.")


def chunk_has_source(chunk_record: dict, chunks_dir: Path) -> bool:
    """
    Quick pre-check: returns True if we can reasonably expect chunk to be retrievable
    from either:
      - an 'assignments' entry with status 'ok' and (ip/port present or node_id known)
      - a legacy node_id mapping (PEER_PORTS) OR
      - a fallback/local chunk file
    This is a cheap check (does not attempt HTTP GET), used to fail fast before streaming.
    """
    # assignments present?
    if "assignments" in chunk_record and isinstance(chunk_record["assignments"], list):
        for a in chunk_record["assignments"]:
            if a.get("status") != "ok":
                continue
            if a.get("ip") and a.get("port"):
                return True
            if a.get("node_id") and DISCOVERY_URL:
                return True
            if a.get("node_id") and a.get("node_id") in PEER_PORTS:
                return True

    # legacy node_id mapping
    node_id = chunk_record.get("node_id")
    if node_id and (node_id in PEER_PORTS or node_id == "local_fallback"):
        return True

    # fallback or local file exists
    fallback_path = FALLBACK_CHUNKS / f"chunk_{chunk_record['index']:06d}_{chunk_record['hash']}.bin"
    if fallback_path.exists():
        return True
    local_path = chunks_dir / f"chunk_{chunk_record['index']:06d}_{chunk_record['hash']}.bin"
    if local_path.exists():
        return True

    return False


# ---------- Routes ----------
@app.route("/")
def index():
    return send_from_directory(str(app.static_folder), "index.html")


@app.route("/_config", methods=["GET"])
def _config():
    return jsonify({"chunk_size": CHUNK_SIZE, "replication": REPLICATION, "discovery": DISCOVERY_URL})


# simple upload (not chunked)
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename or ""):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename or "")
    ts = int(time.time())
    saved_name = f"{ts}_{filename}"
    saved_path = STORAGE_DIR / saved_name
    file.save(str(saved_path))

    size = saved_path.stat().st_size
    sha256 = file_sha256(saved_path)

    metadata = {
        "filename": filename,
        "stored_name": saved_name,
        "size": size,
        "sha256": sha256,
        "timestamp": ts,
    }

    block_entry = chain.add_block([metadata])
    return jsonify({"status": "ok", "metadata": metadata, "block": block_entry})


# upload + chunk + distribute to peers
@app.route("/upload_and_chunk", methods=["POST"])
def upload_and_chunk():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename or ""):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename or "")
    ts = int(time.time())
    saved_name = f"{ts}_{filename}"
    saved_path = STORAGE_DIR / saved_name
    file.save(str(saved_path))

    # 1) chunk locally
    chunks_dir = STORAGE_DIR / f"{saved_name}_chunks"
    chunk_hashes = chunk_file_and_save(saved_path, chunks_dir, chunk_size=CHUNK_SIZE)

    # 2) compute merkle root
    merkle_root = compute_merkle_root(chunk_hashes)

    # 3) distribute each chunk
    chunk_records = []
    for idx, h in enumerate(chunk_hashes):
        chunk_fname = f"chunk_{idx:06d}_{h}.bin"
        chunk_path = chunks_dir / chunk_fname
        if not chunk_path.exists():
            LOG.error("Expected chunk file missing: %s", chunk_path)
            # record missing so download knows
            chunk_records.append({"index": idx, "hash": h, "node_id": "missing"})
            continue

        # If discovery mode is enabled and uploader_client available, use it
        if DISCOVERY_URL and uploader_client:
            try:
                assignments = uploader_client.distribute_chunk(DISCOVERY_URL, str(chunk_path), h, replication=REPLICATION)
            except Exception as e:
                LOG.warning("distribute_chunk failed for %s: %s — falling back to local copy", h, e)
                # fallback: copy chunk to local fallback folder
                fallback_path = FALLBACK_CHUNKS / chunk_fname
                with chunk_path.open("rb") as src, fallback_path.open("wb") as dst:
                    dst.write(src.read())
                assignments = [{"status": "fail"}]
            chunk_records.append({"index": idx, "hash": h, "assignments": assignments})
        else:
            # legacy single-machine behavior
            node_assigned = send_chunk_to_peer_legacy(h, chunk_path)
            chunk_records.append({"index": idx, "hash": h, "node_id": node_assigned})

    # 4) prepare metadata and append to chain
    metadata = {
        "filename": filename,
        "stored_name": saved_name,
        "size": saved_path.stat().st_size,
        "chunk_count": len(chunk_hashes),
        "chunk_size": CHUNK_SIZE,
        "chunk_hashes": chunk_hashes,
        "chunks": chunk_records,
        "merkle_root": merkle_root,
        "timestamp": ts,
    }

    block_entry = chain.add_block([metadata])
    return jsonify({"status": "ok", "metadata": metadata, "block": block_entry})


@app.route("/chain", methods=["GET"])
def get_chain():
    return jsonify(chain.chain)


@app.route("/files/<path:stored_name>", methods=["GET"])
def serve_file(stored_name):
    p = STORAGE_DIR / stored_name
    if not p.exists() or not p.is_file():
        abort(404)
    return send_from_directory(str(STORAGE_DIR), stored_name, as_attachment=True)


@app.route("/chunks/<path:stored_name>", methods=["GET"])
def list_chunks(stored_name):
    chunks_dir = STORAGE_DIR / f"{stored_name}_chunks"
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        return jsonify({"error": "no chunks found for that stored_name"}), 404

    files = sorted(chunks_dir.glob("chunk_*_*.bin"))
    result = []
    for p in files:
        parts = p.name.split("_")
        if len(parts) >= 3:
            idx = parts[1]
            h = parts[2].split(".")[0]
        else:
            idx = None
            h = None
        result.append({"file": p.name, "index": idx, "hash": h, "size": p.stat().st_size})
    return jsonify(result)


@app.route("/download/<path:stored_name>", methods=["GET"])
def download_stored_file(stored_name):
    """
    Reassemble the file by fetching each chunk from the peer recorded in the chain.
    Falls back to local chunk files if peer retrieval fails.
    Streams the assembled bytes to the client.
    """
    # Find block metadata for stored_name
    block_entry_meta = None
    for entry in chain.chain:
        for meta in entry.get("metadata", []):
            if meta.get("stored_name") == stored_name:
                block_entry_meta = meta
                break
        if block_entry_meta:
            break

    if not block_entry_meta:
        return abort(404)

    # Determine chunks list and the local chunks_dir
    chunks_dir = STORAGE_DIR / f"{stored_name}_chunks"
    if "chunks" in block_entry_meta:
        chunk_records = sorted(block_entry_meta["chunks"], key=lambda r: r["index"])
    else:
        chunk_records = []
        for idx, h in enumerate(block_entry_meta.get("chunk_hashes", [])):
            chunk_records.append({"index": idx, "hash": h, "node_id": None})

    expected_merkle = block_entry_meta.get("merkle_root")

    # quick pre-check: confirm every chunk has at least one plausible source
    missing = []
    for rec in chunk_records:
        if not chunk_has_source(rec, chunks_dir):
            missing.append(rec["index"])
    if missing:
        LOG.error("Download pre-check failed: missing chunks %s for %s", missing, stored_name)
        return jsonify({"error": "missing_chunks", "missing_indices": missing}), 502

    # optional pre-check verification (recompute merkle from metadata)
    if expected_merkle:
        hashes_for_root = block_entry_meta.get("chunk_hashes") or [r["hash"] for r in chunk_records]
        recomputed = compute_merkle_root(hashes_for_root)
        if recomputed != expected_merkle:
            LOG.error("Merkle root mismatch for %s: expected %s got %s", stored_name, expected_merkle, recomputed)
            return jsonify({"error": "merkle root mismatch", "expected": expected_merkle, "got": recomputed}), 502

    def generate():
        for rec in chunk_records:
            try:
                chunk_bytes = fetch_chunk_bytes(rec, chunks_dir)
            except FileNotFoundError as e:
                LOG.error("Missing chunk during download (unexpected): %s", e)
                # stop streaming; raise to cause a 500 from Flask
                raise
            yield chunk_bytes

    filename = block_entry_meta.get("filename", stored_name)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(generate(), mimetype="application/octet-stream", headers=headers)


# ---------- Run ----------
if __name__ == "__main__":
    LOG.info("Starting backend on http://0.0.0.0:5000  DISCOVERY=%s  REPLICATION=%s", DISCOVERY_URL or "(none)", REPLICATION)
    app.run(host="0.0.0.0", port=5000, debug=True)
