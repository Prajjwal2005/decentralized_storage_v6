# peers/node_server.py
"""
Peer node server.
Runs a small Flask app that accepts /store and /retrieve.
On startup it registers itself with the discovery service and heartbeats periodically.
"""
import os
import time
import threading
import argparse
import requests
from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)


def make_storage_dir(base_dir):
    storage = os.path.join(base_dir, "node_data")
    os.makedirs(storage, exist_ok=True)
    return storage


@app.route("/store", methods=["POST"])
def store():
    file_hash = request.form.get("file_hash")
    f = request.files.get("file")
    if not file_hash or not f:
        return jsonify({"error": "missing file_hash or file"}), 400
    save_path = os.path.join(STORAGE_DIR, file_hash)
    try:
        f.save(save_path)
    except Exception as e:
        app.logger.exception("Failed to save chunk")
        return jsonify({"error": "failed to save", "detail": str(e)}), 500
    app.logger.info("Stored chunk %s -> %s", file_hash, save_path)
    return jsonify({"status": "stored", "path": save_path, "node_id": NODE_ID}), 200


@app.route("/retrieve/<file_hash>", methods=["GET"])
def retrieve(file_hash):
    path = os.path.join(STORAGE_DIR, file_hash)
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, as_attachment=True)


def announce_loop(discovery_url, node_id, ip, port, meta=None, interval=15):
    """Register once and heartbeat periodically."""
    reg_url = discovery_url.rstrip("/") + "/register"
    hb_url = discovery_url.rstrip("/") + "/heartbeat"
    payload = {"node_id": node_id, "ip": ip, "port": port, "meta": meta or {}}
    try:
        r = requests.post(reg_url, json=payload, timeout=5)
        app.logger.info("Register response: %s %s", r.status_code, r.text)
    except Exception as e:
        app.logger.warning("Register failed: %s", e)
    while True:
        try:
            requests.post(hb_url, json={"node_id": node_id, "ip": ip, "port": port}, timeout=3)
        except Exception:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--discovery", required=True, help="http://DISCOVERY_HOST:4000")
    parser.add_argument("--dir", default=".")
    parser.add_argument("--public-ip", default="127.0.0.1", help="public reachable ip or hostname")
    args = parser.parse_args()

    NODE_ID = args.node_id
    PORT = args.port
    DISCOVERY_URL = args.discovery
    BASE_DIR = args.dir
    STORAGE_DIR = make_storage_dir(BASE_DIR)
    PUBLIC_IP = args.public_ip

    t = threading.Thread(target=announce_loop, args=(DISCOVERY_URL, NODE_ID, PUBLIC_IP, PORT, {"capacity": "unknown"}), daemon=True)
    t.start()

    print(f"Starting node {NODE_ID} on port {PORT}, storage: {STORAGE_DIR}, discovery={DISCOVERY_URL}")
    app.run(host="0.0.0.0", port=PORT)
