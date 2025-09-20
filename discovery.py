# discovery/discovery.py
"""
Simple discovery / bootstrap registry.
Peers register with POST /register and heartbeat with POST /heartbeat.
Clients (backends) fetch current live peers via GET /peers.
"""
import time
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

PEERS = {}  # node_id -> { node_id, ip, port, meta, last_heartbeat }
LOCK = threading.Lock()
TTL = 60  # seconds - mark dead if no heartbeat within TTL


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    node_id = data.get("node_id")
    ip = data.get("ip")
    port = data.get("port")
    meta = data.get("meta", {})
    if not node_id or not ip or not port:
        return jsonify({"error": "node_id, ip, port required"}), 400
    with LOCK:
        PEERS[node_id] = {
            "node_id": node_id,
            "ip": ip,
            "port": int(port),
            "meta": meta,
            "last_heartbeat": time.time(),
        }
    return jsonify({"status": "registered"})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json(force=True)
    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    with LOCK:
        if node_id in PEERS:
            PEERS[node_id]["last_heartbeat"] = time.time()
            # update optional fields
            if "ip" in data:
                PEERS[node_id]["ip"] = data.get("ip")
            if "port" in data:
                PEERS[node_id]["port"] = int(data.get("port"))
            if "meta" in data:
                PEERS[node_id]["meta"] = data.get("meta")
            return jsonify({"status": "ok"})
    return jsonify({"error": "not registered"}), 404


@app.route("/peers", methods=["GET"])
def peers():
    """Return currently alive peers (optionally limit parameter)."""
    limit = int(request.args.get("limit", "50"))
    with LOCK:
        now = time.time()
        alive = [p for p in PEERS.values() if (now - p["last_heartbeat"]) <= TTL]
        return jsonify({"peers": alive[:limit]})


@app.route("/peer/<node_id>", methods=["GET"])
def peer(node_id):
    with LOCK:
        p = PEERS.get(node_id)
        if not p:
            return jsonify({"error": "not found"}), 404
        return jsonify(p)


def reaper():
    while True:
        time.sleep(max(5, TTL // 3))
        with LOCK:
            now = time.time()
            dead = [pid for pid, p in PEERS.items() if (now - p["last_heartbeat"]) > TTL]
            for pid in dead:
                print("[discovery] reaping dead peer", pid)
                del PEERS[pid]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=4000)
    args = p.parse_args()
    t = threading.Thread(target=reaper, daemon=True)
    t.start()
    print(f"Starting discovery on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
