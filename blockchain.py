import json
import hashlib
import time
from typing import List, Optional, Callable


class Block:
    def __init__(
        self,
        index: int,
        prev_hash: str,
        metadata_list: List[dict],
        signer: Optional[str] = None,
        signature: Optional[str] = None,
        timestamp: Optional[int] = None,
    ):
        self.index = index
        self.prev_hash = prev_hash
        self.metadata_list = metadata_list
        self.timestamp = timestamp or int(time.time())
        self.signer = signer
        self.signature = signature

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "metadata": self.metadata_list,
            "timestamp": self.timestamp,
            "signer": self.signer,
            "signature": self.signature,
        }

    def compute_hash(self) -> str:
        # deterministic JSON serialization for hashing
        payload = json.dumps(
            {
                "index": self.index,
                "prev_hash": self.prev_hash,
                "metadata": self.metadata_list,
                "timestamp": self.timestamp,
                "signer": self.signer,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class SimpleChain:
    def __init__(self, path: str = "chain.json"):
        self.path = path
        try:
            with open(self.path, "r") as f:
                self.chain = json.load(f)
        except Exception:
            self.chain = []

    def persist(self):
        with open(self.path, "w") as f:
            json.dump(self.chain, f, indent=2)

    def get_last_hash(self) -> str:
        if not self.chain:
            return "0" * 64
        return self.chain[-1]["hash"]

    def add_block(
        self,
        metadata_list: List[dict],
        signer_id: Optional[str] = None,
        signer_sign_func: Optional[Callable[[str], str]] = None,
    ) -> dict:
        idx = len(self.chain)
        prev = self.get_last_hash()

        block = Block(
            index=idx,
            prev_hash=prev,
            metadata_list=metadata_list,
            signer=signer_id,
        )

        block_hash = block.compute_hash()
        sig = None
        if signer_sign_func:
            sig = signer_sign_func(block_hash)
        block.signature = sig

        entry = block.to_dict()
        entry["hash"] = block_hash

        self.chain.append(entry)
        self.persist()
        return entry
