# chunker.py
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Generator

CHUNK_SIZE = 256 * 1024  # 256 KiB default


def iter_file_chunks(file_path: Path, chunk_size: int = CHUNK_SIZE) -> Generator[bytes, None, None]:
    """Yield raw bytes for each chunk of the file."""
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def chunk_file_and_save(file_path: Path, out_dir: Path, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Read file at `file_path`, write chunk files to `out_dir` and return list of chunk hashes (hex).
    Chunk filenames: chunk_<index>_<hash>.bin
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: List[str] = []
    for idx, chunk in enumerate(iter_file_chunks(file_path, chunk_size)):
        h = sha256_bytes(chunk)
        fname = out_dir / f"chunk_{idx:06d}_{h}.bin"
        with open(fname, "wb") as fh:
            fh.write(chunk)
        hashes.append(h)
    return hashes


def compute_merkle_root(hex_hashes: List[str]) -> str:
    """
    Compute Merkle root from a list of chunk hex-hashes (SHA256).
    - Treat each hex_hash as the leaf value (bytes.fromhex)
    - Pairwise combine: SHA256(left || right)
    - If odd count, duplicate the last leaf
    - Returns hex string of resulting root (lowercase)
    """
    if not hex_hashes:
        # empty file: define merkle root as hash of empty string
        return hashlib.sha256(b"").hexdigest()

    layer = [bytes.fromhex(h) for h in hex_hashes]

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            # duplicate last
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            combined = layer[i] + layer[i + 1]
            next_layer.append(hashlib.sha256(combined).digest())
        layer = next_layer

    return layer[0].hex()


def build_merkle_tree(hex_hashes: List[str]) -> List[List[str]]:
    """
    Build and return the full merkle tree as a list of layers (bottom -> top).
    Each element is a list of hex strings for that layer.
    Useful for proofs/debugging.
    """
    if not hex_hashes:
        return [[hashlib.sha256(b"").hexdigest()]]
    layers: List[List[str]] = []
    current = hex_hashes.copy()
    layers.append(current.copy())
    while len(current) > 1:
        if len(current) % 2 == 1:
            current.append(current[-1])
        next_layer: List[str] = []
        for i in range(0, len(current), 2):
            left = bytes.fromhex(current[i])
            right = bytes.fromhex(current[i + 1])
            next_layer.append(hashlib.sha256(left + right).hexdigest())
        current = next_layer
        layers.append(current.copy())
    return layers


def verify_merkle_root_from_chunks(chunk_bytes_list: Iterable[bytes], expected_root_hex: str, chunk_size: Optional[int] = None) -> bool:
    """
    Compute hashes from chunk bytes and verify the merkle root equals expected_root_hex.
    Accepts an iterable of raw chunk bytes (in order).
    """
    hex_hashes = [sha256_bytes(b) for b in chunk_bytes_list]
    calc_root = compute_merkle_root(hex_hashes)
    return calc_root == expected_root_hex


def reassemble_chunks(chunks_dir: Path, out_path: Path, expected_hash: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Reassemble chunk files (must be present in `chunks_dir`) into `out_path`.
    - Assumes chunk files are named `chunk_<index>_<hash>.bin`. Sorts by index.
    - If expected_hash provided, will compute SHA256 of reassembled file and verify.
    Returns (success, sha256_hex_or_None)
    """
    files = sorted(chunks_dir.glob("chunk_*_*.bin"))
    with open(out_path, "wb") as out_f:
        for p in files:
            with open(p, "rb") as f:
                out_f.write(f.read())

    if expected_hash:
        h = sha256_of_file(out_path)
        return (h == expected_hash, h)
    else:
        return (True, sha256_of_file(out_path))


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            h.update(data)
    return h.hexdigest()
