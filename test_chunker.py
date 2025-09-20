# test_chunker.py
from pathlib import Path
from chunker import chunk_file_and_save, compute_merkle_root, sha256_of_file, reassemble_chunks

# create a sample file
ROOT = Path(__file__).parent
sample = ROOT / "sample.bin"
with open(sample, "wb") as f:
    f.write(b"Hello world!\n" * 10000)  # ~120KB

out_dir = ROOT / "sample_chunks"
if out_dir.exists():
    import shutil
    shutil.rmtree(out_dir)

chunk_hashes = chunk_file_and_save(sample, out_dir, chunk_size=1024)
print("chunks:", len(chunk_hashes))

root = compute_merkle_root(chunk_hashes)
print("merkle root:", root)

# reassemble
reassembled = ROOT / "sample_reassembled.bin"
ok, sha = reassemble_chunks(out_dir, reassembled)
print("reassembled ok:", ok, "sha:", sha)
print("original sha:", sha256_of_file(sample))
