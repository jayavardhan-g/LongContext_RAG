"""Model path resolution.

The HF cache on this box lives on NFS (`/mnt/warm_storage`), and transformers
materializes checkpoint tensors through an mmap. Sequential reads off that mount
run at ~80 MB/s, but mmap page-faults do not: loading bge-m3 from the cache did
not finish in 280 s, while the same checkpoint on local disk loads in 30 s.
Llama pays the same tax on every run.

So models are mirrored to local disk and looked up there first, falling back to
the plain HF id (and therefore the NFS cache) when no mirror exists.

To add a mirror:
    cp -rL <hf_cache_snapshot>/. /data/jayavardhan/lcrag_models/<model-name>/
"""

from pathlib import Path

LOCAL_MODEL_DIR = Path("/data/jayavardhan/lcrag_models")


def resolve_model(name: str) -> str:
    """Return a local mirror of `name` if one exists, else `name` unchanged."""
    local = LOCAL_MODEL_DIR / name.split("/")[-1]
    return str(local) if (local / "config.json").is_file() else name
