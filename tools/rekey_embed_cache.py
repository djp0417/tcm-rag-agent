# -*- coding: utf-8 -*-
"""把 `store/embed_cache.npz` 的键迁移成「块id:内容指纹」格式（零 API）。

背景
----
原缓存键是 `{sha1(文件名)}-{序号}`（只绑位置、不绑内容），已在
`app/index.py::_cache_key` 修成 `{块id}:{内容指纹}`。旧键在库里
既永远命中不了、又占着几十 MB，必须换成新的格式。

做法（为什么不重跑建库）
------------------------
向量本来就存在 Chroma 里，`collection.get(include=["documents","embeddings"])`
可以直接把它们连同最新文字一起读出来。于是：
    新键 = f"{块id}:{sha1(当前库里的文字)[:12]}"
一次读库 + 一次写盘即可完成，**0 次 embedding API**，也不会因为
"老键命中不了"而触发全量重算（那要重嵌入一万两千多块）。

用法：
    python -m tools.rekey_embed_cache            # 就地迁移（先自动备份）
    python -m tools.rekey_embed_cache --dry-run  # 只看会迁移多少条
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import EMBED_CACHE_PATH, COLLECTION, _load_vector_cache   # noqa: E402
from app.paths import chroma_store_path                                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="迁移向量缓存键为「块id:内容指纹」")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    args = ap.parse_args()

    t0 = time.time()
    old = _load_vector_cache()
    legacy = [k for k in old if ":" not in k]
    print(f"[read ] 旧缓存 {len(old)} 条，其中旧格式 {len(legacy)} 条")

    import chromadb
    import numpy as np

    client = chromadb.PersistentClient(path=chroma_store_path())
    col = client.get_collection(COLLECTION)
    got = col.get(include=["documents", "embeddings"])
    ids = got["ids"]
    docs = got["documents"] or []
    embs = got["embeddings"]
    print(f"[db   ] 集合内 {len(ids)} 块，已读出文字与向量")

    # 以**库里的现状**为准重建：文字与向量都取自同一时刻，天然一致
    cache: dict[str, "np.ndarray"] = {}
    for cid, doc, vec in zip(ids, docs, embs):
        key = f"{cid}:{hashlib.sha1((doc or '').encode('utf-8')).hexdigest()[:12]}"
        cache[key] = np.asarray(vec, dtype=np.float32)

    if args.dry_run:
        print(f"[dry  ] 将写入 {len(cache)} 条新格式键（未落盘）")
        return

    if EMBED_CACHE_PATH.exists():
        backup = EMBED_CACHE_PATH.with_suffix(".npz.bak")
        shutil.copy2(EMBED_CACHE_PATH, backup)
        print(f"[bak  ] 旧缓存已备份 → {backup.name}")

    keys = list(cache)
    np.savez_compressed(EMBED_CACHE_PATH,
                        ids=np.array(keys),
                        vectors=np.stack([cache[k] for k in keys]))
    size_mb = EMBED_CACHE_PATH.stat().st_size / 1024 / 1024
    print(f"[ok   ] 已写入 {len(keys)} 条（新格式），{size_mb:.1f} MB "
          f"→ {EMBED_CACHE_PATH.name}")
    print(f"[done ] 耗时 {time.time() - t0:.1f}s，未调用任何 embedding API")


if __name__ == "__main__":
    main()
