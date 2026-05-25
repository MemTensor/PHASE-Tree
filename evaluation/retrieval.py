"""Embedding-based retrieval pool for RAG / PAG baselines.

Loads a dialogue corpus (typically the dataset's ``train`` split), encodes
each ``(role, context)`` pair with an OpenAI-compatible embedding endpoint
(by default a local vLLM-served Qwen3-Embedding-4B), caches
the L2-normalized embedding matrix on disk, and supports top-K cosine
retrieval at query time.

Design choices:

* **Pool source = train split.**  Pool and queries come from disjoint
  ``question_id`` sets, so a query never retrieves *itself*.  However,
  utterance-level random splits (used by all 8 datasets here) routinely
  put different utterances of the *same scene* into different splits;
  retrieving such a "later" pool sample exposes the query's ground-truth
  output as part of that sample's ``input`` context — a subtle but real
  leakage path.  See :meth:`RetrievalPool.scene_qids_for` and the
  ``scene_window`` constructor argument for the mitigation.
* **Global pool, character-aware encoding.**  All characters in a
  dataset share one pool.  We prepend the character name when encoding
  so that intra-character matches are softly preferred but cross-
  character retrieval is not forbidden.
* **Self-id filter at query time.**  ``exclude_qids`` (defaults to the
  query's own ``question_id``) prevents trivial leakage.
* **Scene-level filter (optional).**  When ``scene_window > 0`` the pool
  builds a ``scene_id → [qid, ...]`` index keyed on a hash of each
  sample's first ``scene_window`` context lines.  Callers can then ask
  the pool for the qids that share a scene with the current query and
  union them into ``exclude_qids`` to block the same-scene leakage path.
* **Cache invalidation by content hash.**  The cache key is a SHA1 over
  ``(embed_model, [question_id, ...])`` so changing the pool or the
  model triggers a rebuild automatically.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Embedding helpers: API-based or local sentence-transformers
# ---------------------------------------------------------------------------

def _encode_batch(client, model: str, texts: list[str],
                  max_retries: int = 3) -> list[list[float]]:
    """Encode a batch of texts.

    *client* is either an ``openai.OpenAI`` instance (API mode) or a
    :class:`LocalEmbedClient` wrapper (local mode).  Both expose
    ``embeddings.create(model=..., input=...)``.
    """
    clean = [t if (t and t.strip()) else "[EMPTY]" for t in texts]
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=clean, timeout=120)
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"embedding API failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Local in-process embedding client (drop-in replacement for openai.OpenAI)
# ---------------------------------------------------------------------------

class _EmbObj:
    """Mimics ``openai.types.Embedding``."""
    def __init__(self, embedding: list[float], index: int):
        self.embedding = embedding
        self.index = index

class _EmbResp:
    """Mimics ``openai.types.CreateEmbeddingResponse``."""
    def __init__(self, data: list[_EmbObj]):
        self.data = data

class _EmbNamespace:
    """Mimics ``client.embeddings`` with a ``.create()`` method.

    Holds a reference to the parent :class:`LocalEmbedClient` so we can
    dispatch to either single-GPU ``encode`` or multi-GPU
    ``encode_multi_process`` depending on how the client was configured.
    """
    def __init__(self, parent: "LocalEmbedClient"):
        self._parent = parent

    def create(self, *, model: str = "", input: list[str] | str = "",
               **kwargs) -> _EmbResp:
        texts = [input] if isinstance(input, str) else list(input)
        embs = self._parent._encode_texts(texts)
        return _EmbResp([_EmbObj(embs[i].tolist(), i)
                         for i in range(len(texts))])

class LocalEmbedClient:
    """Wraps a ``SentenceTransformer`` model so it can be used wherever
    ``retrieval.py`` expects an OpenAI-compatible embedding client.

    Two modes:

    1. **Single-GPU** (default): the model is loaded once on the requested
       ``device`` and ``encode_multi_process`` is *not* used.  This is the
       legacy behavior and the safest default for short-context runs.

    2. **Multi-GPU**: pass ``devices=["cuda:0", "cuda:1", ...]`` (with
       ``len(devices) > 1``) to spawn one worker process per device via
       :meth:`SentenceTransformer.start_multi_process_pool`.  Each worker
       loads a copy of the model so the activation memory of a single
       forward pass is divided across cards -- this is the right knob to
       turn when long-context queries OOM a single card without lowering
       the model's ``max_seq_length``.

    The ``st_batch_size`` parameter controls the *internal* mini-batch
    SentenceTransformer uses inside one ``encode`` call.  Long
    dialogue-context texts have very large attention/activation tensors
    so a small internal batch (e.g. 4 or 8) is essential; the previous
    default of 32 is what caused the 72 GB-per-process OOMs we saw.

    Usage::

        client = LocalEmbedClient(
            "models/Qwen3-Embedding-4B",
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            st_batch_size=8,
        )
        pool = RetrievalPool(samples, embed_client=client, ...)
        pool.build_or_load()
        client.unload()
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        devices: list[str] | None = None,
        st_batch_size: int = 8,
    ):
        from sentence_transformers import SentenceTransformer
        import torch
        # Resolve device list: explicit `devices` wins, otherwise fall back
        # to the legacy single-`device` argument.
        if devices is None or len(devices) == 0:
            devices = [device] if device else ["cuda"]
        self._devices = list(devices)
        self._multi_gpu = len(self._devices) > 1
        self._st_batch_size = max(1, int(st_batch_size))

        print(
            f"  Loading local embedding model: {model_path}\n"
            f"    devices={self._devices}, "
            f"multi_gpu={self._multi_gpu}, "
            f"st_batch_size={self._st_batch_size}",
            flush=True,
        )
        t0 = time.time()
        self._st = SentenceTransformer(model_path, trust_remote_code=True)
        # The "main" copy of the model lives on the first device.  In multi-
        # GPU mode the worker processes will load their own copies on the
        # other devices when we start the pool below.
        primary_dev = self._devices[0]
        if torch.cuda.is_available() and primary_dev != "cpu":
            self._st = self._st.to(primary_dev)

        self._mp_pool = None
        if self._multi_gpu:
            print(
                f"    starting multi-process pool on {len(self._devices)} GPUs ...",
                flush=True,
            )
            # encode_multi_process spawns one process per target device.
            self._mp_pool = self._st.start_multi_process_pool(
                target_devices=self._devices)

        self.embeddings = _EmbNamespace(self)
        dim = (self._st.get_embedding_dimension()
               if hasattr(self._st, "get_embedding_dimension")
               else self._st.get_sentence_embedding_dimension())
        print(f"  Embedding model loaded in {time.time() - t0:.1f}s "
              f"(dim={dim})", flush=True)

    # --------------------------------------------------------------
    # Internal: encode a list of texts using the configured backend.
    # --------------------------------------------------------------
    def _encode_texts(self, texts: list[str]):
        if self._multi_gpu and self._mp_pool is not None:
            # encode_multi_process distributes batches across all worker
            # processes, then collates the results.  Use a small per-worker
            # batch_size so long-context attention tensors stay bounded.
            return self._st.encode_multi_process(
                texts,
                pool=self._mp_pool,
                batch_size=self._st_batch_size,
                normalize_embeddings=True,
            )
        return self._st.encode(
            texts,
            batch_size=self._st_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def unload(self):
        """Delete the model and free GPU memory."""
        import gc, torch
        if self._mp_pool is not None:
            try:
                self._st.stop_multi_process_pool(self._mp_pool)
            except Exception as e:
                print(f"  [warn] stop_multi_process_pool failed: {e}",
                      flush=True)
            self._mp_pool = None
        del self._st
        self.embeddings = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Embedding model unloaded, GPU memory freed.", flush=True)


# ---------------------------------------------------------------------------
# Pool object
# ---------------------------------------------------------------------------

class RetrievalPool:
    """Global retrieval pool with disk-cached normalized embeddings.

    Typical lifecycle::

        pool = RetrievalPool(samples, embed_client, embed_model,
                             cache_path="phase_tree_data/processed/RAIDEN/_retrieval_cache/...npz")
        pool.build_or_load()
        for query in test_samples:
            q_emb = pool.encode_query(query)
            hits = pool.query_top_k(q_emb, k=5,
                                    exclude_qids={query["question_id"]})
    """

    def __init__(self,
                 samples: list[dict],
                 embed_client,
                 embed_model: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 cache_path: str | None = None,
                 scene_window: int = 0):
        """Create a retrieval pool.

        Parameters
        ----------
        scene_window : int, default 0
            If > 0, build a same-scene index keyed on the SHA1 of the
            first ``scene_window`` non-empty context lines of each
            sample.  Used by :meth:`scene_qids_for` to filter out
            same-scene neighbors at query time, which prevents the
            "ground truth appears in retrieved demonstration's context"
            leakage caused by utterance-level dataset splits.  Set to 0
            to disable (legacy behavior).
        """
        self.samples = samples
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_path = cache_path
        self.scene_window = max(0, int(scene_window))
        self.embeddings: np.ndarray | None = None
        self.qid_to_idx: dict[str, int] = {
            s["question_id"]: i for i, s in enumerate(samples)
        }
        # Same-scene index: scene_fingerprint -> list of pool qids.  Built
        # lazily inside ``build_or_load`` (cheap; pure string hashing).
        self._scene_to_qids: dict[str, list[str]] | None = None

    # ---- Cache key (content-addressed) -----------------------------------

    def _cache_key(self) -> str:
        h = hashlib.sha1()
        h.update(self.embed_model.encode("utf-8"))
        for s in self.samples:
            h.update(s["question_id"].encode("utf-8"))
        return h.hexdigest()[:16]

    # ---- Text formatting -------------------------------------------------

    @staticmethod
    def _format_for_embedding(s: dict) -> str:
        """Format a pool/query sample as a single text for embedding.

        Includes the speaker name so character-aware retrieval is softly
        encouraged without being strictly enforced.
        """
        role = (s.get("role") or "").strip()
        ctx = (s.get("input") or "").strip()
        return f"Character: {role}\nContext: {ctx}"

    # ---- Scene-fingerprint helpers --------------------------------------

    @staticmethod
    def _scene_fingerprint(sample: dict, window: int) -> str | None:
        """Hash the first ``window`` non-empty context lines.

        Two samples that share a scene must, by construction, share the
        same opening dialogue lines (the cumulative context grows
        monotonically as later utterances are sampled).  Hashing those
        lines therefore yields a robust scene identifier even though the
        per-sample ``_scene_id`` field has been stripped from the
        processed JSON files.

        Returns ``None`` when there is nothing to hash (empty input or
        ``window <= 0``).
        """
        if window <= 0:
            return None
        ctx = (sample.get("input") or "").strip()
        if not ctx:
            return None
        lines = [ln for ln in (l.strip() for l in ctx.split("\n")) if ln]
        if not lines:
            return None
        head = lines[:window]
        return hashlib.sha1("\n".join(head).encode("utf-8")).hexdigest()[:16]

    def _build_scene_index(self) -> None:
        """Populate ``self._scene_to_qids`` from current ``self.samples``."""
        if self.scene_window <= 0:
            self._scene_to_qids = None
            return
        index: dict[str, list[str]] = {}
        for s in self.samples:
            fp = self._scene_fingerprint(s, self.scene_window)
            if fp is None:
                continue
            index.setdefault(fp, []).append(s["question_id"])
        self._scene_to_qids = index
        n_groups = len(index)
        n_indexed = sum(len(v) for v in index.values())
        print(f"  Retrieval pool: built scene index "
              f"(window={self.scene_window}, {n_groups} scenes, "
              f"{n_indexed} indexed samples)", flush=True)

    def scene_qids_for(self, sample: dict) -> set[str]:
        """Return all pool ``question_id``s that share a scene with *sample*.

        Empty when scene filtering is disabled or the sample has no
        usable input.  Callers should union the returned set into
        ``exclude_qids`` when calling :meth:`query_top_k`.
        """
        if self.scene_window <= 0 or self._scene_to_qids is None:
            return set()
        fp = self._scene_fingerprint(sample, self.scene_window)
        if fp is None:
            return set()
        return set(self._scene_to_qids.get(fp, ()))

    # ---- Build / load ----------------------------------------------------

    def build_or_load(self) -> None:
        """Populate ``self.embeddings``.  Loads from cache when valid.

        The scene index (if requested via ``scene_window > 0``) is always
        rebuilt in-memory regardless of whether the embeddings come from
        cache; it is cheap (pure string hashing) and keeping it out of
        the ``.npz`` cache avoids invalidating existing caches.
        """
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with np.load(self.cache_path, allow_pickle=False) as data:
                    cached_key = str(data["cache_key"])
                    if cached_key == self._cache_key():
                        self.embeddings = data["embeddings"]
                        print(f"  Retrieval pool: loaded {len(self.samples)} "
                              f"cached embeddings ({self.embeddings.shape[1]}-D) "
                              f"from {self.cache_path}", flush=True)
                        self._build_scene_index()
                        return
                    print(f"  Retrieval pool: cache key mismatch, rebuilding",
                          flush=True)
            except (OSError, KeyError, ValueError) as e:
                print(f"  Retrieval pool: cache unreadable ({e}), rebuilding",
                      flush=True)

        print(f"  Retrieval pool: encoding {len(self.samples)} pool samples "
              f"with {self.embed_model} ...", flush=True)
        texts = [self._format_for_embedding(s) for s in self.samples]
        n = len(texts)
        all_embs: list[list[float] | None] = [None] * n

        def _worker(i_start: int) -> tuple[int, list[list[float]]]:
            i_end = min(i_start + self.batch_size, n)
            embs = _encode_batch(self.embed_client, self.embed_model,
                                 texts[i_start:i_end])
            return i_start, embs

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(_worker, i)
                       for i in range(0, n, self.batch_size)]
            pbar = tqdm(total=len(futures), desc="encode-pool", unit="batch")
            for f in as_completed(futures):
                i_start, embs = f.result()
                for j, e in enumerate(embs):
                    all_embs[i_start + j] = e
                pbar.update(1)
            pbar.close()

        if any(e is None for e in all_embs):
            raise RuntimeError("some pool samples failed to embed")

        embs = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = embs / norms

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez(self.cache_path,
                     embeddings=self.embeddings,
                     cache_key=np.array(self._cache_key()))
            print(f"  Saved embedding cache: {self.cache_path}", flush=True)

        self._build_scene_index()

    # ---- Query -----------------------------------------------------------

    def encode_query(self, sample: dict) -> np.ndarray:
        """Encode a single query and return its L2-normalized embedding."""
        text = self._format_for_embedding(sample)
        emb = np.array(_encode_batch(self.embed_client, self.embed_model,
                                     [text])[0], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    def encode_queries_batch(self, samples: list[dict]) -> np.ndarray:
        """Encode a list of queries in parallel; returns (N, D) normalized."""
        texts = [self._format_for_embedding(s) for s in samples]
        n = len(texts)
        all_embs: list[list[float] | None] = [None] * n

        def _worker(i_start: int) -> tuple[int, list[list[float]]]:
            i_end = min(i_start + self.batch_size, n)
            embs = _encode_batch(self.embed_client, self.embed_model,
                                 texts[i_start:i_end])
            return i_start, embs

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(_worker, i)
                       for i in range(0, n, self.batch_size)]
            pbar = tqdm(total=len(futures), desc="encode-query", unit="batch")
            for f in as_completed(futures):
                i_start, embs = f.result()
                for j, e in enumerate(embs):
                    all_embs[i_start + j] = e
                pbar.update(1)
            pbar.close()

        if any(e is None for e in all_embs):
            raise RuntimeError("some query samples failed to embed")
        embs = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def query_top_k(self,
                    query_emb: np.ndarray,
                    k: int,
                    exclude_qids: Iterable[str] | None = None,
                    ) -> list[tuple[int, float, dict]]:
        """Return up to k ``(idx, cosine_score, sample)`` triples, sorted desc."""
        if self.embeddings is None:
            raise RuntimeError("call build_or_load() first")
        scores = self.embeddings @ query_emb
        order = np.argsort(-scores)
        excl = set(exclude_qids) if exclude_qids else set()

        out: list[tuple[int, float, dict]] = []
        for idx in order:
            sample = self.samples[int(idx)]
            if sample["question_id"] in excl:
                continue
            out.append((int(idx), float(scores[int(idx)]), sample))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Demonstration formatter (for prompt injection)
# ---------------------------------------------------------------------------

def format_demonstrations(hits: list[tuple[int, float, dict]],
                          max_ctx_chars: int = 8192,
                          max_resp_chars: int = 512) -> str:
    """Render top-K retrieval hits as a numbered, human-readable block.

    Each hit shows speaker / context / response so the LLM can use both
    the conversational situation and the response style as in-context
    cues.
    """
    if not hits:
        return "(no retrieved examples)"

    def _trunc(text: str, n: int) -> str:
        text = (text or "").strip().replace("\n\n", "\n")
        return text if len(text) <= n else text[:n].rstrip() + " …"

    lines: list[str] = []
    for rank, (_, score, s) in enumerate(hits, start=1):
        role = (s.get("role") or "").strip()
        ctx = _trunc(s.get("input") or "", max_ctx_chars)
        out = _trunc(s.get("output") or "", max_resp_chars)
        lines.append(
            f"[Example {rank}] (sim={score:.3f}) Speaker: {role}\n"
            f"Context:\n{ctx}\n"
            f"Response: {out}"
        )
    return "\n\n".join(lines)
