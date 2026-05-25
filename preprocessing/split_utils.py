"""Shared utilities for character-level 3-way split across datasets.

Provides ``compute_character_embeddings`` and ``three_way_split`` so that
all four dataset preprocessing scripts use a single, consistent
implementation.

The split strategy: profile embeddings are clustered via K-Means, then
outlier (small + isolated) clusters form the OOD test set, diverse
sampling from the remainder forms the random test set, and everything
else becomes the training set.
"""

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42


def compute_character_embeddings(
    characters: list[str],
    profile_texts: dict[str, str],
    env_path: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Call the embedding API for each character's profile text.

    Parameters
    ----------
    characters : list[str]
        Character names to embed.
    profile_texts : dict[str, str]
        Mapping from character name to profile text.
    env_path : Path, optional
        Path to ``.env`` file.  If *None*, ``load_dotenv()`` is called
        without an explicit path (searches parent directories).
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    client = OpenAI(
        api_key=os.getenv("EMBED_API_KEY"),
        base_url=os.getenv("EMBED_BASE_URL"),
    )
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    texts, valid_chars = [], []
    for ch in characters:
        t = profile_texts.get(ch, "")
        if t:
            texts.append(t)
            valid_chars.append(ch)

    print(f"Computing embeddings for {len(valid_chars)} characters "
          f"(model={model}) ...")
    resp = client.embeddings.create(input=texts, model=model)
    emb = np.array([d.embedding for d in resp.data])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return valid_chars, emb


def three_way_split(
    characters: list[str],
    embeddings: np.ndarray,
    n_random_test: int,
    n_ood_test: int,
    max_k: int = 15,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Cluster-based 3-way split (train / random_test / ood_test).

    Parameters
    ----------
    characters : list[str]
        Character names (same order as *embeddings* rows).
    embeddings : np.ndarray
        Shape ``(n_characters, dim)`` — L2-normalised embeddings.
    n_random_test, n_ood_test : int
        Target sizes for random-test and OOD-test splits.
    max_k : int
        Upper bound for the K-Means k search (default 15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_chars, random_test_chars, ood_test_chars)
    """
    random.seed(seed)
    np.random.seed(seed)
    n_total = len(characters)

    best_k, best_score = 3, -1.0
    for k in range(3, min(max_k, n_total)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lab = km.fit_predict(embeddings)
        sc = silhouette_score(embeddings, lab)
        if sc > best_score:
            best_score, best_k = sc, k
    print(f"Optimal k={best_k}, silhouette={best_score:.3f}")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    centers = km.cluster_centers_

    cluster_info = []
    for cid in range(best_k):
        idx = np.where(labels == cid)[0]
        inter = [np.linalg.norm(centers[cid] - centers[j])
                 for j in range(best_k) if j != cid]
        avg_inter = float(np.mean(inter)) if inter else 0.0
        score = avg_inter / (1 + len(idx))
        cluster_info.append(dict(cid=cid, idx=idx, size=len(idx),
                                 avg_inter=avg_inter, score=score))
        print(f"  Cluster {cid}: {len(idx):>2} chars, "
              f"inter_dist={avg_inter:.3f}, score={score:.3f}")
    cluster_info.sort(key=lambda x: x["score"], reverse=True)

    # --- OOD: top-scoring clusters ---
    ood_indices: list[int] = []
    for info in cluster_info:
        if len(ood_indices) >= n_ood_test:
            break
        remaining = n_ood_test - len(ood_indices)
        if info["size"] <= remaining:
            ood_indices.extend(info["idx"].tolist())
        else:
            dists = np.linalg.norm(
                embeddings[info["idx"]] - centers[info["cid"]], axis=1)
            furthest = np.argsort(dists)[-remaining:]
            ood_indices.extend(info["idx"][furthest].tolist())
    ood_set = set(ood_indices[:n_ood_test])

    # --- Random test: diverse sampling from remaining ---
    remaining_idx = [i for i in range(n_total) if i not in ood_set]
    rem_emb = embeddings[remaining_idx]
    n_rc = min(5, len(remaining_idx))
    if len(remaining_idx) > n_random_test and n_rc >= 2:
        rk = KMeans(n_clusters=n_rc, random_state=seed, n_init=10)
        rl = rk.fit_predict(rem_emb)
        counts = np.bincount(rl, minlength=n_rc)
        alloc = np.round(
            counts / len(remaining_idx) * n_random_test).astype(int)
        diff = n_random_test - int(alloc.sum())
        for ix in np.argsort(counts)[::-1]:
            if diff <= 0:
                break
            alloc[ix] += 1
            diff -= 1
        rt_local: list[int] = []
        for cid in range(n_rc):
            ci = np.where(rl == cid)[0]
            take = min(int(alloc[cid]), len(ci))
            if take > 0:
                chosen = np.random.choice(ci, size=take, replace=False)
                rt_local.extend(chosen.tolist())
    else:
        rt_local = list(range(min(n_random_test, len(remaining_idx))))
    rt_global = set(remaining_idx[i] for i in rt_local[:n_random_test])

    # --- Train: everything else ---
    train_idx = [i for i in range(n_total)
                 if i not in ood_set and i not in rt_global]

    return (
        [characters[i] for i in train_idx],
        [characters[i] for i in sorted(rt_global)],
        [characters[i] for i in sorted(ood_set)],
    )
