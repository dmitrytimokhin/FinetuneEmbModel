"""
metrics.py
==========
Модуль оценки качества retrieval для сравнения оригинальной и дообученной моделей.

Метрики
────────
┌────────────────┬───────────────────────────────────────────────────────────────┐
│ Метрика        │ Интерпретация                                                 │
├────────────────┼───────────────────────────────────────────────────────────────┤
│ Recall@k       │ Доля запросов, для которых релевантный документ               │
│                │ попал в top-k (ключевая метрика для RAG)                      │
├────────────────┼───────────────────────────────────────────────────────────────┤
│ MRR@k          │ Mean Reciprocal Rank — средняя обратная позиция               │
│                │ первого релевантного результата                               │
├────────────────┼───────────────────────────────────────────────────────────────┤
│ NDCG@k         │ Normalized Discounted Cumulative Gain —                       │
│                │ учитывает позицию и степень релевантности                     │
├────────────────┼───────────────────────────────────────────────────────────────┤
│ MAP@k          │ Mean Average Precision — средняя точность по всем запросам    │
└────────────────┴───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Попытка импорта faiss (опционально)
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info("faiss не найден — используем numpy для поиска")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Результаты оценки retrieval для одной модели."""
    model_name: str
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    map_at_10: float = 0.0
    n_queries: int = 0
    n_corpus: int = 0
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "Recall@1": round(self.recall_at_1, 4),
            "Recall@3": round(self.recall_at_3, 4),
            "Recall@5": round(self.recall_at_5, 4),
            "Recall@10": round(self.recall_at_10, 4),
            "MRR@10": round(self.mrr_at_10, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
            "MAP@10": round(self.map_at_10, 4),
            "n_queries": self.n_queries,
            "n_corpus": self.n_corpus,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculations
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_k(
    rankings: List[List[str]],
    qrels: Dict[str, List[str]],
    query_ids: List[str],
    k: int,
) -> float:
    hits = 0
    for qid, ranked_docs in zip(query_ids, rankings):
        relevant = set(qrels.get(qid, []))
        if relevant & set(ranked_docs[:k]):
            hits += 1
    return hits / len(query_ids) if query_ids else 0.0


def _mrr_at_k(
    rankings: List[List[str]],
    qrels: Dict[str, List[str]],
    query_ids: List[str],
    k: int,
) -> float:
    scores = []
    for qid, ranked_docs in zip(query_ids, rankings):
        relevant = set(qrels.get(qid, []))
        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[:k], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def _ndcg_at_k(
    rankings: List[List[str]],
    qrels: Dict[str, List[str]],
    query_ids: List[str],
    k: int,
) -> float:
    """NDCG@k с бинарной релевантностью."""
    scores = []
    for qid, ranked_docs in zip(query_ids, rankings):
        relevant = set(qrels.get(qid, []))
        n = min(k, len(ranked_docs))

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[:n], start=1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_n = min(k, len(relevant))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_n + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def _map_at_k(
    rankings: List[List[str]],
    qrels: Dict[str, List[str]],
    query_ids: List[str],
    k: int,
) -> float:
    scores = []
    for qid, ranked_docs in zip(query_ids, rankings):
        relevant = set(qrels.get(qid, []))
        if not relevant:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked_docs[:k], start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / rank
        scores.append(precision_sum / min(k, len(relevant)))

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Index
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyIndex:
    """Простой cosine-similarity поиск через numpy (fallback без faiss)."""

    def __init__(self, embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._embs = embeddings / norms

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (scores, indices) топ-k."""
        q = query / (np.linalg.norm(query) + 1e-9)
        sims = self._embs @ q
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return sims[top_idx], top_idx


class _FaissIndex:
    """Cosine-similarity поиск через faiss."""

    def __init__(self, embeddings: np.ndarray) -> None:
        d = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(d)
        embs = embeddings.astype(np.float32)
        faiss.normalize_L2(embs)
        self._index.add(embs)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, k)
        return scores[0], indices[0]


def _build_index(embeddings: np.ndarray):
    if _FAISS_AVAILABLE:
        return _FaissIndex(embeddings)
    return _NumpyIndex(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Оценивает качество retrieval для одной или нескольких моделей.

    Параметры
    ---------
    queries  : Dict[query_id, query_text]
    corpus   : Dict[doc_id, doc_text]
    qrels    : Dict[query_id, List[doc_id]]  — релевантные документы для запроса

    Пример
    ------
    >>> evaluator = RetrievalEvaluator(queries, corpus, qrels)
    >>> metrics = evaluator.evaluate("deepvk/USER-bge-m3")
    >>> comparison = evaluator.compare_models({
    ...     "original": "deepvk/USER-bge-m3",
    ...     "finetuned": "./output/checkpoint-best",
    ... })
    >>> evaluator.plot_comparison(comparison)
    """

    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels: Dict[str, List[str]],
        ks: Tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.ks = ks

        self._query_ids = list(queries.keys())
        self._corpus_ids = list(corpus.keys())
        self._corpus_texts = [corpus[cid] for cid in self._corpus_ids]

        logger.info(
            "RetrievalEvaluator: %d запросов, %d документов",
            len(queries),
            len(corpus),
        )

    # ── Encode ───────────────────────────────────────────────────────────────

    def _encode_corpus(
        self, model: SentenceTransformer, batch_size: int = 64
    ) -> np.ndarray:
        logger.info("Кодирование корпуса (%d текстов)...", len(self._corpus_texts))
        return model.encode(
            self._corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _encode_queries(
        self, model: SentenceTransformer, batch_size: int = 64
    ) -> np.ndarray:
        query_texts = [self.queries[qid] for qid in self._query_ids]
        logger.info("Кодирование запросов (%d)...", len(query_texts))
        return model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    # ── Evaluate single model ────────────────────────────────────────────────

    def evaluate(
        self,
        model_name_or_path: str,
        batch_size: int = 64,
        top_k: int = 10,
    ) -> RetrievalMetrics:
        """
        Оценивает одну модель.

        Параметры
        ---------
        model_name_or_path : HuggingFace ID или локальный путь
        batch_size         : размер батча для encode
        top_k              : глубина поиска
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name_or_path, device=device)

        corpus_embs = self._encode_corpus(model, batch_size)
        query_embs = self._encode_queries(model, batch_size)

        index = _build_index(corpus_embs)

        # Поиск
        rankings: List[List[str]] = []
        for q_emb in tqdm(query_embs, desc="Поиск", leave=False):
            _, top_indices = index.search(q_emb, top_k)
            ranked_doc_ids = [self._corpus_ids[int(i)] for i in top_indices]
            rankings.append(ranked_doc_ids)

        k_max = max(self.ks)
        metrics = RetrievalMetrics(
            model_name=model_name_or_path,
            n_queries=len(self._query_ids),
            n_corpus=len(self._corpus_ids),
        )

        for k in self.ks:
            recall = _recall_at_k(rankings, self.qrels, self._query_ids, k)
            setattr(metrics, f"recall_at_{k}", recall)

        metrics.mrr_at_10 = _mrr_at_k(rankings, self.qrels, self._query_ids, 10)
        metrics.ndcg_at_10 = _ndcg_at_k(rankings, self.qrels, self._query_ids, 10)
        metrics.map_at_10 = _map_at_k(rankings, self.qrels, self._query_ids, 10)

        logger.info(
            "Оценка %s | R@1=%.4f R@5=%.4f R@10=%.4f MRR@10=%.4f NDCG@10=%.4f",
            model_name_or_path,
            metrics.recall_at_1,
            metrics.recall_at_5,
            metrics.recall_at_10,
            metrics.mrr_at_10,
            metrics.ndcg_at_10,
        )
        return metrics

    # ── Compare models ───────────────────────────────────────────────────────

    def compare_models(
        self,
        models: Dict[str, str],
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """
        Сравнивает несколько моделей.

        Параметры
        ---------
        models : Dict[alias -> model_name_or_path]
                 Например: {"original": "deepvk/USER-bge-m3", "finetuned": "./output"}

        Возвращает
        ----------
        pd.DataFrame с метриками для каждой модели
        """
        results = []
        for alias, model_path in models.items():
            logger.info("Оценка модели: %s (%s)", alias, model_path)
            m = self.evaluate(model_path, batch_size=batch_size)
            row = m.to_dict()
            row["alias"] = alias
            results.append(row)

        df = pd.DataFrame(results).set_index("alias")
        return df

    # ── Plot ─────────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Строит bar-chart сравнения метрик между моделями.

        Параметры
        ---------
        results_df : результат compare_models()
        save_path  : путь для сохранения (None = только показать)
        """
        metric_cols = [c for c in results_df.columns if c.startswith(("Recall", "MRR", "NDCG", "MAP"))]
        plot_df = results_df[metric_cols].copy()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── График 1: Recall@k ───────────────────────────────────────────────
        recall_cols = [c for c in metric_cols if c.startswith("Recall")]
        recall_data = plot_df[recall_cols].T
        recall_data.index = [c.replace("Recall@", "k=") for c in recall_cols]
        recall_data.plot(
            kind="bar",
            ax=axes[0],
            colormap="viridis",
            width=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_title("Recall@k", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("k", fontsize=11)
        axes[0].set_ylabel("Recall", fontsize=11)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(title="Модель", fontsize=9)
        axes[0].tick_params(axis="x", rotation=0)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt="%.3f", fontsize=7, padding=2)

        # ── График 2: MRR / NDCG / MAP ──────────────────────────────────────
        other_cols = [c for c in metric_cols if not c.startswith("Recall")]
        if other_cols:
            other_data = plot_df[other_cols]
            other_data.plot(
                kind="bar",
                ax=axes[1],
                colormap="plasma",
                width=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            axes[1].set_title("MRR@10 / NDCG@10 / MAP@10", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Модель", fontsize=11)
            axes[1].set_ylabel("Score", fontsize=11)
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(title="Метрика", fontsize=9)
            axes[1].tick_params(axis="x", rotation=15)
            for container in axes[1].containers:
                axes[1].bar_label(container, fmt="%.3f", fontsize=7, padding=2)

        plt.suptitle(
            "Сравнение оригинальной и дообученных моделей",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("График сохранён: %s", save_path)

        return fig

    def plot_recall_improvement(
        self,
        results_df: pd.DataFrame,
        baseline_alias: str = "original",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Строит график абсолютного прироста Recall@k относительно baseline."""
        recall_cols = [c for c in results_df.columns if c.startswith("Recall")]
        if baseline_alias not in results_df.index:
            raise ValueError(f"Alias '{baseline_alias}' не найден в results_df")

        baseline = results_df.loc[baseline_alias, recall_cols]
        delta = results_df[recall_cols].subtract(baseline)
        delta = delta.drop(index=baseline_alias)

        fig, ax = plt.subplots(figsize=(10, 5))
        delta.T.plot(
            kind="bar",
            ax=ax,
            colormap="coolwarm",
            width=0.6,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"Прирост Recall@k относительно {baseline_alias}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("ΔRecall", fontsize=11)
        ax.legend(title="Модель", fontsize=9)
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation set builder (из TripletSampler-данных)
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_set(
    triplets_df: pd.DataFrame,
    n_queries: int = 200,
    seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Формирует набор данных для оценки retrieval из DataFrame триплетов.

    Логика:
      - anchor    → запрос (query)
      - positive  → релевантный документ корпуса
      - negatives → нерелевантные документы (из колонки negative + остальные positive)

    Возвращает
    ----------
    queries : Dict[query_id, query_text]
    corpus  : Dict[doc_id, doc_text]
    qrels   : Dict[query_id, List[doc_id]]
    """
    import hashlib

    rng = np.random.default_rng(seed)
    sample = triplets_df.sample(n=min(n_queries, len(triplets_df)), random_state=seed)

    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    qrels: Dict[str, List[str]] = {}

    def _text_id(text: str) -> str:
        return "doc_" + hashlib.md5(text.encode()).hexdigest()[:12]

    # Строим корпус из всех positives и negatives
    for _, row in triplets_df.iterrows():
        pid = _text_id(row["positive"])
        corpus[pid] = row["positive"]
        if "negative" in row and pd.notna(row["negative"]):
            nid = _text_id(row["negative"])
            corpus[nid] = row["negative"]

    # Строим queries и qrels из выборки
    for i, (_, row) in enumerate(sample.iterrows()):
        qid = f"q_{i:05d}"
        queries[qid] = row["anchor"]
        pid = _text_id(row["positive"])
        qrels[qid] = [pid]

    logger.info(
        "Evaluation set: %d запросов, %d документов в корпусе",
        len(queries),
        len(corpus),
    )
    return queries, corpus, qrels
