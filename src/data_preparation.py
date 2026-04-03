"""
data_preparation.py
===================
Подготовка обучающих триплетов (anchor, positive, negative) для дообучения
эмбеддинговой модели на банковских документах.

Стратегии формирования выборки
───────────────────────────────
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Стратегия       │ Описание                                                 │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ ADJACENT        │ anchor = чанк_i,  positive = чанк_i±1 того же документа  │
│                 │ negative = случайный чанк из другого документа           │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ SECTION         │ anchor и positive из одной секции документа;             │
│                 │ negative из другой секции / другого документа            │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ BM25_HARD       │ negative = BM25-похожий чанк из другого документа        │
│                 │ (тяжёлый негатив — ближе к якорю семантически)           │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ QUERY_TEMPLATE  │ anchor = шаблонный запрос по чанку (без LLM)             │
│                 │ positive = исходный чанк                                 │
│                 │ negative = BM25-hard из другого документа               │
└─────────────────┴──────────────────────────────────────────────────────────┘

Итоговый DataFrame:
┌──────────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ anchor                           │ positive                     │ negative                     │
├──────────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Каковы условия ипотечного кредита│ Ставка по ипотеке составляет │ Правила открытия вклада...   │
│ Как открыть вклад в банке?       │ Для открытия вклада нужно... │ Требования к заёмщикам...    │
└──────────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Единица текста — чанк / абзац банковского документа."""
    text: str
    doc_id: str
    chunk_id: str
    section: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    def tokens(self) -> List[str]:
        """Простая токенизация для BM25."""
        return re.sub(r"[^\w\s]", " ", self.text.lower()).split()


@dataclass
class Triplet:
    anchor: str
    positive: str
    negative: str
    strategy: str = ""
    anchor_chunk_id: str = ""
    positive_chunk_id: str = ""
    negative_chunk_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Hard Negative Miner
# ─────────────────────────────────────────────────────────────────────────────

class BM25HardNegativeMiner:
    """
    Формирует тяжёлые негативы через BM25: находит чанки с высоким
    лексическим сходством, но принадлежащие другому документу.

    Пример использования
    --------------------
    >>> miner = BM25HardNegativeMiner(chunks)
    >>> negatives = miner.get_hard_negatives(
    ...     query="ставка по ипотеке",
    ...     exclude_doc_id="doc_001",
    ...     top_k=5,
    ... )
    """

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [c.tokens() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25HardNegativeMiner инициализирован на %d чанках", len(chunks))

    def get_hard_negatives(
        self,
        query: str,
        exclude_doc_id: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Возвращает top_k чанков с наивысшим BM25-score,
        исключая документ `exclude_doc_id`.
        """
        query_tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
        scores = self.bm25.get_scores(query_tokens)

        # np.argpartition — O(n) вместо O(n log n) полной сортировки.
        # Берём top_k * 4 кандидатов с запасом (часть отсеется по exclude_doc_id).
        k = min(top_k * 4, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        result: List[Chunk] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            if chunk.doc_id != exclude_doc_id:
                result.append(chunk)
            if len(result) >= top_k:
                break
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Triplet Sampler
# ─────────────────────────────────────────────────────────────────────────────

class TripletSampler:
    """
    Формирует обучающую выборку триплетов из списка чанков.

    Параметры
    ---------
    chunks          : List[Chunk]  — все чанки корпуса
    negative_strategy : str        — "bm25" | "random" | "mixed"
    n_hard_neg_pool : int          — из скольких hard-негативов выбирать 1
    seed            : int

    Пример
    ------
    >>> sampler = TripletSampler(chunks, negative_strategy="bm25")
    >>> df = sampler.sample_adjacent(n_per_chunk=2)
    >>> df_q = sampler.sample_query_template(n_per_chunk=1)
    >>> full_df = sampler.to_dataframe(strategies=["adjacent", "query_template"])
    """

    QUERY_TEMPLATES = [
        "Что такое {topic}?",
        "Каковы условия {topic}?",
        "Как работает {topic}?",
        "Расскажи о {topic}",
        "Объясни {topic}",
        "Правила {topic}",
        "Требования к {topic}",
        "Порядок оформления {topic}",
        "Какие документы нужны для {topic}?",
        "В чём особенности {topic}?",
    ]

    def __init__(
        self,
        chunks: List[Chunk],
        negative_strategy: str = "bm25",
        n_hard_neg_pool: int = 10,
        seed: int = 42,
        max_chunks: Optional[int] = None,
    ) -> None:
        if not chunks:
            raise ValueError("Список чанков не может быть пустым")

        self.negative_strategy = negative_strategy
        self.n_hard_neg_pool = n_hard_neg_pool
        self.seed = seed
        random.seed(seed)

        # Ограничиваем корпус если задан max_chunks.
        # Выборка стратифицирована по doc_id — берём пропорционально из каждого документа.
        if max_chunks and len(chunks) > max_chunks:
            by_doc: Dict[str, List[Chunk]] = {}
            for c in chunks:
                by_doc.setdefault(c.doc_id, []).append(c)
            sampled: List[Chunk] = []
            ratio = max_chunks / len(chunks)
            for doc_chunks in by_doc.values():
                k = max(1, round(len(doc_chunks) * ratio))
                sampled.extend(random.sample(doc_chunks, min(k, len(doc_chunks))))
            chunks = sampled[:max_chunks]
            logger.info(
                "max_chunks=%d: корпус сокращён %d → %d чанков",
                max_chunks, len(sampled), len(chunks),
            )

        self.chunks = chunks

        # Индексы: doc_id -> список чанков
        self._by_doc: Dict[str, List[Chunk]] = {}
        for c in chunks:
            self._by_doc.setdefault(c.doc_id, []).append(c)

        # Индексы: section -> список чанков (кроссдокументные)
        self._by_section: Dict[str, List[Chunk]] = {}
        for c in chunks:
            if c.section:
                self._by_section.setdefault(c.section, []).append(c)

        if negative_strategy in ("bm25", "mixed"):
            self._miner = BM25HardNegativeMiner(chunks)
        else:
            self._miner = None  # type: ignore[assignment]

        logger.info(
            "TripletSampler: %d чанков, %d документов, стратегия=%s",
            len(chunks),
            len(self._by_doc),
            negative_strategy,
        )

    # ── Получение негатива ───────────────────────────────────────────────────

    def _get_negative(self, anchor_chunk: Chunk) -> Optional[Chunk]:
        """Выбирает негатив согласно стратегии."""
        strategy = self.negative_strategy
        if strategy == "mixed":
            strategy = random.choice(["bm25", "random"])

        if strategy == "bm25" and self._miner is not None:
            candidates = self._miner.get_hard_negatives(
                anchor_chunk.text,
                exclude_doc_id=anchor_chunk.doc_id,
                top_k=self.n_hard_neg_pool,
            )
            if candidates:
                return random.choice(candidates)

        # Fallback: случайный чанк из другого документа
        other_docs = [d for d in self._by_doc if d != anchor_chunk.doc_id]
        if not other_docs:
            return None
        neg_doc = random.choice(other_docs)
        return random.choice(self._by_doc[neg_doc])

    # ── Стратегия 1: ADJACENT ────────────────────────────────────────────────

    def sample_adjacent(self, n_per_chunk: int = 1) -> List[Triplet]:
        """
        Positive = соседний чанк того же документа (i±1).
        Хорошо работает для документов с линейной структурой.
        """
        triplets: List[Triplet] = []

        for doc_id, doc_chunks in self._by_doc.items():
            if len(doc_chunks) < 2:
                continue
            for i, anchor in enumerate(doc_chunks):
                # Позитив: следующий или предыдущий чанк
                pos_indices = []
                if i + 1 < len(doc_chunks):
                    pos_indices.append(i + 1)
                if i - 1 >= 0:
                    pos_indices.append(i - 1)

                for _ in range(n_per_chunk):
                    pos_chunk = doc_chunks[random.choice(pos_indices)]
                    neg_chunk = self._get_negative(anchor)
                    if neg_chunk is None:
                        continue
                    triplets.append(
                        Triplet(
                            anchor=anchor.text,
                            positive=pos_chunk.text,
                            negative=neg_chunk.text,
                            strategy="adjacent",
                            anchor_chunk_id=anchor.chunk_id,
                            positive_chunk_id=pos_chunk.chunk_id,
                            negative_chunk_id=neg_chunk.chunk_id,
                        )
                    )
        logger.info("adjacent: сформировано %d триплетов", len(triplets))
        return triplets

    # ── Стратегия 2: SECTION ─────────────────────────────────────────────────

    def sample_section(self, n_per_chunk: int = 1) -> List[Triplet]:
        """
        Positive = другой чанк той же секции.
        Negative = чанк из другой секции или другого документа.
        """
        triplets: List[Triplet] = []

        for section, sec_chunks in self._by_section.items():
            if len(sec_chunks) < 2:
                continue
            for anchor in sec_chunks:
                pool = [c for c in sec_chunks if c.chunk_id != anchor.chunk_id]
                if not pool:
                    continue
                for _ in range(n_per_chunk):
                    pos_chunk = random.choice(pool)
                    neg_chunk = self._get_negative(anchor)
                    if neg_chunk is None:
                        continue
                    triplets.append(
                        Triplet(
                            anchor=anchor.text,
                            positive=pos_chunk.text,
                            negative=neg_chunk.text,
                            strategy="section",
                            anchor_chunk_id=anchor.chunk_id,
                            positive_chunk_id=pos_chunk.chunk_id,
                            negative_chunk_id=neg_chunk.chunk_id,
                        )
                    )
        logger.info("section: сформировано %d триплетов", len(triplets))
        return triplets

    # ── Стратегия 3: QUERY TEMPLATE (без LLM) ────────────────────────────────

    def _extract_topic(self, text: str, max_words: int = 5) -> str:
        """Извлекает ключевые слова для формирования запроса."""
        # Берём существительные из начала текста как тему (простая эвристика)
        words = text.split()[:max_words]
        return " ".join(words).strip(".,;:—")

    def sample_query_template(
        self,
        n_per_chunk: int = 1,
        max_chunks_qt: Optional[int] = None,
    ) -> List[Triplet]:
        """
        Anchor = синтетический запрос по шаблону.
        Positive = исходный чанк.
        Negative = BM25-hard из другого документа.

        max_chunks_qt : ограничить число чанков для этой стратегии
                        (BM25-поиск дорогой на больших корпусах).
        """
        triplets: List[Triplet] = []
        pool = self.chunks
        if max_chunks_qt and len(pool) > max_chunks_qt:
            pool = random.sample(pool, max_chunks_qt)
            logger.info("query_template: ограничено до %d чанков", max_chunks_qt)

        for chunk in tqdm(pool, desc="query_template", leave=False):
            topic = self._extract_topic(chunk.text)
            neg_chunk = self._get_negative(chunk)
            if neg_chunk is None:
                continue
            for _ in range(n_per_chunk):
                template = random.choice(self.QUERY_TEMPLATES)
                query = template.format(topic=topic)
                triplets.append(
                    Triplet(
                        anchor=query,
                        positive=chunk.text,
                        negative=neg_chunk.text,
                        strategy="query_template",
                        anchor_chunk_id="",
                        positive_chunk_id=chunk.chunk_id,
                        negative_chunk_id=neg_chunk.chunk_id,
                    )
                )
        logger.info("query_template: сформировано %d триплетов", len(triplets))
        return triplets

    # ── Основной метод ───────────────────────────────────────────────────────

    def to_dataframe(
        self,
        strategies: Optional[List[str]] = None,
        n_per_chunk: int = 1,
        deduplicate: bool = True,
        max_chunks_qt: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Формирует итоговый DataFrame со всеми триплетами.

        Параметры
        ---------
        strategies    : список стратегий (default: все три)
        n_per_chunk   : количество триплетов на каждый чанк
        deduplicate   : убрать дубликаты по (anchor, positive)

        Возвращает
        ----------
        pd.DataFrame с колонками:
          anchor, positive, negative, strategy,
          anchor_chunk_id, positive_chunk_id, negative_chunk_id
        """
        if strategies is None:
            strategies = ["adjacent", "section", "query_template"]

        all_triplets: List[Triplet] = []

        for strategy in strategies:
            if strategy == "adjacent":
                all_triplets.extend(self.sample_adjacent(n_per_chunk))
            elif strategy == "section":
                all_triplets.extend(self.sample_section(n_per_chunk))
            elif strategy == "query_template":
                all_triplets.extend(
                    self.sample_query_template(n_per_chunk, max_chunks_qt=max_chunks_qt)
                )
            else:
                raise ValueError(f"Неизвестная стратегия: {strategy!r}")

        df = pd.DataFrame(
            [
                {
                    "anchor": t.anchor,
                    "positive": t.positive,
                    "negative": t.negative,
                    "strategy": t.strategy,
                    "anchor_chunk_id": t.anchor_chunk_id,
                    "positive_chunk_id": t.positive_chunk_id,
                    "negative_chunk_id": t.negative_chunk_id,
                }
                for t in all_triplets
            ]
        )

        if deduplicate and not df.empty:
            before = len(df)
            df = df.drop_duplicates(subset=["anchor", "positive"]).reset_index(drop=True)
            logger.info(
                "Дедупликация: %d -> %d триплетов", before, len(df)
            )

        logger.info(
            "Итого триплетов: %d  (стратегии: %s)",
            len(df),
            df["strategy"].value_counts().to_dict() if not df.empty else {},
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Document Processor
# ─────────────────────────────────────────────────────────────────────────────

# Диапазоны Unicode для эмоджи и спецсимволов
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002500-\U00002BEF"   # chinese/japanese/korean
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)

# Паттерн разбивки на предложения: точка / ! / ? → конец предложения
# Смотрим за 1+ пробельных символов и заглавной буквы или цифры следующего
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class MarkdownDocumentProcessor:
    """
    Очищает markdown-документы и нарезает их на чанки из законченных предложений.

    Этапы
    -----
    1. clean(text)       — убирает markdown-разметку, HTML, эмоджи, лишние символы
    2. split_sentences() — разбивает очищенный текст на предложения
    3. analyze_lengths() — считает распределение длин предложений (слова)
    4. build_chunks()    — объединяет предложения в чанки с перекрытием 1 предложение
    5. process_df()      — обрабатывает весь DataFrame сразу → List[Chunk]

    Параметры
    ---------
    sentences_per_chunk : int | None
        Если None — определяется автоматически из медианы длин предложений
        с целью ~target_words_per_chunk слов в чанке.
    target_words_per_chunk : int
        Целевой размер чанка в словах (используется при автовыборе sentences_per_chunk).
    min_sentence_words : int
        Предложения короче этого порога отбрасываются (колонтитулы, заголовки).
    """

    def __init__(
        self,
        sentences_per_chunk: Optional[int] = None,
        target_words_per_chunk: int = 120,
        min_sentence_words: int = 5,
    ) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self.target_words_per_chunk = target_words_per_chunk
        self.min_sentence_words = min_sentence_words

    # ── Шаг 1: Очистка текста ────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Убирает markdown, HTML, эмоджи; оставляет буквы, цифры и точки."""
        # Markdown: блоки кода ```...```
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # Markdown: инлайн-код `...`
        text = re.sub(r"`[^`]+`", " ", text)
        # Markdown: изображения ![alt](url)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        # Markdown: ссылки [текст](url) → оставляем текст
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        # Markdown: заголовки # ## ###
        text = re.sub(r"^\s*#{1,6}\s+", " ", text, flags=re.MULTILINE)
        # Markdown: жирный/курсив **text** *text* __text__ _text_
        text = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", text)
        # Markdown: горизонтальные линии --- *** ___
        text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
        # Markdown: таблицы (строки с |)
        text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-|: ]+\s*$", " ", text, flags=re.MULTILINE)
        # HTML-теги
        text = re.sub(r"<[^>]+>", " ", text)
        # HTML-сущности &nbsp; &amp; и т.д.
        text = re.sub(r"&[a-zA-Z]{2,6};", " ", text)
        # Эмоджи
        text = _EMOJI_RE.sub(" ", text)
        # Спецсимволы: оставляем буквы, цифры, пробел, точку, запятую, дефис, скобки
        # Запятая нужна чтобы не терять числа типа "1,5%"; дефис — для составных слов
        text = re.sub(r"[^\w\s.,;:\-—()\[\]]", " ", text, flags=re.UNICODE)
        # Множественные пробелы и пустые строки → один пробел
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", ". ", text)   # двойной перенос = конец абзаца
        text = re.sub(r"\n", " ", text)
        # Множественные точки → одна
        text = re.sub(r"\.{2,}", ".", text)
        # Пробел перед точкой → убрать
        text = re.sub(r"\s+\.", ".", text)
        return text.strip()

    # ── Шаг 2: Разбивка на предложения ───────────────────────────────────────

    def split_sentences(self, text: str) -> List[str]:
        """Разбивает очищенный текст на непустые предложения."""
        parts = _SENT_SPLIT_RE.split(text)
        sentences = []
        for s in parts:
            s = s.strip().strip(".")
            if len(s.split()) >= self.min_sentence_words:
                sentences.append(s)
        return sentences

    # ── Шаг 3: Анализ длин предложений ───────────────────────────────────────

    def analyze_lengths(self, all_sentences: List[str]) -> Dict:
        """
        Считает статистику длин предложений (в словах).
        Возвращает словарь с percentiles и рекомендованным sentences_per_chunk.
        """
        lengths = np.array([len(s.split()) for s in all_sentences])
        p25, p50, p75, p90 = np.percentile(lengths, [25, 50, 75, 90])
        median = float(p50)
        recommended = max(2, round(self.target_words_per_chunk / median))

        stats = {
            "count":           len(lengths),
            "mean":            float(lengths.mean()),
            "median":          median,
            "p25":             float(p25),
            "p75":             float(p75),
            "p90":             float(p90),
            "max":             int(lengths.max()),
            "min":             int(lengths.min()),
            "recommended_spc": recommended,   # sentences_per_chunk
        }
        logger.info(
            "Длины предложений: median=%.1f слов | p25=%.1f p75=%.1f p90=%.1f "
            "| рекомендуется sentences_per_chunk=%d (цель ~%d слов/чанк)",
            stats["median"], stats["p25"], stats["p75"], stats["p90"],
            recommended, self.target_words_per_chunk,
        )
        return stats

    # ── Шаг 4: Сборка чанков ─────────────────────────────────────────────────

    def build_chunks(
        self,
        sentences: List[str],
        doc_id: str,
        spc: int,
    ) -> List[Chunk]:
        """
        Объединяет предложения в чанки по spc штук с перекрытием 1 предложение слева.

        Пример при spc=3, предложения [A, B, C, D, E, F]:
          чанк 0: [A, B, C]
          чанк 1: [C, D, E]   ← C — перекрытие с предыдущим
          чанк 2: [E, F]      ← последний может быть короче
        """
        chunks: List[Chunk] = []
        step = spc - 1   # шаг = spc минус 1 перекрывающееся предложение
        i = 0
        chunk_idx = 0
        while i < len(sentences):
            window = sentences[i: i + spc]
            text = ". ".join(window) + "."
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_c{chunk_idx:04d}",
            ))
            chunk_idx += 1
            i += step
        return chunks

    # ── Шаг 5: Обработка DataFrame ───────────────────────────────────────────

    def process_df(
        self,
        df: pd.DataFrame,
        doc_col: str = "filename",
        text_col: str = "full_text",
    ) -> List[Chunk]:
        """
        Обрабатывает DataFrame с документами → List[Chunk] для TSDAE.

        Параметры
        ---------
        df       : DataFrame с колонками doc_col и text_col
        doc_col  : имя колонки с идентификатором документа
        text_col : имя колонки с полным текстом документа

        Возвращает
        ----------
        List[Chunk] готовый для TripletSampler или для corpus.csv
        """
        # Проход 1: очищаем и собираем все предложения для анализа длин
        all_sentences: List[str] = []
        cleaned_docs: List[Tuple[str, List[str]]] = []

        logger.info("Очистка %d документов...", len(df))
        for _, row in tqdm(df.iterrows(), total=len(df), desc="clean"):
            doc_id = str(row[doc_col])
            cleaned = self.clean(str(row[text_col]))
            sents = self.split_sentences(cleaned)
            cleaned_docs.append((doc_id, sents))
            all_sentences.extend(sents)

        if not all_sentences:
            logger.warning("Ни одного предложения не извлечено. Проверьте входные данные.")
            return []

        # Анализ длин → автовыбор sentences_per_chunk если не задан
        stats = self.analyze_lengths(all_sentences)
        spc = self.sentences_per_chunk or stats["recommended_spc"]
        logger.info("sentences_per_chunk=%d (перекрытие=1 предложение)", spc)

        # Проход 2: собираем чанки
        all_chunks: List[Chunk] = []
        for doc_id, sents in tqdm(cleaned_docs, desc="chunk"):
            if not sents:
                continue
            all_chunks.extend(self.build_chunks(sents, doc_id, spc))

        logger.info(
            "Итого: %d документов → %d чанков (spc=%d)",
            len(df), len(all_chunks), spc,
        )
        return all_chunks

    @staticmethod
    def chunks_to_df(chunks: List[Chunk]) -> pd.DataFrame:
        """Конвертирует List[Chunk] в DataFrame формата corpus.csv."""
        return pd.DataFrame([
            {
                "doc_id":   c.doc_id,
                "chunk_id": c.chunk_id,
                "sentence": c.text,
            }
            for c in chunks
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Document Chunker
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Разбивает текст документа на чанки по абзацам или скользящим окном.

    Пример
    ------
    >>> chunker = DocumentChunker(chunk_size=256, overlap=32)
    >>> chunks = chunker.chunk_document(text, doc_id="mortgage_policy")
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        split_by: str = "paragraph",  # "paragraph" | "tokens"
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_by = split_by

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        section: str = "",
    ) -> List[Chunk]:
        """Разбивает документ на чанки."""
        if self.split_by == "paragraph":
            return self._split_by_paragraph(text, doc_id, section)
        return self._split_by_tokens(text, doc_id, section)

    def _split_by_paragraph(
        self, text: str, doc_id: str, section: str
    ) -> List[Chunk]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para.split()) < 5:  # пропускаем заголовки-одиночки
                continue
            chunks.append(
                Chunk(
                    text=para,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_p{i:04d}",
                    section=section,
                )
            )
        return chunks

    def _split_by_tokens(
        self, text: str, doc_id: str, section: str
    ) -> List[Chunk]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + self.chunk_size]
            if len(chunk_words) < 10:
                break
            chunks.append(
                Chunk(
                    text=" ".join(chunk_words),
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_w{i:04d}",
                    section=section,
                )
            )
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Генератор демонстрационных банковских данных
# ─────────────────────────────────────────────────────────────────────────────

_BANKING_CORPUS: Dict[str, Dict[str, str]] = {
    # ── Кредитные продукты ────────────────────────────────────────────────────
    "doc_mortgage": {
        "section": "Ипотека",
        "text": """
Ипотечное кредитование — долгосрочный банковский продукт, позволяющий приобрести
недвижимость в кредит под залог этой же или иной недвижимости.

Процентная ставка по ипотеке определяется индивидуально для каждого заёмщика
и зависит от размера первоначального взноса, срока кредита, кредитной истории
и программы государственного субсидирования.

Первоначальный взнос — сумма, которую заёмщик вносит из собственных средств
при оформлении ипотеки. Минимальный первоначальный взнос составляет 15–20% от
стоимости приобретаемой недвижимости.

Страхование при ипотеке является обязательным: заёмщик обязан застраховать
предмет залога от рисков утраты и повреждения. Личное страхование заёмщика
носит добровольный характер, однако влияет на процентную ставку.

Досрочное погашение ипотеки разрешено без штрафных санкций в соответствии
с действующим законодательством. Заёмщик вправе погасить кредит полностью
или частично в любой момент после уведомления банка за 30 дней.

Рефинансирование ипотеки позволяет снизить текущую процентную ставку,
переоформив кредит в другом банке или в том же банке на новых условиях.

Льготная ипотека с государственной поддержкой действует по программам:
«Семейная ипотека» (ставка до 6%), «Дальневосточная ипотека» (до 2%),
«IT-ипотека» (до 5%). Условия и лимиты сумм устанавливаются Правительством РФ.

Эскроу-счёт при покупке строящегося жилья (ДДУ) аккумулирует средства дольщика
до момента ввода объекта в эксплуатацию. Застройщик получает деньги только после
регистрации права собственности покупателя. Это защищает от рисков недостроя.
""",
    },
    "doc_auto_loan": {
        "section": "Автокредитование",
        "text": """
Автокредит — целевой банковский кредит на покупку транспортного средства
под залог приобретаемого автомобиля. Срок кредита составляет от 1 до 7 лет.

Процентная ставка по автокредиту зависит от первоначального взноса, срока
кредитования, типа автомобиля (новый/подержанный) и кредитной истории заёмщика.
Ставки по программам господдержки («Семейный автомобиль», «Первый автомобиль»)
ниже рыночных на 2–3 процентных пункта.

Страхование КАСКО является обязательным условием автокредита: автомобиль
страхуется от угона и ущерба на весь срок кредитования. Стоимость КАСКО
включается в полную стоимость кредита (ПСК).

При покупке подержанного автомобиля банк требует проверку по реестру залогов
и базе ГИБДД, чтобы исключить наличие обременений. Автомобили старше 10 лет
большинство банков в залог не принимают.

Первоначальный взнос по автокредиту составляет обычно 10–20% от стоимости
автомобиля. Trade-in (зачёт стоимости старого автомобиля) принимается в счёт
первоначального взноса в большинстве банков-партнёров автосалонов.

Комиссия за выдачу автокредита не взимается: все расходы отражаются в ПСК.
При досрочном погашении автокредита пересчёт страховой премии осуществляется
пропорционально оставшемуся сроку страхования.
""",
    },
    "doc_loans": {
        "section": "Потребительские кредиты",
        "text": """
Потребительский кредит — банковский продукт для финансирования личных нужд
граждан: покупки товаров, оплаты услуг, ремонта, путешествий и т.д.

Требования к заёмщику: гражданство РФ, возраст от 18 до 70 лет, постоянная
регистрация, стаж на текущем месте работы не менее 3 месяцев, подтверждённый
доход. Хорошая кредитная история повышает вероятность одобрения и снижает ставку.

Полная стоимость кредита (ПСК) — показатель, отражающий все расходы заёмщика
по кредиту в процентах годовых. ПСК включает процентную ставку, комиссии
и стоимость обязательного страхования. Банк обязан раскрыть ПСК до подписания.

Аннуитетный платёж — равные ежемесячные платежи на протяжении всего срока
кредита. Дифференцированный платёж уменьшается каждый месяц, так как основной
долг погашается равными долями. Суммарная переплата при аннуитете выше.

Кредитная история формируется в Бюро кредитных историй (БКИ) и отражает
все кредитные обязательства гражданина. Просрочки платежей негативно
влияют на кредитный рейтинг и условия будущих кредитов.

Рефинансирование потребительского кредита позволяет объединить несколько
кредитов в один и снизить ежемесячный платёж за счёт более низкой ставки
или увеличения срока. Для рефинансирования требуется отсутствие текущих просрочек.

Микрозаймы и кредиты «до зарплаты» отличаются от потребительских кредитов
коротким сроком (до 30 дней), высокой ставкой и упрощённой процедурой выдачи.
Максимальный размер начисленных процентов ограничен законом: не более 130% суммы займа.
""",
    },
    "doc_sme_loans": {
        "section": "Кредиты для бизнеса",
        "text": """
Кредитование малого и среднего бизнеса (МСБ) — комплекс банковских продуктов
для финансирования операционной деятельности и развития предприятий.

Виды кредитов для МСБ: оборотный кредит (финансирование текущей деятельности),
инвестиционный кредит (приобретение оборудования, недвижимости), кредитная линия
(возобновляемый лимит задолженности), овердрафт (покрытие кассовых разрывов).

Требования к заёмщику-юрлицу: срок ведения деятельности не менее 6–12 месяцев,
положительная кредитная история, наличие оборотов по расчётному счёту в банке.
Для МСП, включённых в реестр ФНС, доступны льготные программы по ставке ниже 8%.

Обеспечение по бизнес-кредиту: залог недвижимости, оборудования, товарно-материальных
ценностей, поручительство собственников бизнеса или гарантии фондов поддержки МСП.
Незалоговые кредиты до 5 млн руб. доступны при оборотах по счёту от 1 млн руб./мес.

Программа «1764» субсидирует ставки по кредитам для МСП в приоритетных отраслях
(сельское хозяйство, обрабатывающее производство, туризм): ставка — ключевая +2,75%.

Скоринг для бизнеса учитывает финансовую отчётность, налоговую нагрузку,
обороты по счетам, количество сотрудников и отраслевые риски. Банк вправе
запросить управленческую отчётность и данные о контрагентах.
""",
    },
    # ── Сберегательные продукты ───────────────────────────────────────────────
    "doc_deposits": {
        "section": "Вклады",
        "text": """
Банковский вклад (депозит) — денежные средства, размещаемые физическими или
юридическими лицами в банке на определённый срок или до востребования.

Процентная ставка по вкладу зависит от суммы, срока размещения и условий
договора. Максимальные ставки предлагаются по срочным вкладам без
возможности пополнения и частичного снятия.

Страхование вкладов в России осуществляется Агентством по страхованию
вкладов (АСВ). Возмещению подлежит сумма до 1 400 000 рублей на одного
вкладчика в одном банке. Средства на счетах эскроу застрахованы до 10 млн руб.

Виды вкладов: вклад «до востребования» — средства можно снять в любой момент;
срочный вклад — фиксированный срок и ставка; накопительный вклад — позволяет
пополнять счёт в течение срока хранения.

Капитализация процентов означает начисление процентов на уже начисленные
проценты. Вклады с капитализацией обеспечивают более высокую эффективную
доходность по сравнению со вкладами с выплатой процентов на отдельный счёт.

Валютные вклады размещаются в долларах США, евро и других иностранных валютах.
Доходность по валютным вкладам ниже рублёвых, однако они защищают от девальвации.
Страхование распространяется на валютные вклады в эквиваленте рублей по курсу ЦБ.

Мультивалютный вклад позволяет хранить средства в нескольких валютах одновременно
и переводить их между корзинами без потери накопленных процентов. Это инструмент
хеджирования валютного риска для частных лиц.
""",
    },
    "doc_savings_account": {
        "section": "Накопительные счета",
        "text": """
Накопительный счёт — гибридный банковский продукт, совмещающий удобство
текущего счёта и доходность срочного вклада.

Процентная ставка по накопительному счёту устанавливается на ежемесячной
основе и зависит от остатка. Проценты начисляются ежедневно на минимальный
остаток за расчётный период. Снятие и пополнение средств доступно в любой момент.

Отличие накопительного счёта от вклада: отсутствие фиксированного срока
и возможность свободного управления средствами без потери процентов. Ставки
немного ниже, чем по срочным вкладам с аналогичными суммами.

Бонусные программы на накопительных счетах: повышенная ставка на первые 2–3 месяца,
дополнительные проценты при использовании кредитной карты банка или подключении
пакета услуг. Промо-ставки достигают 18–20% годовых в рублях.

Страхование средств на накопительных счетах осуществляется в рамках системы
страхования вкладов (АСВ) на сумму до 1 400 000 рублей. Это делает накопительные
счета таким же надёжным инструментом сбережений, как и срочные вклады.

Налог на доходы по накопительным счетам: проценты, превышающие произведение
ключевой ставки ЦБ РФ и 1 000 000 рублей, облагаются НДФЛ по ставке 13%.
Налоговый агент — банк, удерживающий налог в конце года.
""",
    },
    # ── Карты и платежи ───────────────────────────────────────────────────────
    "doc_cards": {
        "section": "Банковские карты",
        "text": """
Дебетовая карта — платёжный инструмент, привязанный к текущему счёту клиента.
Операции проводятся только в пределах остатка на счёте.

Кредитная карта предоставляет клиенту возобновляемый кредитный лимит.
Льготный период (грейс-период) позволяет пользоваться кредитными средствами
без уплаты процентов в течение 50–120 дней при полном погашении задолженности.

Кэшбэк — возврат части потраченных средств на карту клиента в виде бонусов
или реальных рублей. Размер кэшбэка варьируется от 0,5% до 30% в зависимости
от категории трат и тарифного плана.

Бесконтактная оплата осуществляется с помощью технологии NFC. Для платежей
до 1 000 рублей ввод PIN-кода не требуется. Оплата через Samsung Pay, Mir Pay
и другие кошельки работает аналогично бесконтактной карте.

Лимиты на снятие наличных устанавливаются банком и зависят от типа карты
и тарифного плана. Стандартный дневной лимит снятия составляет 100 000–300 000 рублей.

3D Secure — дополнительный уровень защиты при онлайн-платежах. При совершении
операции в интернете на телефон клиента отправляется одноразовый пароль (OTP).

Виртуальная карта — цифровой аналог пластиковой карты для онлайн-покупок.
Выпускается мгновенно в мобильном приложении. Реквизиты хранятся в токенизированном
виде и не передаются торговым площадкам напрямую.
""",
    },
    # ── Страхование ───────────────────────────────────────────────────────────
    "doc_insurance_property": {
        "section": "Страхование имущества",
        "text": """
Страхование имущества физических лиц — защита квартир, домов, загородной
недвижимости, автомобилей и другого имущества от рисков утраты и повреждения.

Риски по страхованию недвижимости: пожар, залив, кража, вандализм, стихийные
бедствия, взрыв газа. Страховая сумма устанавливается не выше рыночной стоимости
объекта. При ипотеке обязательно страхование предмета залога.

КАСКО — добровольное страхование автомобиля от ущерба и угона. Стоимость полиса
зависит от марки, года выпуска, стажа водителя и региона. Франшиза позволяет
снизить стоимость полиса: безусловная франшиза 15 000 руб. снижает страховой
взнос на 20–25%.

ОСАГО — обязательное страхование гражданской ответственности владельцев
транспортных средств. Страховая выплата по ОСАГО покрывает ущерб, нанесённый
имуществу и здоровью третьих лиц, до 400 000 и 500 000 рублей соответственно.

Коэффициент бонус-малус (КБМ) учитывает историю страховых выплат при расчёте
тарифа ОСАГО. Безаварийные водители получают скидку до 50% после 10 лет без выплат.

Страхование ипотечного имущества оформляется одновременно с договором ипотеки.
Банк является выгодоприобретателем в части непогашенного остатка кредита.
При наступлении страхового события страховщик погашает задолженность перед банком.
""",
    },
    "doc_insurance_life": {
        "section": "Страхование жизни",
        "text": """
Страхование жизни — финансовый инструмент защиты от риска потери жизни
или трудоспособности застрахованного лица и накопления капитала.

Накопительное страхование жизни (НСЖ) объединяет страховую защиту и накопление:
часть взноса идёт на формирование страхового резерва, часть инвестируется.
По окончании срока договора клиент получает накопленную сумму с инвестиционным доходом.

Инвестиционное страхование жизни (ИСЖ) — продукт с защитой вложенного капитала
и участием в росте выбранного инвестиционного индекса. Минимальная гарантия
возврата 100% вложений обеспечивается облигационной частью портфеля.

Срочное страхование жизни обеспечивает выплату страховой суммы наследникам
в случае смерти застрахованного в течение срока действия полиса. Применяется
при ипотечном кредитовании: выплата погашает остаток долга перед банком.

Страхование от несчастных случаев и болезней защищает от временной нетрудоспособности,
инвалидности и смерти по причине несчастного случая. Выплата производится
единовременно или в виде ежедневного пособия за период нетрудоспособности.

Налоговый вычет по договорам страхования жизни сроком от 5 лет предоставляется
в размере взносов до 120 000 рублей в год (налоговая экономия до 15 600 руб.).
Вычет входит в общий лимит социальных вычетов.
""",
    },
    # ── Инвестиции и брокеридж ────────────────────────────────────────────────
    "doc_brokerage": {
        "section": "Брокерское обслуживание",
        "text": """
Брокерский счёт — счёт для проведения операций с ценными бумагами (акциями,
облигациями, ETF) на фондовом рынке через банк-брокера.

Индивидуальный инвестиционный счёт (ИИС) даёт налоговые льготы: тип А —
вычет 13% от суммы взносов до 400 000 руб./год; тип Б — освобождение прибыли
от налога при закрытии через 3 года. С 2024 года доступен ИИС-3 с комбинированными льготами.

Тарифы на брокерское обслуживание включают комиссию за сделку (0,03–0,3% от объёма)
и плату за хранение бумаг на депозитарии. Нулевые тарифы по обслуживанию счёта
устанавливаются при выполнении условий по оборотам.

Маржинальная торговля позволяет заключать сделки на сумму, превышающую собственные
средства. Плечо 1:2 — 1:5 предоставляется квалифицированным инвесторам. Маржин-колл
(принудительное закрытие позиций) наступает при падении стоимости обеспечения.

Облигации федерального займа (ОФЗ) — государственные долговые бумаги с фиксированным
купоном. Доходность к погашению определяется рыночными условиями. ОФЗ-ИН имеют
номинал, индексируемый на инфляцию, и являются инструментом защиты капитала.

Паевые инвестиционные фонды (ПИФы) — коллективный инвестиционный инструмент.
Управляющая компания инвестирует объединённые средства пайщиков в соответствии
с инвестиционной декларацией. Стоимость пая рассчитывается ежедневно.
""",
    },
    # ── Корпоративные услуги ──────────────────────────────────────────────────
    "doc_rko": {
        "section": "Расчётно-кассовое обслуживание",
        "text": """
Расчётно-кассовое обслуживание (РКО) — комплекс банковских услуг для юридических
лиц и индивидуальных предпринимателей по ведению расчётного счёта.

Открытие расчётного счёта ИП и ООО осуществляется на основании пакета документов:
паспорт, ОГРН/ОГРНИП, ИНН, устав (для ООО), решение об учреждении общества.
Онлайн-открытие счёта возможно для большинства организационно-правовых форм за 1 день.

Интернет-банк для бизнеса предоставляет возможность управлять счётом онлайн:
проводить платежи, формировать выписки, подключать дополнительные сервисы.
Бизнес API позволяет интегрировать банковские операции в учётную систему компании.

Эквайринг — услуга приёма безналичной оплаты через POS-терминалы или интернет.
Торговый эквайринг подключается к расчётному счёту предприятия. Интернет-эквайринг
обеспечивает приём платежей через сайт с поддержкой 3DS.

Зарплатный проект позволяет автоматизировать выплату заработной платы
сотрудникам через систему банка. Реестр платежей загружается одним файлом,
а средства зачисляются на карты сотрудников в тот же день.

Корпоративная карта привязывается к расчётному счёту компании и используется
для оплаты командировочных и хозяйственных расходов. Операции отражаются в выписке
по расчётному счёту с описанием назначения платежа.
""",
    },
    "doc_factoring": {
        "section": "Факторинг",
        "text": """
Факторинг — финансирование поставщика под уступку дебиторской задолженности:
банк-фактор выплачивает поставщику до 90% суммы поставки до оплаты покупателем.

Отличие факторинга от кредита: не требует залогового обеспечения, финансирование
привязано к конкретным поставкам, лимит растёт вместе с объёмом продаж. Факторинг
не отражается как кредиторская задолженность в балансе поставщика.

Виды факторинга: с регрессом (при неоплате покупателем поставщик возвращает
финансирование), без регресса (риск неплатежа переходит на фактора), реверсивный
(инициируется покупателем для управления кредиторской задолженностью).

Стоимость факторинга складывается из комиссии за финансирование (проценты за
использование денег), сервисного сбора (обработка документов) и возможного
сбора за оценку дебитора. Эффективная ставка составляет 14–22% годовых.

Электронный факторинг (e-факторинг) ускоряет верификацию поставок: накладные
и акты загружаются в личный кабинет, проверяются автоматически. Финансирование
доступно в день подтверждения поставки покупателем.

Международный факторинг применяется при экспортных сделках: экспортёр получает
финансирование в рублях сразу после отгрузки, риск неплатежа иностранного
покупателя принимает на себя банк-фактор.
""",
    },
    "doc_guarantee": {
        "section": "Банковские гарантии",
        "text": """
Банковская гарантия — письменное обязательство банка выплатить бенефициару
определённую сумму при невыполнении принципалом договорных обязательств.

Виды банковских гарантий: тендерная (обеспечение заявки на участие в тендере),
гарантия исполнения контракта (обеспечение выполнения условий договора),
авансовая гарантия (возврат аванса при нарушении условий поставки).

Для получения гарантии принципал предоставляет финансовую отчётность, контракт
или тендерную документацию, сведения о деятельности. Лимит гарантий устанавливается
в зависимости от оборота и финансового состояния компании.

Гарантия по 44-ФЗ и 223-ФЗ обеспечивает исполнение государственных контрактов.
Требования к банку-гаранту: включение в реестр Минфина, собственный капитал
от 1 млрд рублей. Электронная гарантия размещается в ЕИС в сфере закупок.

Стоимость банковской гарантии составляет 1–5% от суммы гарантии в год
в зависимости от срока, типа гарантии и финансового состояния принципала.
Комиссия уплачивается единовременно при выдаче гарантии.

Электронная банковская гарантия выдаётся за 1 рабочий день по стандартным
тендерным формам. Верификация осуществляется через платформу «Сбербанк Бизнес».
""",
    },
    "doc_leasing": {
        "section": "Лизинг",
        "text": """
Лизинг — финансовый инструмент, при котором лизингодатель приобретает имущество
и передаёт его лизингополучателю в долгосрочную аренду с правом выкупа.

Финансовый лизинг отличается от аренды: по окончании договора лизингополучатель
выкупает оборудование по остаточной стоимости (обычно 1–10% от первоначальной цены).
Имущество в течение срока договора числится на балансе лизингодателя.

Налоговые преимущества лизинга: лизинговые платежи полностью относятся на расходы,
уменьшая налог на прибыль. Ускоренная амортизация с коэффициентом до 3 сокращает
налог на имущество и срок списания актива.

Предметы лизинга: коммерческий транспорт, спецтехника, производственное оборудование,
медицинская техника, недвижимость (лизинг недвижимости). Лизинг автомобилей для
физических лиц (каршеринг-схемы) набирает популярность как альтернатива автокредиту.

Авансовый платёж по лизингу составляет 0–30% от стоимости имущества. Срок договора
лизинга — от 12 до 84 месяцев. Удорожание (переплата) при лизинге составляет
5–15% годовых от стоимости актива.

Оперативный лизинг (операционная аренда) не предполагает перехода права собственности:
по истечении срока имущество возвращается лизингодателю. Применяется для авиатехники,
флота, IT-оборудования с быстрым моральным износом.
""",
    },
    # ── Цифровые сервисы ──────────────────────────────────────────────────────
    "doc_digital_banking": {
        "section": "Цифровые банковские сервисы",
        "text": """
Мобильный банк — приложение для управления счетами, картами, кредитами
и инвестициями со смартфона. Авторизация через биометрию (Face ID, Touch ID)
заменяет ввод пин-кода при входе и подтверждении платежей.

Система быстрых платежей (СБП) позволяет переводить деньги по номеру телефона
между любыми банками России мгновенно и без комиссии до 100 000 руб./мес.
Оплата по QR-коду через СБП заменяет терминал в торговых точках.

Открытый банкинг (Open Banking) — концепция обмена финансовыми данными клиента
между банками через API с согласия самого клиента. Агрегаторы финансов
используют Open Banking для отображения счетов в разных банках в одном интерфейсе.

Цифровой рубль — третья форма российской национальной валюты наряду с наличными
и безналичными деньгами. Эмитируется и хранится на платформе Банка России.
Операции по цифровым рублям доступны через банки-участники пилота с 2024 года.

Антифрод-системы в режиме реального времени анализируют каждую транзакцию
по поведенческим паттернам. Подозрительные операции блокируются и требуют
дополнительного подтверждения через звонок или push-уведомление.

Финансовый маркетплейс позволяет сравнивать и оформлять банковские продукты
разных банков на одной платформе. Маркетплейс «Финуслуги» (Московская биржа)
агрегирует вклады, кредиты и ипотечные предложения всех подключённых банков.
""",
    },
    # ── Forex и валютные операции ─────────────────────────────────────────────
    "doc_forex": {
        "section": "Валютные операции",
        "text": """
Конверсионные операции — сделки по обмену одной валюты на другую. Курс банка
отличается от официального курса ЦБ РФ на величину спреда, который составляет
типично 0,5–2% в зависимости от валютной пары и канала обслуживания.

Валютный счёт открывается в иностранной валюте (USD, EUR, CNY и др.) и используется
для хранения сбережений, проведения международных платежей и конверсионных операций.
Проценты на валютный счёт начисляются по ставке, близкой к нулю.

Обязательная продажа валютной выручки — требование для экспортёров реализовывать
часть поступившей иностранной валюты на внутреннем рынке. Норматив устанавливается
Правительством РФ и Банком России в зависимости от экономических условий.

SWIFT — международная система межбанковских сообщений для трансграничных переводов.
Переводы в долларах США и евро проходят через банки-корреспонденты. Альтернативой
служат системы СПФС (российская) и CIPS (китайская).

Хеджирование валютного риска осуществляется с помощью форвардных контрактов,
валютных опционов и валютных свопов. Форвард фиксирует курс обмена на будущую дату,
защищая от неблагоприятного движения курса.

Валютный контроль — система надзора за законностью валютных операций резидентов
и нерезидентов. При зачислении валютной выручки свыше 1 млн руб. требуется
предоставить контракт-основание. Нарушение валютного законодательства влечёт штраф.
""",
    },
}


def generate_sample_banking_chunks(
    chunker: Optional[DocumentChunker] = None,
) -> List[Chunk]:
    """
    Генерирует список демонстрационных банковских чанков из встроенного корпуса.

    Параметры
    ---------
    chunker : DocumentChunker | None
        Если None — используется DocumentChunker(chunk_size=256, split_by="paragraph")

    Возвращает
    ----------
    List[Chunk]
    """
    if chunker is None:
        chunker = DocumentChunker(chunk_size=256, split_by="paragraph")

    all_chunks: List[Chunk] = []
    for doc_id, doc_info in _BANKING_CORPUS.items():
        chunks = chunker.chunk_document(
            text=doc_info["text"],
            doc_id=doc_id,
            section=doc_info["section"],
        )
        all_chunks.extend(chunks)

    logger.info("Сгенерировано %d чанков из %d документов", len(all_chunks), len(_BANKING_CORPUS))
    return all_chunks
