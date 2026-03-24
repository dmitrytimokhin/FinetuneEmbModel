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

        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: scores[i],
            reverse=True,
        )
        result: List[Chunk] = []
        for idx in ranked:
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
    ) -> None:
        if not chunks:
            raise ValueError("Список чанков не может быть пустым")

        self.chunks = chunks
        self.negative_strategy = negative_strategy
        self.n_hard_neg_pool = n_hard_neg_pool
        self.seed = seed
        random.seed(seed)

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

    def sample_query_template(self, n_per_chunk: int = 1) -> List[Triplet]:
        """
        Anchor = синтетический запрос по шаблону.
        Positive = исходный чанк.
        Negative = BM25-hard из другого документа.
        """
        triplets: List[Triplet] = []

        for chunk in tqdm(self.chunks, desc="query_template", leave=False):
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
                all_triplets.extend(self.sample_query_template(n_per_chunk))
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
""",
    },
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
вкладчика в одном банке.

Виды вкладов: вклад «до востребования» — средства можно снять в любой момент;
срочный вклад — фиксированный срок и ставка; накопительный вклад — позволяет
пополнять счёт в течение срока хранения.

Капитализация процентов означает начисление процентов на уже начисленные
проценты. Вклады с капитализацией обеспечивают более высокую эффективную
доходность по сравнению со вкладами с выплатой процентов на отдельный счёт.
""",
    },
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
до 1 000 рублей ввод PIN-кода не требуется.

Лимиты на снятие наличных устанавливаются банком и зависят от типа карты
и тарифного плана. Стандартный дневной лимит снятия составляет 100 000–300 000 рублей.

3D Secure — дополнительный уровень защиты при онлайн-платежах. При совершении
операции в интернете на телефон клиента отправляется одноразовый пароль (OTP).
""",
    },
    "doc_loans": {
        "section": "Потребительские кредиты",
        "text": """
Потребительский кредит — банковский продукт для финансирования личных нужд
граждан: покупки товаров, оплаты услуг, ремонта и т.д.

Требования к заёмщику: гражданство РФ, возраст от 18 до 70 лет, постоянная
регистрация, стаж на текущем месте работы не менее 3 месяцев, подтверждённый
доход. Хорошая кредитная история повышает вероятность одобрения.

Полная стоимость кредита (ПСК) — показатель, отражающий все расходы заёмщика
по кредиту в процентах годовых. ПСК включает процентную ставку, комиссии
и стоимость обязательного страхования.

Аннуитетный платёж — равные ежемесячные платежи на протяжении всего срока
кредита. Дифференцированный платёж уменьшается каждый месяц, так как основной
долг погашается равными долями.

Кредитная история формируется в Бюро кредитных историй (БКИ) и отражает
все кредитные обязательства гражданина. Просрочки платежей негативно
влияют на кредитный рейтинг и условия будущих кредитов.
""",
    },
    "doc_rko": {
        "section": "Расчётно-кассовое обслуживание",
        "text": """
Расчётно-кассовое обслуживание (РКО) — комплекс банковских услуг для юридических
лиц и индивидуальных предпринимателей по ведению расчётного счёта.

Открытие расчётного счёта ИП и ООО осуществляется на основании пакета документов:
паспорт, ОГРН/ОГРНИП, ИНН, устав (для ООО), решение об учреждении общества.

Интернет-банк для бизнеса предоставляет возможность управлять счётом онлайн:
проводить платежи, формировать выписки, подключать дополнительные сервисы.

Эквайринг — услуга приёма безналичной оплаты через POS-терминалы или интернет.
Торговый эквайринг подключается к расчётному счёту предприятия.

Зарплатный проект позволяет автоматизировать выплату заработной платы
сотрудникам через систему банка. Реестр платежей загружается одним файлом,
а средства зачисляются на карты сотрудников в тот же день.
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
