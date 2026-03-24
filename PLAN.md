# План улучшения качества Retriever на банковских документах

## 1. Почему дообучение эмбеддингов улучшает RAG

Стандартные модели (BGE, E5, USER) обучены на **общих** корпусах. Банковская
документация содержит специфическую терминологию («аннуитетный платёж»,
«грейс-период», «ПСК»), аббревиатуры (АСВ, БКИ, РКО) и формулировки,
которых мало в общих данных. Дообучение решает:

- **Domain shift**: модель видела мало финансовых текстов
- **Lexical gap**: пользователь пишет разговорно, документ — официально
- **Retrieval gap**: общие эмбеддинги не разграничивают похожие банковские продукты

---

## 2. Стратегии формирования триплетов

### 2.1 Adjacent chunks (без разметки)
```
anchor   = чанк_i
positive = чанк_{i±1} (тот же документ, соседний абзац)
negative = BM25-hard из другого документа
```
**Гипотеза**: соседние абзацы одного документа семантически близки.
**Плюс**: не нужны лейблы. **Минус**: шум (заголовок может не совпадать с абзацем).

### 2.2 Section-based
```
anchor   = чанк из секции X
positive = другой чанк из той же секции X
negative = чанк из другой секции Y
```
**Плюс**: чёткое понятие "релевантности" через раздел документа.

### 2.3 Query template (pseudo-labeling без LLM)
```
anchor   = шаблонный запрос: "Что такое {тема}?"
positive = исходный чанк
negative = BM25-hard из другого документа
```
**Плюс**: имитирует реальные пользовательские запросы.

### 2.4 Synthetic Q&A через LLM (продвинутый, рекомендован)
Использовать LLM (GPT-4, Qwen, YandexGPT) для генерации:
```python
prompt = f"Сгенерируй 3 вопроса, на которые отвечает следующий текст:\n{chunk}"
```
Даёт наиболее реалистичные anchor-запросы.

### 2.5 Hard negative mining (embedding-based)
После 1-й эпохи заменить BM25-негативы на embedding-hard negatives:
найти топ-K чанков по косинусной близости → взять не-позитивы.

---

## 3. Функции потерь: лучшие практики

### 3.1 TripletLoss
```
L = max(0, d(a,p) − d(a,n) + margin)
```
- `margin=0.5` — стандартный. При большом margin модель тренируется агрессивнее.
- Расстояние: `cosine` предпочтительнее `euclidean` для текстовых эмбеддингов.
- **Проблема**: при случайных негативах большинство триплетов легко и не учат модель.
- **Решение**: Hard/Semi-hard negative sampling.

### 3.2 MultipleNegativesRankingLoss (MNRL) ⭐ РЕКОМЕНДОВАН
```
L = -log( exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, p_j)/τ) )
```
- In-batch negatives: каждый positive_j в батче — негатив для запроса_i.
- При `batch_size=32` → 31 негатив на запрос. При `batch_size=64` → 63.
- `τ (temperature)=0.05` — стандарт для retrieval.
- Не нужны явные негативы (но можно добавить как `hard_negatives`).
- **Эффективнее TripletLoss** при тех же данных.

### 3.3 CachedMultipleNegativesRankingLoss
Аналог MNRL, но кэширует эмбеддинги позитивов в `mini_batch_size`.
Позволяет эффективно работать с `effective_batch_size=256+` на GPU с ограниченной VRAM.

### 3.4 TSDAE (Denoising Auto-Encoder) — без учителя
```
L = -log P(sentence | encoder(corrupted_sentence))
```
- Без разметки, только тексты. Отлично для доменной адаптации.
- `noise_ratio=0.6` — удаляет 60% слов. Модель учится восстанавливать.
- Используется как **этап 1** перед MNRL.

### 3.5 MatryoshkaLoss (опционально, для продакшн)
Оборачивает любой loss — обучает эмбеддинги работать на разных размерностях
(768, 512, 256, 128). Полезно для ускорения retrieval при минимальной потере качества.

---

## 4. Архитектура pipeline дообучения

```
Документы
    │
    ▼
DocumentChunker (paragraph/token-window)
    │
    ▼
TripletSampler
  ├─ BM25HardNegativeMiner
  ├─ adjacent / section / query_template
  └─ → pd.DataFrame(anchor, positive, negative)
    │
    ▼
EmbeddingTrainer
  ├─ TrainingConfig (CPU / GPU auto-detect)
  ├─ LossType: TRIPLET | MNRL | CACHED_MNRL | TSDAE
  ├─ SentenceTransformerTrainer (sentence-transformers v3)
  └─ → SentenceTransformer (сохранение checkpoint)
    │
    ▼
RetrievalEvaluator
  ├─ encode(queries) + encode(corpus)
  ├─ FAISS / numpy index
  ├─ Recall@{1,3,5,10}, MRR@10, NDCG@10, MAP@10
  └─ compare_models() → pd.DataFrame + plots
```

---

## 5. Рекомендованная конфигурация

### CPU (малые ресурсы, долгое обучение)
```python
TrainingConfig(
    model_name="deepvk/USER-bge-m3",
    loss_type=LossType.MNRL,
    batch_size=4,
    gradient_accumulation_steps=8,  # effective_batch_size = 32
    num_epochs=3,
    learning_rate=2e-5,
    fp16=False,
    bf16=False,
)
```

### GPU (A100/V100/T4)
```python
TrainingConfig(
    model_name="deepvk/USER-bge-m3",
    loss_type=LossType.CACHED_MNRL,
    batch_size=32,
    gradient_accumulation_steps=1,
    num_epochs=5,
    learning_rate=2e-5,
    bf16=True,  # A100 — bf16; V100 — fp16
)
```

---

## 6. Метрики для RAG

| Метрика | Формула | Смысл | Приоритет |
|---------|---------|-------|-----------|
| **Recall@k** | hits(k) / total | Нашёл ли хоть раз в top-k | ⭐⭐⭐ |
| **MRR@10** | Σ 1/rank_first | Насколько высоко первый релевантный | ⭐⭐⭐ |
| NDCG@10 | DCG/IDCG | Порядок + релевантность | ⭐⭐ |
| MAP@10 | mean AP | Средняя точность по всем запросам | ⭐⭐ |

**Для RAG**: `Recall@5` и `Recall@10` — самые важные, т.к. LLM читает top-k чанков.

---

## 7. Двухэтапная стратегия (лучшая практика)

```
Этап 1 — Доменная адаптация (без разметки):
  TSDAE на всех банковских текстах → 1-2 эпохи

Этап 2 — Retrieval fine-tuning (с триплетами):
  MNRL на (anchor, positive, hard_negative) → 3-5 эпох
```

Такой подход даёт +5-15% Recall@5 по сравнению с только одним этапом.
