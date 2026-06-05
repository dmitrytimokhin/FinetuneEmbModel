"""
trainer.py
==========
Промышленный класс дообучения эмбеддинговой модели deepvk/USER-bge-m3
с поддержкой нескольких функций потерь и автодетекцией CPU / GPU.

Поддерживаемые функции потерь
──────────────────────────────
┌───────────────────┬──────────────────────────────────────────────────────────┐
│ LossType          │ Описание                                                 │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ TRIPLET           │ L = max(0, d(a,p) − d(a,n) + margin)                    │
│                   │ Нужны колонки: anchor, positive, negative               │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ MNRL              │ Contrastive с in-batch negatives (рекомендован)         │
│                   │ Нужны колонки: anchor, positive                         │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ CACHED_MNRL       │ MNRL с кэшированием эмбеддингов — экономит память GPU   │
│                   │ Нужны колонки: anchor, positive                         │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ TSDAE             │ Unsupervised Denoising Auto-Encoder (без разметки)      │
│                   │ Нужна колонка: sentence                                  │
└───────────────────┴──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Отключаем параллелизм токенизатора до форка DataLoader-воркеров,
# иначе HuggingFace tokenizers выдаёт предупреждения о deadlock.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
    TripletLoss,
    TripletDistanceMetric,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Config
# ─────────────────────────────────────────────────────────────────────────────

class LossType(str, Enum):
    TRIPLET = "triplet"
    MNRL = "mnrl"
    CACHED_MNRL = "cached_mnrl"
    TSDAE = "tsdae"


@dataclass
class TrainingConfig:
    """
    Конфигурация обучения с разумными defaults для CPU и GPU.

    CPU-рекомендации:
      batch_size=4, gradient_accumulation_steps=8, fp16=False, max_steps=500

    GPU-рекомендации:
      batch_size=32, gradient_accumulation_steps=1, fp16=True (A100/V100)
    """

    # ── Модель ───────────────────────────────────────────────────────────────
    model_name: str = "deepvk/USER-bge-m3"
    max_seq_length: int = 512

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_type: LossType = LossType.MNRL
    triplet_margin: float = 0.5          # для TripletLoss
    temperature: float = 0.05            # для MNRL / CACHED_MNRL

    # ── Обучение ─────────────────────────────────────────────────────────────
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # ── Precision ────────────────────────────────────────────────────────────
    fp16: bool = False
    bf16: bool = False

    # ── Checkpointing ────────────────────────────────────────────────────────
    output_dir: str = "./output"
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10           # используется только при logging_strategy="steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    # Сколько чекпоинтов хранить на диске (None = все; рекомендуется 2-3)
    # Модель ~2 ГБ × save_total_limit = место на диске
    save_total_limit: int = 2

    # ── Ранняя остановка ─────────────────────────────────────────────────────
    # 0 = выключено; supervised — по eval_loss, unsupervised — по train_loss
    early_stopping_patience: int = 3

    # ── Логирование ──────────────────────────────────────────────────────────
    # "epoch" — каждую эпоху (рекомендуется); "steps" — каждые logging_steps шагов
    logging_strategy: str = "epoch"

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    dataloader_num_workers: int = 0

    # TSDAE-специфично
    tsdae_noise_ratio: float = 0.6

    # ── Частичная разморозка (layer-wise fine-tuning) ─────────────────────────
    # -1 = все слои разморожены (стандартное full fine-tuning, default)
    #  0 = заморожена вся модель, обучается только pooling-слой
    #  N = разморожены последние N encoder-слоёв + pooling
    # Рекомендации: N=2–4 для TSDAE, N=4–6 для supervised fine-tuning
    unfreeze_last_n_layers: int = -1
    # Изменить dropout во последних M encoder-слоях (0 = не менять)
    # Увеличение p (0.1→0.2) добавляет регуляризацию для малых корпусов
    dropout_last_m_layers: int = 0
    dropout_p: float = 0.1

    def auto_adjust_for_device(self, device: str) -> None:
        """Автонастройка гиперпараметров под устройство."""
        if device == "cpu":
            if self.batch_size > 8:
                logger.info("CPU режим: batch_size %d → 4", self.batch_size)
                self.batch_size = 4
            if self.gradient_accumulation_steps < 4:
                logger.info("CPU режим: gradient_accumulation_steps → 8")
                self.gradient_accumulation_steps = 8
            self.fp16 = False
            self.bf16 = False
            self.dataloader_num_workers = 0
        else:
            # GPU: включаем mixed precision если не задано явно
            if not self.fp16 and not self.bf16:
                if torch.cuda.is_bf16_supported():
                    self.bf16 = True
                    logger.info("GPU: включён bf16")
                else:
                    self.fp16 = True
                    logger.info("GPU: включён fp16")


# ─────────────────────────────────────────────────────────────────────────────
# Logging callback
# ─────────────────────────────────────────────────────────────────────────────

class _EpochLogCallback(TrainerCallback):
    """
    Структурированное логирование через Python logger.
    Срабатывает при каждом log-событии трейнера (on_log):
      - train: выводит loss, lr
      - eval:  выводит все eval_* метрики
    С logging_strategy="epoch" вызывается ровно раз в эпоху.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        epoch = state.epoch or 0
        total = int(args.num_train_epochs)

        if "loss" in logs:
            parts = [f"Epoch {epoch:.0f}/{total} [train]",
                     f"loss={logs['loss']:.4f}"]
            if "learning_rate" in logs:
                parts.append(f"lr={logs['learning_rate']:.2e}")
            logger.info(" | ".join(parts))

        elif any(k.startswith("eval_") for k in logs):
            parts = [f"Epoch {epoch:.0f}/{total} [eval]"]
            for k, v in logs.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
            logger.info(" | ".join(parts))


class _EarlyStoppingOnTrainLoss(TrainerCallback):
    """
    Ранняя остановка по train_loss — для unsupervised этапа (без eval_dataset).

    Лучшие веса хранятся в памяти (_best_state) и записываются на диск
    один раз в on_train_end — так же, как ведут себя supervised-методы
    (load_best_model_at_end + save в конце).
    """

    def __init__(self, patience: int, best_model_dir: str) -> None:
        self.patience = patience
        self.best_model_dir = best_model_dir
        self.best_loss = float("inf")
        self.no_improve = 0
        self._best_state: Optional[dict] = None   # веса лучшей эпохи в памяти

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = logs["loss"]
        epoch = state.epoch or 0

        if loss < self.best_loss - 1e-5:
            self.best_loss = loss
            self.no_improve = 0
            # Сохраняем веса в RAM — без записи на диск
            if model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
            logger.info(
                "EarlyStopping [train] epoch=%.0f: loss=%.4f ↓ (best в памяти)",
                epoch, loss,
            )
        else:
            self.no_improve += 1
            logger.info(
                "EarlyStopping [train] epoch=%.0f: loss=%.4f (best=%.4f, no improve %d/%d)",
                epoch, loss, self.best_loss, self.no_improve, self.patience,
            )
            if self.no_improve >= self.patience:
                logger.info(
                    "Ранняя остановка: %d эпох без улучшения train_loss.", self.patience
                )
                control.should_training_stop = True

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Восстанавливаем лучшие веса и сохраняем на диск один раз."""
        if model is None:
            return
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info(
                "Восстановлены лучшие веса (loss=%.4f)", self.best_loss
            )
        model.save(self.best_model_dir)
        logger.info("Лучшая модель сохранена → %s", self.best_model_dir)


# ─────────────────────────────────────────────────────────────────────────────
# TSDAE Dataset Helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(text: str, noise_ratio: float = 0.6) -> str:
    """Простое удаление слов для TSDAE denoising."""
    words = text.split()
    if len(words) <= 1:
        return text
    mask = [random_drop < noise_ratio for random_drop in
            [torch.rand(1).item() for _ in words]]
    kept = [w for w, drop in zip(words, mask) if not drop]
    return " ".join(kept) if kept else words[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingTrainer:
    """
    Обёртка над SentenceTransformerTrainer с поддержкой CPU/GPU,
    нескольких loss функций и удобным API.

    Пример (MNRL, GPU/CPU auto):
    ----------------------------
    >>> config = TrainingConfig(
    ...     model_name="deepvk/USER-bge-m3",
    ...     loss_type=LossType.MNRL,
    ...     num_epochs=3,
    ...     batch_size=16,
    ... )
    >>> trainer = EmbeddingTrainer(config)
    >>> trainer.train(train_df, eval_df)
    >>> trainer.save_model("./my_model")

    Пример (TripletLoss, только CPU):
    ----------------------------------
    >>> config = TrainingConfig(
    ...     loss_type=LossType.TRIPLET,
    ...     batch_size=4,
    ...     num_epochs=5,
    ...     gradient_accumulation_steps=8,
    ... )
    >>> trainer = EmbeddingTrainer(config)
    >>> trainer.train(train_df)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = self._detect_device()
        logger.info("Устройство: %s", self.device)

        config.auto_adjust_for_device(self.device)

        logger.info("Загрузка модели: %s", config.model_name)
        self.model = SentenceTransformer(
            config.model_name,
            device=self.device,
        )
        self.model.max_seq_length = config.max_seq_length
        logger.info(
            "Модель загружена. max_seq_length=%d", config.max_seq_length
        )

        self._apply_layer_config()

    # ── Device detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── Layer-wise freeze / dropout ──────────────────────────────────────────

    def _apply_layer_config(self) -> None:
        """
        Замораживает все слои кроме последних unfreeze_last_n_layers
        и/или меняет dropout в последних dropout_last_m_layers слоях.

        Пропускается (no-op) при unfreeze_last_n_layers=-1 и dropout_last_m_layers=0.

        Схема заморозки
        ---------------
        -1  → full fine-tuning (ничего не меняем)
         0  → заморожен весь encoder; обучается только pooling
         N  → заморожены слои 0...(total-N-1); разморожены последние N + pooling

        Метод универсален: работает для любой HuggingFace-модели с
        auto_model.encoder.layer (BERT, RoBERTa, XLM-RoBERTa, ...).
        """
        cfg = self.config
        if cfg.unfreeze_last_n_layers == -1 and cfg.dropout_last_m_layers == 0:
            return  # full fine-tuning — ничего не делаем

        # Transformer-модуль — первый элемент SentenceTransformer
        transformer_module = self.model[0]
        auto_model = transformer_module.auto_model

        try:
            encoder_layers = list(auto_model.encoder.layer)
        except AttributeError:
            logger.warning(
                "Не удалось получить encoder.layer — "
                "unfreeze/dropout config пропускается"
            )
            return

        n_layers = len(encoder_layers)

        # ── Заморозка ─────────────────────────────────────────────────────────
        if cfg.unfreeze_last_n_layers >= 0:
            # Шаг 1: заморозить всё
            for param in self.model.parameters():
                param.requires_grad = False

            # Шаг 2: разморозить последние N encoder-слоёв
            unfreeze_from = max(0, n_layers - cfg.unfreeze_last_n_layers)
            for layer in encoder_layers[unfreeze_from:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Шаг 3: разморозить pooling (model[1]) — нужен всегда
            if len(self.model) > 1:
                for param in self.model[1].parameters():
                    param.requires_grad = True

            total     = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
            logger.info(
                "Layer freeze: разморожены последние %d/%d слоёв + pooling. "
                "Обучаемых параметров: %s / %s (%.1f%%)",
                cfg.unfreeze_last_n_layers, n_layers,
                f"{trainable:,}", f"{total:,}",
                100.0 * trainable / max(total, 1),
            )

        # ── Dropout ───────────────────────────────────────────────────────────
        if cfg.dropout_last_m_layers > 0:
            import torch.nn as nn
            dropout_from = max(0, n_layers - cfg.dropout_last_m_layers)
            changed = 0
            for layer in encoder_layers[dropout_from:]:
                for module in layer.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = cfg.dropout_p
                        changed += 1
            logger.info(
                "Dropout: изменено %d модулей в последних %d/%d слоях → p=%.2f",
                changed, cfg.dropout_last_m_layers, n_layers, cfg.dropout_p,
            )

    # ── Dataset preparation ──────────────────────────────────────────────────

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Конвертирует DataFrame в HuggingFace Dataset для нужного типа loss.

        Для TRIPLET/MNRL/CACHED_MNRL ожидаются колонки anchor, positive, [negative].
        Для TSDAE ожидается колонка sentence (или anchor).
        """
        loss_type = self.config.loss_type

        if loss_type == LossType.TSDAE:
            # TSDAE / SimCSE-стиль: пары (corrupted → anchor, original → positive).
            # Колонки anchor/positive совместимы и с DenoisingAutoEncoderLoss
            # (порядок: noisy→encoder, clean→decoder), и с MNRL-fallback.
            col = "sentence" if "sentence" in df.columns else "anchor"
            sentences = df[col].tolist()
            noised = [_add_noise(s, self.config.tsdae_noise_ratio) for s in sentences]
            return Dataset.from_dict({
                "anchor": noised,
                "positive": sentences,
            })

        elif loss_type == LossType.TRIPLET:
            required = {"anchor", "positive", "negative"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"TripletLoss требует колонки: {missing}")
            return Dataset.from_dict({
                "anchor": df["anchor"].tolist(),
                "positive": df["positive"].tolist(),
                "negative": df["negative"].tolist(),
            })

        else:  # MNRL / CACHED_MNRL
            required = {"anchor", "positive"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"MNRL требует колонки: {missing}")

            data: Dict[str, List[str]] = {
                "anchor": df["anchor"].tolist(),
                "positive": df["positive"].tolist(),
            }
            # Если есть колонка negative — добавляем как hard negative
            if "negative" in df.columns:
                data["negative"] = df["negative"].tolist()
            return Dataset.from_dict(data)

    # ── Loss construction ────────────────────────────────────────────────────

    def _build_loss(self, train_dataset: Dataset):
        """Создаёт объект loss функции."""
        loss_type = self.config.loss_type

        if loss_type == LossType.TRIPLET:
            return TripletLoss(
                model=self.model,
                distance_metric=TripletDistanceMetric.COSINE,
                triplet_margin=self.config.triplet_margin,
            )

        elif loss_type == LossType.MNRL:
            from sentence_transformers import util as st_util
            return MultipleNegativesRankingLoss(
                model=self.model,
                scale=1.0 / self.config.temperature,
                similarity_fct=st_util.cos_sim,
            )

        elif loss_type == LossType.CACHED_MNRL:
            return CachedMultipleNegativesRankingLoss(
                model=self.model,
                scale=1.0 / self.config.temperature,
                mini_batch_size=self.config.batch_size,
            )

        elif loss_type == LossType.TSDAE:
            # USER-bge-m3 — encoder-only (XLM-RoBERTa).
            # DenoisingAutoEncoderLoss создаёт decoder с ~200 рандомными тензорами,
            # большинство градиентов уходит в decoder, а не в encoder-эмбеддинги.
            # Эффективнее: MNRL на парах (noised → clean) — SimCSE-стиль,
            # напрямую обучает encoder-пространство без seq2seq overhead.
            from sentence_transformers import util as st_util
            logger.info(
                "TSDAE: SimCSE-MNRL на парах (noised, original) "
                "— encoder-only режим, без seq2seq decoder"
            )
            return MultipleNegativesRankingLoss(
                model=self.model,
                scale=1.0 / self.config.temperature,
                similarity_fct=st_util.cos_sim,
            )

        raise ValueError(f"Неизвестный тип loss: {loss_type}")

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Запускает обучение.

        Параметры
        ---------
        train_df : DataFrame с колонками anchor/positive/[negative]
        eval_df  : опциональный eval DataFrame (та же схема)
        """
        logger.info(
            "Подготовка датасета: %d обучающих строк, loss=%s",
            len(train_df),
            self.config.loss_type,
        )

        train_dataset = self.prepare_dataset(train_df)
        eval_dataset = self.prepare_dataset(eval_df) if eval_df is not None else None

        loss = self._build_loss(train_dataset)

        # Определяем BatchSampler
        if self.config.loss_type in (LossType.MNRL, LossType.CACHED_MNRL):
            batch_sampler = BatchSamplers.NO_DUPLICATES
        else:
            batch_sampler = BatchSamplers.BATCH_SAMPLER

        has_eval = eval_dataset is not None

        # ── Имя директории лучшей модели (по типу loss) ──────────────────────
        _loss_dir_name = {
            LossType.TSDAE:        "best_unsupervised",
            LossType.TRIPLET:      "best_triplet",
            LossType.MNRL:         "best_mnrl",
            LossType.CACHED_MNRL:  "best_mnrl",
        }.get(self.config.loss_type, f"best_{self.config.loss_type.value}")
        best_model_dir = os.path.join(self.config.output_dir, _loss_dir_name)

        # ── Callbacks ────────────────────────────────────────────────────────
        callbacks = [_EpochLogCallback()]
        early_stop_best_dir: Optional[str] = None

        if self.config.early_stopping_patience > 0:
            if has_eval:
                # Supervised: встроенный EarlyStoppingCallback по eval_loss
                from transformers import EarlyStoppingCallback as _ESC
                callbacks.append(_ESC(
                    early_stopping_patience=self.config.early_stopping_patience,
                ))
                logger.info(
                    "EarlyStopping (eval_loss): patience=%d, best→%s",
                    self.config.early_stopping_patience,
                    best_model_dir,
                )
            else:
                # Unsupervised (TSDAE): ранняя остановка по train_loss
                early_stop_best_dir = best_model_dir
                callbacks.append(_EarlyStoppingOnTrainLoss(
                    patience=self.config.early_stopping_patience,
                    best_model_dir=early_stop_best_dir,
                ))
                logger.info(
                    "EarlyStopping (train_loss): patience=%d, best→%s",
                    self.config.early_stopping_patience,
                    early_stop_best_dir,
                )

        # ── Стратегии eval / save / log ──────────────────────────────────────
        # Оба параметра должны совпадать для load_best_model_at_end
        eval_strategy_val = "epoch" if has_eval else "no"
        save_strategy_val = "epoch"
        load_best = self.config.load_best_model_at_end and has_eval

        # warmup_steps должен быть int — вычисляем из ratio
        steps_per_epoch = max(1, len(train_dataset) // self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        training_args = SentenceTransformerTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            eval_strategy=eval_strategy_val,
            save_strategy=save_strategy_val,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=load_best,
            metric_for_best_model=self.config.metric_for_best_model,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            batch_sampler=batch_sampler,
            save_total_limit=self.config.save_total_limit,
            report_to="none",
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            callbacks=callbacks,
        )

        logger.info(
            "Старт обучения: epochs=%d, batch=%d, lr=%s, device=%s, "
            "early_stop_patience=%d, logging=%s",
            self.config.num_epochs,
            self.config.batch_size,
            self.config.learning_rate,
            self.device,
            self.config.early_stopping_patience,
            self.config.logging_strategy,
        )
        trainer.train()
        logger.info("Обучение завершено.")

        if early_stop_best_dir and Path(early_stop_best_dir).exists():
            # Unsupervised: callback уже сохранил лучшую модель — загружаем
            self.load_model(early_stop_best_dir)
            logger.info("Загружена лучшая модель (train_loss) из: %s", early_stop_best_dir)
        elif has_eval:
            # Supervised: load_best_model_at_end=True уже загрузил лучшие веса в trainer.model,
            # явно сохраняем в именованную директорию
            self.save_model(best_model_dir)
            logger.info("Лучшая модель (eval_loss) сохранена в: %s", best_model_dir)

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Сохраняет модель в формате sentence-transformers."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Модель сохранена: %s", path)

    def load_model(self, path: str) -> None:
        """Загружает дообученную модель."""
        self.model = SentenceTransformer(path, device=self.device)
        self.model.max_seq_length = self.config.max_seq_length
        logger.info("Модель загружена из: %s", path)

    def get_model(self) -> SentenceTransformer:
        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cpu_config(
    loss_type: LossType = LossType.MNRL,
    num_epochs: int = 3,
    output_dir: str = "./output_cpu",
) -> TrainingConfig:
    """Конфигурация для CPU с малыми ресурсами."""
    return TrainingConfig(
        model_name="deepvk/USER-bge-m3",
        loss_type=loss_type,
        num_epochs=num_epochs,
        batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=False,
        output_dir=output_dir,
        eval_steps=50,
        save_steps=100,
        logging_steps=5,
        dataloader_num_workers=0,
    )


def make_gpu_config(
    loss_type: LossType = LossType.CACHED_MNRL,
    num_epochs: int = 5,
    output_dir: str = "./output_gpu",
) -> TrainingConfig:
    """Конфигурация для GPU (A100/V100/T4)."""
    return TrainingConfig(
        model_name="deepvk/USER-bge-m3",
        loss_type=loss_type,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        fp16=False,       # auto_adjust_for_device включит bf16/fp16
        bf16=False,
        output_dir=output_dir,
        eval_steps=100,
        save_steps=200,
        logging_steps=20,
        dataloader_num_workers=4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reranker (Cross-Encoder) — Config & Trainer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RerankerConfig:
    """
    Конфигурация дообучения CrossEncoder-реранкера.

    Из триплетов (anchor, positive, negative) автоматически формируются пары:
        (anchor, positive) → label = 1.0
        (anchor, negative) → label = 0.0

    CPU-рекомендации:
      batch_size=4, gradient_accumulation_steps=8, num_epochs=3
    GPU-рекомендации:
      batch_size=16, gradient_accumulation_steps=1, fp16=True
    """

    # ── Модель ───────────────────────────────────────────────────────────────
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_seq_length: int = 512

    # ── Обучение ─────────────────────────────────────────────────────────────
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # ── Precision ────────────────────────────────────────────────────────────
    fp16: bool = False
    bf16: bool = False

    # ── Checkpointing ────────────────────────────────────────────────────────
    output_dir: str = "./output"
    save_total_limit: int = 2
    logging_steps: int = 10
    logging_strategy: str = "epoch"

    # ── Ранняя остановка ─────────────────────────────────────────────────────
    early_stopping_patience: int = 3

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    dataloader_num_workers: int = 0

    # ── Частичная разморозка (layer-wise fine-tuning) ─────────────────────────
    # Рекомендации для bge-reranker-v2-m3 (24 слоя XLM-RoBERTa-large):
    #   лёгкое: N=4,  умеренное: N=6,  агрессивное: N=8
    unfreeze_last_n_layers: int = -1
    dropout_last_m_layers: int = 0
    dropout_p: float = 0.1

    def auto_adjust_for_device(self, device: str) -> None:
        if device == "cpu":
            if self.batch_size > 8:
                logger.info("CPU режим: batch_size %d → 4", self.batch_size)
                self.batch_size = 4
            if self.gradient_accumulation_steps < 4:
                self.gradient_accumulation_steps = 8
            self.fp16 = False
            self.bf16 = False
            self.dataloader_num_workers = 0
        else:
            if not self.fp16 and not self.bf16:
                if torch.cuda.is_bf16_supported():
                    self.bf16 = True
                    logger.info("GPU: включён bf16")
                else:
                    self.fp16 = True
                    logger.info("GPU: включён fp16")


class RerankerTrainer:
    """
    Дообучение CrossEncoder-реранкера (sentence-transformers CrossEncoder).

    Поддерживает два бэкенда:
    - ST >= 3.x: CrossEncoderTrainer (предпочтительный)
    - Fallback:  CrossEncoder.fit()  (старый API, работает всегда)

    Пример
    ------
    >>> cfg = RerankerConfig(
    ...     model_name="BAAI/bge-reranker-v2-m3",
    ...     learning_rate=2e-5,
    ...     unfreeze_last_n_layers=6,
    ... )
    >>> trainer = RerankerTrainer(cfg)
    >>> trainer.train(train_df, eval_df)
    >>> trainer.save_model("./output/best_reranker")
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.device = EmbeddingTrainer._detect_device()
        logger.info("Устройство: %s", self.device)

        config.auto_adjust_for_device(self.device)

        from sentence_transformers import CrossEncoder
        logger.info("Загрузка реранкера: %s", config.model_name)
        self.model = CrossEncoder(
            config.model_name,
            max_length=config.max_seq_length,
            device=self.device,
        )
        logger.info("Реранкер загружен.")

        self._apply_layer_config()

    # ── Layer-wise freeze / dropout ──────────────────────────────────────────

    def _apply_layer_config(self) -> None:
        """
        Заморозка encoder-слоёв и изменение dropout.
        CrossEncoder хранит HF-модель в self.model.model.
        """
        cfg = self.config
        if cfg.unfreeze_last_n_layers == -1 and cfg.dropout_last_m_layers == 0:
            return

        hf_model = self.model.model

        try:
            encoder_layers = list(hf_model.encoder.layer)
        except AttributeError:
            logger.warning(
                "Не удалось получить encoder.layer у реранкера — layer config пропускается"
            )
            return

        n_layers = len(encoder_layers)

        if cfg.unfreeze_last_n_layers >= 0:
            for param in hf_model.parameters():
                param.requires_grad = False

            unfreeze_from = max(0, n_layers - cfg.unfreeze_last_n_layers)
            for layer in encoder_layers[unfreeze_from:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Classifier head — всегда разморожен (иначе нечего обучать)
            for name, module in hf_model.named_children():
                if name not in ("embeddings", "encoder", "pooler"):
                    for param in module.parameters():
                        param.requires_grad = True

            total     = sum(p.numel() for p in hf_model.parameters())
            trainable = sum(p.numel() for p in hf_model.parameters()
                            if p.requires_grad)
            logger.info(
                "Reranker freeze: разморожены последние %d/%d слоёв + head. "
                "Обучаемых: %s / %s (%.1f%%)",
                cfg.unfreeze_last_n_layers, n_layers,
                f"{trainable:,}", f"{total:,}",
                100.0 * trainable / max(total, 1),
            )

        if cfg.dropout_last_m_layers > 0:
            import torch.nn as nn
            dropout_from = max(0, n_layers - cfg.dropout_last_m_layers)
            changed = 0
            for layer in encoder_layers[dropout_from:]:
                for module in layer.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = cfg.dropout_p
                        changed += 1
            logger.info(
                "Dropout: изменено %d модулей в последних %d/%d слоях → p=%.2f",
                changed, cfg.dropout_last_m_layers, n_layers, cfg.dropout_p,
            )

    # ── Dataset ──────────────────────────────────────────────────────────────

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Конвертирует DataFrame с триплетами → Dataset для CrossEncoderTrainer.

        Из каждой строки (anchor, positive, negative) создаём 2 примера:
            (anchor, positive) → label = 1.0
            (anchor, negative) → label = 0.0
        """
        s1, s2, labels = [], [], []
        for _, row in df.iterrows():
            s1.append(str(row["anchor"]))
            s2.append(str(row["positive"]))
            labels.append(1.0)

            s1.append(str(row["anchor"]))
            s2.append(str(row["negative"]))
            labels.append(0.0)

        return Dataset.from_dict({"sentence_A": s1, "sentence_B": s2, "label": labels})

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _build_loss(self):
        """BinaryCrossEntropyLoss если доступен, иначе None (авто-выбор трейнером)."""
        try:
            from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
            return BinaryCrossEntropyLoss(model=self.model)
        except ImportError:
            return None

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Запускает дообучение реранкера (ST 3.x → fallback на .fit())."""
        logger.info(
            "Подготовка датасета: %d триплетов → %d пар",
            len(train_df), len(train_df) * 2,
        )

        train_dataset = self.prepare_dataset(train_df)
        eval_dataset  = self.prepare_dataset(eval_df) if eval_df is not None else None
        has_eval      = eval_dataset is not None

        steps_per_epoch = max(1, len(train_dataset) // self.config.batch_size)
        warmup_steps    = max(1, int(
            steps_per_epoch * self.config.num_epochs * self.config.warmup_ratio
        ))

        best_model_dir = os.path.join(self.config.output_dir, "best_reranker")

        callbacks: List = [_EpochLogCallback()]
        if self.config.early_stopping_patience > 0:
            if has_eval:
                from transformers import EarlyStoppingCallback as _ESC
                callbacks.append(_ESC(
                    early_stopping_patience=self.config.early_stopping_patience,
                ))
            else:
                callbacks.append(_EarlyStoppingOnTrainLoss(
                    patience=self.config.early_stopping_patience,
                    best_model_dir=best_model_dir,
                ))

        try:
            self._train_v3(
                train_dataset, eval_dataset, warmup_steps,
                has_eval, best_model_dir, callbacks,
            )
        except (ImportError, TypeError, AttributeError) as exc:
            logger.warning(
                "CrossEncoderTrainer недоступен (%s) → fallback на CrossEncoder.fit()", exc
            )
            self._train_legacy(train_df, eval_df, warmup_steps, best_model_dir)

    def _train_v3(
        self,
        train_dataset,
        eval_dataset,
        warmup_steps: int,
        has_eval: bool,
        best_model_dir: str,
        callbacks: List,
    ) -> None:
        """Обучение через CrossEncoderTrainer (ST >= 3.0)."""
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        from sentence_transformers.cross_encoder.training_args import (
            CrossEncoderTrainingArguments,
        )

        loss = self._build_loss()

        args = CrossEncoderTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            eval_strategy="epoch" if has_eval else "no",
            save_strategy="epoch",
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=has_eval,
            metric_for_best_model="eval_loss",
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to="none",
        )

        trainer_kwargs = dict(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        if loss is not None:
            trainer_kwargs["loss"] = loss

        trainer = CrossEncoderTrainer(**trainer_kwargs)
        trainer.train()
        logger.info("Reranker обучение завершено.")

        if has_eval:
            self.save_model(best_model_dir)
            logger.info("Лучший реранкер сохранён → %s", best_model_dir)

    def _train_legacy(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame],
        warmup_steps: int,
        best_model_dir: str,
    ) -> None:
        """Fallback: CrossEncoder.fit() (ST < 3.0 или ошибка импорта)."""
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        def _to_examples(df: pd.DataFrame) -> List:
            examples = []
            for _, row in df.iterrows():
                examples.append(InputExample(texts=[row["anchor"], row["positive"]], label=1.0))
                examples.append(InputExample(texts=[row["anchor"], row["negative"]], label=0.0))
            return examples

        train_dl = DataLoader(
            _to_examples(train_df),
            shuffle=True,
            batch_size=self.config.batch_size,
        )

        evaluator = None
        if eval_df is not None:
            try:
                from sentence_transformers.cross_encoder.evaluation import (
                    CEBinaryClassificationEvaluator,
                )
                evaluator = CEBinaryClassificationEvaluator.from_input_examples(
                    _to_examples(eval_df)
                )
            except Exception:
                pass

        Path(best_model_dir).mkdir(parents=True, exist_ok=True)
        self.model.fit(
            train_dataloader=train_dl,
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.config.learning_rate},
            output_path=best_model_dir,
            save_best_model=True,
        )
        logger.info("Reranker (legacy) обучение завершено → %s", best_model_dir)

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Реранкер сохранён: %s", path)

    def load_model(self, path: str) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(
            path, max_length=self.config.max_seq_length, device=self.device
        )
        logger.info("Реранкер загружен из: %s", path)

    def get_model(self):
        return self.model


def make_reranker_cpu_config(
    num_epochs: int = 3,
    output_dir: str = "./output_cpu",
) -> RerankerConfig:
    """Конфигурация реранкера для CPU."""
    return RerankerConfig(
        num_epochs=num_epochs,
        batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=False,
        output_dir=output_dir,
    )


def make_reranker_gpu_config(
    num_epochs: int = 3,
    output_dir: str = "./output_gpu",
) -> RerankerConfig:
    """Конфигурация реранкера для GPU (A100/V100/T4)."""
    return RerankerConfig(
        num_epochs=num_epochs,
        batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        fp16=False,
        bf16=False,
        output_dir=output_dir,
        dataloader_num_workers=4,
    )
