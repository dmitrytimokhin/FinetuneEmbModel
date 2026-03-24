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

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

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
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    dataloader_num_workers: int = 0

    # Логирование: каждые N эпох (если задано — переопределяет logging_steps)
    log_every_n_epochs: Optional[int] = None

    # TSDAE-специфично
    tsdae_noise_ratio: float = 0.6

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

    # ── Device detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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
            # Пытаемся использовать DenoisingAutoEncoderLoss (sentence-transformers
            # + transformers < 5.0.0). Если недоступен или несовместим —
            # переходим на MNRL по парам (noised, original): это SimCSE-стиль,
            # дающий аналогичный эффект доменной адаптации без seq2seq-архитектуры.
            try:
                from sentence_transformers.losses import DenoisingAutoEncoderLoss
                loss = DenoisingAutoEncoderLoss(
                    model=self.model,
                    tie_encoder_decoder=True,
                )
                logger.info("TSDAE: используется DenoisingAutoEncoderLoss")
                return loss
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "DenoisingAutoEncoderLoss недоступен (%s). "
                    "Переключаемся на MNRL(noised, original) — SimCSE-стиль доменной адаптации.",
                    exc,
                )
                return MultipleNegativesRankingLoss(
                    model=self.model,
                    scale=1.0 / self.config.temperature,
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

        # Вычисляем logging_steps: если задан log_every_n_epochs — пересчитываем из эпох
        if self.config.log_every_n_epochs is not None:
            steps_per_epoch = max(
                1, math.ceil(len(train_dataset) / self.config.batch_size)
            )
            effective_logging_steps = steps_per_epoch * self.config.log_every_n_epochs
            logger.info(
                "logging_steps=%d (log_every_n_epochs=%d × steps_per_epoch=%d)",
                effective_logging_steps,
                self.config.log_every_n_epochs,
                steps_per_epoch,
            )
        else:
            effective_logging_steps = self.config.logging_steps

        training_args = SentenceTransformerTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_steps=effective_logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end and eval_dataset is not None,
            metric_for_best_model=self.config.metric_for_best_model,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            batch_sampler=batch_sampler,
            report_to="none",  # отключаем W&B/TensorBoard по умолчанию
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        logger.info(
            "Старт обучения: epochs=%d, batch=%d, lr=%s, device=%s",
            self.config.num_epochs,
            self.config.batch_size,
            self.config.learning_rate,
            self.device,
        )
        trainer.train()
        logger.info("Обучение завершено.")

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
