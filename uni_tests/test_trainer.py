from __future__ import annotations

import pytest

from src.training.trainer import (
    _cosine_learning_rate_schedule,
    _epoch_table_header,
    _format_epoch_row,
    _init_reduce_on_plateau_state,
    _maybe_update_plateau_scheduler,
    _scheduled_learning_rate,
    _should_early_stop,
)


def test_reduce_on_plateau_scheduler_keeps_linear_warmup():
    scheduler = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 1,
        "threshold": 1.0e-4,
    }
    plateau_state = _init_reduce_on_plateau_state(1.0e-3)

    assert _scheduled_learning_rate(
        step=0,
        total_steps=8,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=plateau_state,
    ) == pytest.approx(5.0e-4)
    assert _scheduled_learning_rate(
        step=1,
        total_steps=8,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=plateau_state,
    ) == pytest.approx(1.0e-3)
    assert _scheduled_learning_rate(
        step=2,
        total_steps=8,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=plateau_state,
    ) == pytest.approx(1.0e-3)


def test_reduce_on_plateau_scheduler_reduces_after_patience():
    scheduler = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 1,
        "threshold": 1.0e-4,
    }
    plateau_state = _init_reduce_on_plateau_state(1.0e-3)
    plateau_state = _maybe_update_plateau_scheduler(
        scheduler=scheduler,
        plateau_state=plateau_state,
        metric=1.0,
        global_step=2,
        warmup_steps=2,
        min_lr=1.0e-5,
    )
    plateau_state = _maybe_update_plateau_scheduler(
        scheduler=scheduler,
        plateau_state=plateau_state,
        metric=1.0,
        global_step=3,
        warmup_steps=2,
        min_lr=1.0e-5,
    )
    assert plateau_state.current_lr == pytest.approx(1.0e-3)

    plateau_state = _maybe_update_plateau_scheduler(
        scheduler=scheduler,
        plateau_state=plateau_state,
        metric=1.0,
        global_step=4,
        warmup_steps=2,
        min_lr=1.0e-5,
    )
    assert plateau_state.current_lr == pytest.approx(5.0e-4)


def test_cosine_scheduler_retains_previous_schedule_shape():
    scheduler = {"name": "cosine"}

    assert _scheduled_learning_rate(
        step=0,
        total_steps=6,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=None,
    ) == pytest.approx(5.0e-4)
    assert _scheduled_learning_rate(
        step=1,
        total_steps=6,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=None,
    ) == pytest.approx(1.0e-3)
    assert _scheduled_learning_rate(
        step=2,
        total_steps=6,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
        scheduler=scheduler,
        plateau_state=None,
    ) == pytest.approx(1.0e-3)
    assert _cosine_learning_rate_schedule(
        step=6,
        total_steps=6,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_steps=2,
    ) == pytest.approx(1.0e-5)


def test_epoch_table_header_uses_plain_column_names():
    header = _epoch_table_header()

    assert "Epoch" in header
    assert "Train Loss" in header
    assert "Val Loss" in header
    assert "LR" in header
    assert "Time" in header
    assert "INFO" not in header


def test_epoch_row_formats_plain_values_without_logger_prefix():
    row = _format_epoch_row(
        epoch=12,
        epochs=300,
        train_loss=1.23e-2,
        val_loss=4.56e-2,
        learning_rate=1.0e-4,
        epoch_seconds=0.071,
    )

    assert "12/300" in row
    assert "1.2300e-02" in row
    assert "4.5600e-02" in row
    assert "1.0000e-04" in row
    assert "0.071s" in row
    assert "INFO" not in row
    assert "src.training.trainer" not in row


def test_early_stopping_triggers_after_thirty_non_improving_epochs():
    assert _should_early_stop(29) is False
    assert _should_early_stop(30) is True
