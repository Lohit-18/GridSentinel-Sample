"""Synthetic attack generators — used ONLY to create labelled training/test data.

Every function here operates on the in-memory synthetic simulation. Nothing
targets, probes or transmits to any real device or network. See ``ETHICS.md``.
"""
from __future__ import annotations

from .fdi import naive_fdi, stealth_fdi, ramp_fdi
from .goose_attacks import goose_spoof, goose_replay, goose_flood, GooseMeta
from .scenarios import AttackCampaign, generate_campaign, ATTACK_TYPES

__all__ = [
    "naive_fdi",
    "stealth_fdi",
    "ramp_fdi",
    "goose_spoof",
    "goose_replay",
    "goose_flood",
    "GooseMeta",
    "AttackCampaign",
    "generate_campaign",
    "ATTACK_TYPES",
]
