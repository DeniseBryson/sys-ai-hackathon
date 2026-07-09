from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AgingConfig:
    max_hours: float = 50_000.0
    t90_min_hours: float = 10_000.0
    t90_max_hours: float = 50_000.0
    flicker_probability: float = 0.001
    gaussian_sigma_fraction: float = 0.005
    seed: int = 123


@dataclass(frozen=True)
class AgedRssEpisodes:
    x: np.ndarray
    sample_hours: np.ndarray
    episode_ids: np.ndarray
    episode_hours: np.ndarray
    channel_k: np.ndarray


def age_rss_random_hours(
    x: np.ndarray,
    *,
    config: AgingConfig = AgingConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-LED exponential decay at a random age for every sample.

    k_j is derived from a random per-LED T90 time: exp(-k_j*T90_j)=0.9.
    Returns (aged_x, sample_hours, channel_k).
    """
    if x.ndim != 2:
        raise ValueError("x must be shaped (samples, features)")
    rng = np.random.default_rng(config.seed)
    t90 = rng.uniform(config.t90_min_hours, config.t90_max_hours, size=x.shape[1])
    k = -np.log(0.9) / t90
    hours = rng.uniform(0.0, config.max_hours, size=x.shape[0])
    decay = np.exp(-hours[:, None] * k[None, :]).astype(np.float32)
    aged = x.astype(np.float32) * decay

    if config.flicker_probability > 0:
        flicker = rng.random(aged.shape) < config.flicker_probability
        aged = np.where(flicker, 0.0, aged)

    if config.gaussian_sigma_fraction > 0:
        scale = np.maximum(np.std(x, axis=0, keepdims=True), 1e-8)
        noise = rng.normal(0.0, config.gaussian_sigma_fraction, size=aged.shape) * scale
        aged = aged + noise.astype(np.float32)

    return aged.astype(np.float32), hours.astype(np.float32), k.astype(np.float32)


def age_rss_episodes(
    x: np.ndarray,
    *,
    config: AgingConfig = AgingConfig(),
    episode_count: int = 3,
) -> AgedRssEpisodes:
    """Apply fixed per-LED aging within contiguous evaluation episodes.

    Each run samples one per-LED T90 vector for the physical installation.
    Every episode then represents one deployment/session age applied to that
    same decay profile. Episode ages are sorted from youngest to oldest so the
    streamed order follows increasing installation age. Flicker and additive
    noise remain per-sample perturbations.
    """
    if x.ndim != 2:
        raise ValueError("x must be shaped (samples, features)")
    if episode_count < 1:
        raise ValueError("episode_count must be at least 1")
    if len(x) == 0:
        raise ValueError("x must contain at least one sample")

    rng = np.random.default_rng(config.seed)
    actual_episode_count = min(int(episode_count), len(x))
    episode_ids = (
        np.arange(len(x), dtype=np.int64) * actual_episode_count // len(x)
    )

    t90 = rng.uniform(config.t90_min_hours, config.t90_max_hours, size=x.shape[1])
    k = -np.log(0.9) / t90
    episode_hours = np.sort(
        rng.uniform(0.0, config.max_hours, size=actual_episode_count)
    )
    decay = np.exp(-episode_hours[:, None] * k[None, :]).astype(np.float32)

    aged = x.astype(np.float32).copy()
    for episode_id in range(actual_episode_count):
        aged[episode_ids == episode_id] *= decay[episode_id]

    if config.flicker_probability > 0:
        flicker = rng.random(aged.shape) < config.flicker_probability
        aged = np.where(flicker, 0.0, aged)

    if config.gaussian_sigma_fraction > 0:
        scale = np.maximum(np.std(x, axis=0, keepdims=True), 1e-8)
        noise = rng.normal(0.0, config.gaussian_sigma_fraction, size=aged.shape) * scale
        aged = aged + noise.astype(np.float32)

    sample_hours = episode_hours[episode_ids]
    return AgedRssEpisodes(
        x=aged.astype(np.float32),
        sample_hours=sample_hours.astype(np.float32),
        episode_ids=episode_ids.astype(np.int64),
        episode_hours=episode_hours.astype(np.float32),
        channel_k=k.astype(np.float32),
    )
