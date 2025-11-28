"""收益分布生成模块。

提供统一的接口来生成不同分布的资产收益。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def generate_returns(
    dist_name: str,
    size: int | tuple[int, ...],
    params: dict,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """生成指定分布的资产收益。

    Args:
        dist_name: 分布名称，当前支持 "normal"、"student_t"、"empirical_bootstrap"
        size: 生成样本的数量，可以是标量或元组（如 (trials, assets)）
        params: 分布参数字典
            - 对于 "normal": 需要 "mean" 和 "vol" 键
        rng: 可选的随机数生成器。如果为 None，使用全局 np.random

    Returns:
        生成的收益数组

    Raises:
        ValueError: 当分布名称不支持或参数缺失时
    """
    if dist_name == "normal":
        if "mean" not in params or "vol" not in params:
            raise ValueError('normal 分布需要 "mean" 和 "vol" 参数')

        mean = params["mean"]
        vol = params["vol"]

        if rng is not None:
            return rng.normal(loc=mean, scale=vol, size=size)
        return np.random.normal(loc=mean, scale=vol, size=size)

    if dist_name == "student_t":
        df = float(params.get("df", 5.0))
        if df <= 0:
            raise ValueError("student_t 分布的 df 必须为正数")
        scale = float(params.get("scale", params.get("vol", 0.02)))
        mean = float(params.get("mean", 0.0))
        generator = rng or np.random.default_rng()
        z = generator.standard_normal(size=size)
        chi2 = generator.chisquare(df, size=size)
        t_samples = z / np.sqrt(chi2 / df)
        return mean + scale * t_samples

    if dist_name == "empirical_bootstrap":
        if "historical_returns" not in params:
            raise ValueError('empirical_bootstrap 分布需要 "historical_returns" 参数')
        hist = np.asarray(params["historical_returns"], dtype=float)
        if hist.size == 0:
            raise ValueError("historical_returns 不能为空")
        generator = rng or np.random.default_rng()
        target_shape = size if isinstance(size, tuple) else (size,)
        num_samples = int(np.prod(target_shape))
        idx = generator.integers(0, hist.size, size=num_samples)
        samples = hist[idx]
        return samples.reshape(target_shape)

    raise ValueError(f"不支持的分布类型: {dist_name}")

