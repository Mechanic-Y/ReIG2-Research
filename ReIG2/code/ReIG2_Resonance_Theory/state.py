"""
reig2.state — 状態空間定義
===========================
Definition 2.1: Ψ_i := (z_i, W_i, g_i, θ_i)

  z_i  : 表出状態 (振幅・位相・出力特性)         — complex vector
  W_i  : 内部結合 (重み・連想構造)               — matrix
  g_i  : 内界計量 (情報幾何的距離 / Fisher計量)  — symmetric positive-definite matrix
  θ_i  : 閾値・文脈・信念パラメータ              — real vector
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubjectState:
    """
    主体状態 Ψ_i ∈ X
    
    Parameters
    ----------
    z : np.ndarray, shape (d,)
        表出状態 (complex): 振幅 A = |z|, 位相 φ = arg(z)
    W : np.ndarray, shape (d, d)
        内部結合行列 (学習重み / 連想構造)
    g : np.ndarray, shape (p, p)
        内界計量 (Fisher情報行列, symmetric positive-definite)
    theta : np.ndarray, shape (p,)
        閾値・文脈・信念パラメータ
    label : str
        識別ラベル (optional)
    """
    z: np.ndarray
    W: np.ndarray
    g: np.ndarray
    theta: np.ndarray
    label: str = ""

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=complex)
        self.W = np.asarray(self.W, dtype=float)
        self.g = np.asarray(self.g, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)

    # --- 便利プロパティ ---
    @property
    def amplitude(self) -> np.ndarray:
        """振幅 A_i = |z_i|"""
        return np.abs(self.z)

    @property
    def phase(self) -> np.ndarray:
        """位相 φ_i = arg(z_i)"""
        return np.angle(self.z)

    @property
    def dim_z(self) -> int:
        return self.z.shape[0]

    @property
    def dim_theta(self) -> int:
        return self.theta.shape[0]

    def copy(self) -> SubjectState:
        return SubjectState(
            z=self.z.copy(),
            W=self.W.copy(),
            g=self.g.copy(),
            theta=self.theta.copy(),
            label=self.label,
        )

    @staticmethod
    def random(dim_z: int = 4, dim_theta: int = 3,
               label: str = "") -> SubjectState:
        """ランダム初期化された主体状態を生成"""
        z = np.random.randn(dim_z) + 1j * np.random.randn(dim_z)
        z /= np.linalg.norm(z)  # 正規化

        W = np.random.randn(dim_z, dim_z) * 0.1
        W = (W + W.T) / 2  # 対称化

        # 正定値計量の生成
        A = np.random.randn(dim_theta, dim_theta) * 0.3
        g = A @ A.T + np.eye(dim_theta) * 0.1

        theta = np.random.randn(dim_theta) * 0.5

        return SubjectState(z=z, W=W, g=g, theta=theta, label=label)


@dataclass
class Ensemble:
    """
    集合系 Ψ := (Ψ_1, ..., Ψ_N) ∈ X^N
    
    Parameters
    ----------
    states : list[SubjectState]
        主体状態のリスト
    """
    states: list[SubjectState] = field(default_factory=list)

    @property
    def N(self) -> int:
        """主体数"""
        return len(self.states)

    def __getitem__(self, idx: int) -> SubjectState:
        return self.states[idx]

    def __setitem__(self, idx: int, state: SubjectState):
        self.states[idx] = state

    def __iter__(self):
        return iter(self.states)

    def phases(self) -> np.ndarray:
        """全主体の代表位相 (各 z の第0成分の偏角)"""
        return np.array([np.angle(s.z[0]) for s in self.states])

    def order_parameter(self) -> tuple[float, float]:
        """
        秩序パラメータ (Kuramoto型)
        R e^{iΘ} = (1/N) Σ_j exp(i φ_j)
        
        Returns (R, Θ)
        """
        phis = self.phases()
        z_mean = np.mean(np.exp(1j * phis))
        R = float(np.abs(z_mean))
        Theta = float(np.angle(z_mean))
        return R, Theta

    def copy(self) -> Ensemble:
        return Ensemble(states=[s.copy() for s in self.states])

    @staticmethod
    def random(N: int = 10, dim_z: int = 4,
               dim_theta: int = 3) -> Ensemble:
        """N体のランダム集合系を生成"""
        states = [
            SubjectState.random(dim_z, dim_theta, label=f"subject_{i}")
            for i in range(N)
        ]
        return Ensemble(states=states)
