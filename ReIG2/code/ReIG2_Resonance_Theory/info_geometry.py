"""
reig2.info_geometry — 情報幾何層
=================================
§6: Information Geometry Layer

- Fisher計量 g_ab = E[∂_a log p · ∂_b log p]
- Christoffel接続 Γ^k_ij
- 曲率テンソル R^l_ijk
- 自由エネルギー L_i(θ_i) = D_KL(p_i || q_i) - λ E_i(θ_i)
- 自然勾配更新 θ̇ = -η g(θ)^{-1} ∇_θ L(θ)
- 測地線計算
- 条件付き可逆 (吸引域分類)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable

from .state import SubjectState


class InformationGeometry:
    """
    情報幾何エンジン

    主体の内界を確率多様体 M_i = {p_i(x; θ_i)} として扱い，
    Fisher計量・曲率・測地線を計算する。
    """

    def __init__(self, n_samples: int = 1000):
        """
        Parameters
        ----------
        n_samples : int
            Fisher計量のモンテカルロ推定に使うサンプル数
        """
        self.n_samples = n_samples

    # ═══════════════════════════════════════════
    #  Fisher 情報計量 (§6.2)
    # ═══════════════════════════════════════════
    @staticmethod
    def fisher_metric_gaussian(theta: np.ndarray) -> np.ndarray:
        """
        ガウス分布族 N(μ, σ²) に対する Fisher 情報計量

        θ = (μ, σ) の場合:
        g = [[1/σ², 0], [0, 2/σ²]]
        
        一般の多変量ガウスの場合は theta の後半を分散パラメータとして扱う。
        
        Parameters
        ----------
        theta : np.ndarray, shape (p,)
            パラメータ (前半: 平均, 後半: 分散の対数)
        
        Returns
        -------
        g : np.ndarray, shape (p, p)
            Fisher情報計量行列
        """
        p = len(theta)
        half = p // 2
        if half == 0:
            half = 1

        g = np.zeros((p, p))

        # 平均パラメータに対する Fisher 情報
        sigma_sq = np.exp(theta[half:]) if half < p else np.ones(half)
        for a in range(min(half, p)):
            idx = min(a, len(sigma_sq) - 1)
            g[a, a] = 1.0 / max(sigma_sq[idx], 1e-8)

        # 分散パラメータに対する Fisher 情報
        for a in range(half, p):
            g[a, a] = 2.0 / max(sigma_sq[min(a - half, len(sigma_sq) - 1)], 1e-8)

        return g

    @staticmethod
    def fisher_metric_from_state(state: SubjectState) -> np.ndarray:
        """
        SubjectState の g をそのまま Fisher 計量として返す
        (状態空間定義で g_i が Fisher 計量に対応)
        """
        return state.g.copy()

    # ═══════════════════════════════════════════
    #  Christoffel 接続 (§6.5)
    # ═══════════════════════════════════════════
    @staticmethod
    def christoffel_symbols(
        g_func: Callable[[np.ndarray], np.ndarray],
        theta: np.ndarray,
        h: float = 1e-5,
    ) -> np.ndarray:
        """
        Christoffel 記号 Γ^k_ij の数値計算

        Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

        Parameters
        ----------
        g_func : callable(theta) -> g_matrix
            パラメータから計量行列を返す関数
        theta : np.ndarray, shape (p,)
        h : float — 有限差分のステップ

        Returns
        -------
        Gamma : np.ndarray, shape (p, p, p)
            Gamma[i, j, k] = Γ^k_ij
        """
        p = len(theta)
        g = g_func(theta)
        g_inv = np.linalg.inv(g)

        # 計量の偏微分 ∂_m g_{ab}
        dg = np.zeros((p, p, p))  # dg[m, a, b] = ∂_m g_{ab}
        for m in range(p):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[m] += h
            theta_minus[m] -= h
            dg[m] = (g_func(theta_plus) - g_func(theta_minus)) / (2 * h)

        # Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        Gamma = np.zeros((p, p, p))
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    val = 0.0
                    for l in range(p):
                        val += g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )
                    Gamma[i, j, k] = 0.5 * val

        return Gamma

    # ═══════════════════════════════════════════
    #  曲率テンソル (§6.5)
    # ═══════════════════════════════════════════
    @staticmethod
    def riemann_curvature(
        g_func: Callable[[np.ndarray], np.ndarray],
        theta: np.ndarray,
        h: float = 1e-4,
    ) -> np.ndarray:
        """
        Riemann 曲率テンソル R^l_{ijk} の数値計算

        R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
                  + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}

        Returns
        -------
        R : np.ndarray, shape (p, p, p, p)
            R[i, j, k, l] = R^l_{ijk}
        """
        p = len(theta)
        Gamma = InformationGeometry.christoffel_symbols(g_func, theta, h)

        # Γ の偏微分
        dGamma = np.zeros((p, p, p, p))  # dGamma[m, i, j, k] = ∂_m Γ^k_ij
        for m in range(p):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[m] += h
            theta_minus[m] -= h
            G_plus = InformationGeometry.christoffel_symbols(g_func, theta_plus, h)
            G_minus = InformationGeometry.christoffel_symbols(g_func, theta_minus, h)
            dGamma[m] = (G_plus - G_minus) / (2 * h)

        # R^l_{ijk}
        R = np.zeros((p, p, p, p))
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    for l in range(p):
                        val = dGamma[j, i, k, l] - dGamma[k, i, j, l]
                        for m in range(p):
                            val += Gamma[j, m, l] * Gamma[i, k, m]
                            val -= Gamma[k, m, l] * Gamma[i, j, m]
                        R[i, j, k, l] = val

        return R

    @staticmethod
    def scalar_curvature(
        g_func: Callable[[np.ndarray], np.ndarray],
        theta: np.ndarray,
        h: float = 1e-4,
    ) -> float:
        """
        スカラー曲率 R = g^{ij} R_{ij} の計算

        Returns
        -------
        scalar_R : float
        """
        p = len(theta)
        g = g_func(theta)
        g_inv = np.linalg.inv(g)
        R_tensor = InformationGeometry.riemann_curvature(g_func, theta, h)

        # Ricci テンソル R_{ij} = R^k_{ikj}
        Ricci = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    Ricci[i, j] += R_tensor[i, k, j, k]

        # スカラー曲率
        scalar_R = np.sum(g_inv * Ricci)
        return float(scalar_R)

    # ═══════════════════════════════════════════
    #  自由エネルギー (§6.6)
    # ═══════════════════════════════════════════
    @staticmethod
    def free_energy(
        theta: np.ndarray,
        theta_target: np.ndarray,
        g: np.ndarray,
        resonance_value: float = 0.0,
        lam: float = 1.0,
    ) -> float:
        """
        L_i(θ_i) = D_KL(p_i || q_i) - λ E_i(θ_i)

        近似: ガウス族では D_KL ∝ (θ - θ_target)^T g (θ - θ_target) / 2

        Parameters
        ----------
        theta : np.ndarray — 現在のパラメータ
        theta_target : np.ndarray — 目標パラメータ (q_i に対応)
        g : np.ndarray — Fisher 計量
        resonance_value : float — 共鳴価値 E_i(θ_i)
        lam : float — 共鳴価値の重み λ

        Returns
        -------
        L : float — 自由エネルギー
        """
        diff = theta - theta_target
        kl_approx = 0.5 * diff @ g @ diff
        return float(kl_approx - lam * resonance_value)

    # ═══════════════════════════════════════════
    #  自然勾配更新 (§6.6)
    # ═══════════════════════════════════════════
    @staticmethod
    def natural_gradient_step(
        theta: np.ndarray,
        grad_L: np.ndarray,
        g: np.ndarray,
        eta: float = 0.01,
    ) -> np.ndarray:
        """
        自然勾配更新:
        θ̇ = -η g(θ)^{-1} ∇_θ L(θ)

        Parameters
        ----------
        theta : np.ndarray — 現在のパラメータ
        grad_L : np.ndarray — ∇_θ L(θ)
        g : np.ndarray — Fisher 計量
        eta : float — 学習率

        Returns
        -------
        theta_new : np.ndarray
        """
        g_inv = np.linalg.inv(g + 1e-8 * np.eye(g.shape[0]))
        delta = -eta * g_inv @ grad_L
        return theta + delta

    @staticmethod
    def natural_gradient_update(
        state: SubjectState,
        theta_target: np.ndarray,
        eta: float = 0.01,
        lam: float = 1.0,
        resonance_value: float = 0.0,
    ) -> SubjectState:
        """
        SubjectState に対する自然勾配による一回更新

        Returns
        -------
        updated_state : SubjectState
        """
        g = state.g
        theta = state.theta

        # ∇_θ L ≈ g (θ - θ_target) - λ ∇_θ E
        # 簡易: ∇_θ E ≈ resonance_value の方向
        grad_L = g @ (theta - theta_target) - lam * resonance_value * np.ones_like(theta)

        theta_new = InformationGeometry.natural_gradient_step(
            theta, grad_L, g, eta
        )

        result = state.copy()
        result.theta = theta_new
        return result

    # ═══════════════════════════════════════════
    #  測地線 (§6.4)
    # ═══════════════════════════════════════════
    @staticmethod
    def geodesic(
        g_func: Callable[[np.ndarray], np.ndarray],
        theta_start: np.ndarray,
        velocity: np.ndarray,
        n_steps: int = 100,
        dt: float = 0.01,
    ) -> np.ndarray:
        """
        測地線の数値積分 (Euler法)

        測地線方程式: d²θ^k/dt² + Γ^k_ij (dθ^i/dt)(dθ^j/dt) = 0

        Parameters
        ----------
        g_func : callable(theta) -> g
        theta_start : 初期位置
        velocity : 初期速度
        n_steps : ステップ数
        dt : 時間刻み

        Returns
        -------
        trajectory : np.ndarray, shape (n_steps+1, p)
        """
        p = len(theta_start)
        trajectory = np.zeros((n_steps + 1, p))
        trajectory[0] = theta_start.copy()

        theta = theta_start.copy()
        v = velocity.copy()

        for step in range(n_steps):
            # Christoffel 記号を現在位置で計算
            Gamma = InformationGeometry.christoffel_symbols(g_func, theta)

            # 加速度: a^k = -Γ^k_ij v^i v^j
            a = np.zeros(p)
            for k in range(p):
                for i in range(p):
                    for j in range(p):
                        a[k] -= Gamma[i, j, k] * v[i] * v[j]

            # Euler 更新
            theta = theta + dt * v
            v = v + dt * a
            trajectory[step + 1] = theta.copy()

        return trajectory

    # ═══════════════════════════════════════════
    #  吸引域分類 (§6.6: 条件付き可逆)
    # ═══════════════════════════════════════════
    @staticmethod
    def classify_basin(
        delta_theta: np.ndarray,
        epsilon: float = 0.1,
    ) -> str:
        """
        ‖Δθ‖ < ε → 同じ吸引域 (可逆)
        ‖Δθ‖ ≥ ε → 別の吸引域 (準不可逆)
        """
        norm = float(np.linalg.norm(delta_theta))
        return "reversible" if norm < epsilon else "quasi-irreversible"

    @staticmethod
    def geodesic_distance(
        theta_a: np.ndarray,
        theta_b: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """
        Fisher 計量による近似測地線距離
        ds² = (θ_a - θ_b)^T g (θ_a - θ_b)
        """
        diff = theta_a - theta_b
        return float(np.sqrt(max(diff @ g @ diff, 0.0)))
