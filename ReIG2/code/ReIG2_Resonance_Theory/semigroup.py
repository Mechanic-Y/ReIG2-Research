"""
reig2.semigroup — 共鳴事象作用素半群
======================================
§4: 合成構造 (Resonance Event Semigroup)

𝔑 = (ρ̂ε ∘ Û) ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ

接触 → 協力層生成 → 環境共有 → 整合 → 臨界判定
  → [τ̂=1] → 共感 → 更新 → 可逆制御

特性:
  - 非線形
  - 半群構造 (常に逆を持たない)
  - 条件付き可逆 (ρ̂ε による)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .state import SubjectState, Ensemble
from .operators import (
    ContactOperator,
    CooperationLayerOperator,
    EnvironmentShareOperator,
    AlignmentOperator,
    ThresholdGate,
    EmpathyOperator,
    UpdateOperator,
    RelaxationOperator,
)


@dataclass
class ResonanceEvent:
    """共鳴事象の結果を格納するデータクラス"""
    t: float                        # 時刻
    tau: int                        # 臨界判定 (0 or 1)
    R: float                        # 秩序パラメータ
    Theta: float                    # 集団位相
    R_c: float                      # 臨界閾値
    delta_g_norms: list[float]      # 各主体の ‖Δg‖
    relaxation_modes: list[str]     # 各主体の "relax" / "fix"
    ensemble_before: Ensemble       # 事象前の状態
    ensemble_after: Ensemble        # 事象後の状態
    E: np.ndarray                   # 干渉行列
    Phi: np.ndarray                 # 協力層場

    @property
    def fired(self) -> bool:
        return self.tau == 1


class ResonanceSemigroup:
    """
    共鳴事象作用素半群 𝔑

    𝔑 = (ρ̂ε ∘ Û) ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ

    Parameters
    ----------
    R_c : float
        臨界閾値
    coupling_strength : float
        Kuramoto 結合強度
    epsilon : float
        可逆/固定の閾値
    mixing_rate : float
        共感の視点交換率
    """

    def __init__(
        self,
        R_c: float = 0.7,
        coupling_strength: float = 0.1,
        epsilon: float = 0.1,
        mixing_rate: float = 0.1,
        eta_W: float = 0.05,
        eta_g: float = 0.02,
        eta_theta: float = 0.03,
        relax_rate: float = 0.5,
    ):
        # 8作用素のインスタンス化
        self.C_hat = ContactOperator()
        self.L_hat = CooperationLayerOperator()
        self.E_hat = EnvironmentShareOperator()
        self.A_hat = AlignmentOperator(coupling_strength=coupling_strength)
        self.tau_hat = ThresholdGate(R_c=R_c)
        self.M_hat = EmpathyOperator(mixing_rate=mixing_rate)
        self.U_hat = UpdateOperator(eta_W=eta_W, eta_g=eta_g, eta_theta=eta_theta)
        self.rho_hat = RelaxationOperator(epsilon=epsilon, relax_rate=relax_rate)

        # 共鳴点集合 P = {t_k}
        self.resonance_points: list[ResonanceEvent] = []

    def __call__(
        self,
        ensemble: Ensemble,
        u: Optional[np.ndarray] = None,
        t: float = 0.0,
    ) -> tuple[Ensemble, ResonanceEvent]:
        """
        共鳴事象作用素半群の一回適用

        𝔑(Ψ) = (ρ̂ε ∘ Û) ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ (Ψ)

        Parameters
        ----------
        ensemble : Ensemble — 入力集合系
        u : np.ndarray or None — 外部駆動
        t : float — 現在時刻

        Returns
        -------
        (ensemble_final, event)
        """
        ensemble_before = ensemble.copy()

        # ── Step 1: 接触 Ĉ ──
        E = self.C_hat(ensemble)

        # ── Step 2: 協力層生成 L̂ ──
        Phi = self.L_hat(E)

        # ── Step 3: 環境共有 Ê ──
        ensemble_E = self.E_hat(ensemble, Phi, u)

        # ── Step 4: 整合 Â (Kuramoto 同期) ──
        ensemble_A = self.A_hat(ensemble_E, Phi)

        # ── Step 5: 臨界判定 τ̂ ──
        info = self.tau_hat.evaluate(ensemble_A)
        tau = info["tau"]
        R = info["R"]
        Theta = info["Theta"]

        # ── Step 6: 共感 M̂ (τ̂=1 のみ) ──
        ensemble_M = self.M_hat(ensemble_A, tau)

        # ── Step 7: 更新 Û ──
        ensemble_U, delta_g_norms = self.U_hat(ensemble_M, Phi, tau)

        # ── Step 8: 可逆制御 ρ̂ε ──
        ensemble_final, modes = self.rho_hat(
            ensemble_before, ensemble_U, delta_g_norms
        )

        # 共鳴事象の記録
        event = ResonanceEvent(
            t=t,
            tau=tau,
            R=R,
            Theta=Theta,
            R_c=self.tau_hat.R_c,
            delta_g_norms=delta_g_norms,
            relaxation_modes=modes,
            ensemble_before=ensemble_before,
            ensemble_after=ensemble_final,
            E=E,
            Phi=Phi,
        )

        if event.fired:
            self.resonance_points.append(event)

        return ensemble_final, event

    def run(
        self,
        ensemble: Ensemble,
        T_steps: int = 100,
        u_func=None,
    ) -> tuple[Ensemble, list[ResonanceEvent]]:
        """
        時間発展シミュレーション

        Parameters
        ----------
        ensemble : Ensemble — 初期集合系
        T_steps : int — ステップ数
        u_func : callable(t) -> np.ndarray or None — 時刻依存の外部駆動

        Returns
        -------
        (final_ensemble, event_history)
        """
        current = ensemble.copy()
        history = []

        for step in range(T_steps):
            t = float(step)
            u = u_func(t) if u_func is not None else None
            current, event = self(current, u=u, t=t)
            history.append(event)

        return current, history

    @property
    def star_points(self) -> list[float]:
        """
        星点集合 P = {t_k} — 共鳴が発火した時刻の集合
        §10: 人生の軌道は点集合が作る測地線の折れとして決まる
        """
        return [e.t for e in self.resonance_points]

    def summary(self, history: list[ResonanceEvent]) -> dict:
        """シミュレーション結果のサマリ"""
        total = len(history)
        fired = sum(1 for e in history if e.fired)
        R_values = [e.R for e in history]
        return {
            "total_steps": total,
            "resonance_events": fired,
            "resonance_rate": fired / total if total > 0 else 0.0,
            "R_mean": float(np.mean(R_values)),
            "R_max": float(np.max(R_values)),
            "R_final": R_values[-1] if R_values else 0.0,
            "star_points": self.star_points,
        }
