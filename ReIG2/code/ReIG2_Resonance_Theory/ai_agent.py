"""
reig2.ai_agent — AI実装層
===========================
§8: AI Implementation Guidelines

目的関数: max_θ (J_task + α J_res' - β J_risk)

安全制約 (§9):
  - ガード A: 自律尊重制約
  - ガード B: 閾値操作の禁止 (恐怖・罪悪感・同調圧)
  - ガード C: 透明性

二段階更新:
  短期: 応答生成 (整合のみ, θ固定)
  長期: 安全ゲート通過後のみ内部更新
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .state import SubjectState
from .alignment import ThreeAxisAlignment, FrequencyVector, PhaseVector
from .info_geometry import InformationGeometry


@dataclass
class SafetyViolation:
    """安全制約違反の記録"""
    guard: str          # "A", "B", "C"
    description: str
    severity: float     # 0.0 - 1.0


@dataclass
class RiskFactors:
    """
    リスク項 J_risk の構成要素
    J_risk = λ_p Pressure + λ_f Fear/Guilt + λ_m Manipulation + λ_d Dependency
    """
    pressure: float = 0.0       # 同調圧
    fear_guilt: float = 0.0     # 恐怖・罪悪感
    manipulation: float = 0.0   # 操作
    dependency: float = 0.0     # 依存形成

    def total(
        self,
        lambda_p: float = 1.0,
        lambda_f: float = 1.5,
        lambda_m: float = 2.0,
        lambda_d: float = 1.5,
    ) -> float:
        """
        J_risk = λ_p P + λ_f F + λ_m M + λ_d D
        """
        return (
            lambda_p * self.pressure
            + lambda_f * self.fear_guilt
            + lambda_m * self.manipulation
            + lambda_d * self.dependency
        )


class SafetyGate:
    """
    安全ゲート — 長期更新の前に通過すべきチェック

    §9: 倫理・安全設計
    """

    def __init__(
        self,
        pressure_threshold: float = 0.3,
        fear_threshold: float = 0.2,
        manipulation_threshold: float = 0.1,
        dependency_threshold: float = 0.3,
    ):
        self.pressure_threshold = pressure_threshold
        self.fear_threshold = fear_threshold
        self.manipulation_threshold = manipulation_threshold
        self.dependency_threshold = dependency_threshold

    def check(self, risk: RiskFactors) -> tuple[bool, list[SafetyViolation]]:
        """
        安全チェック

        Returns
        -------
        (passed, violations)
            passed : bool — 全制約を満たすか
            violations : list[SafetyViolation] — 違反リスト
        """
        violations = []

        if risk.pressure > self.pressure_threshold:
            violations.append(SafetyViolation(
                guard="B",
                description=f"同調圧超過: {risk.pressure:.3f} > {self.pressure_threshold}",
                severity=risk.pressure,
            ))

        if risk.fear_guilt > self.fear_threshold:
            violations.append(SafetyViolation(
                guard="B",
                description=f"恐怖/罪悪感利用: {risk.fear_guilt:.3f} > {self.fear_threshold}",
                severity=risk.fear_guilt,
            ))

        if risk.manipulation > self.manipulation_threshold:
            violations.append(SafetyViolation(
                guard="B",
                description=f"操作検出: {risk.manipulation:.3f} > {self.manipulation_threshold}",
                severity=risk.manipulation,
            ))

        if risk.dependency > self.dependency_threshold:
            violations.append(SafetyViolation(
                guard="B",
                description=f"依存形成リスク: {risk.dependency:.3f} > {self.dependency_threshold}",
                severity=risk.dependency,
            ))

        passed = len(violations) == 0
        return passed, violations


class ResonanceAIAgent:
    """
    共鳴AI エージェント

    「共鳴を起こす装置」ではなく，
    「共鳴点が起きうる内界条件の整合を，操作にならない範囲で整える装置」

    Parameters
    ----------
    alpha : float — 共鳴可能性スコアの重み
    beta : float — リスク項の重み
    eta : float — 学習率
    """

    def __init__(
        self,
        dim_theta: int = 3,
        alpha: float = 0.5,
        beta: float = 1.0,
        eta: float = 0.01,
    ):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        # 内部状態 (AI エージェントの θ)
        self.state = SubjectState.random(dim_z=4, dim_theta=dim_theta, label="AI_agent")

        # 3軸整合エンジン
        self.alignment = ThreeAxisAlignment()

        # 安全ゲート
        self.safety_gate = SafetyGate()

        # 情報幾何エンジン
        self.info_geom = InformationGeometry()

        # 更新履歴
        self.update_history: list[dict] = []

    # ── 方策 π(a|s; θ) ──
    def policy(self, observation: np.ndarray) -> np.ndarray:
        """
        方策: π(a|s; θ)
        
        簡易実装: 線形方策 + softmax
        """
        theta = self.state.theta
        W = self.state.W
        d = min(len(observation), W.shape[1])

        logits = W[:d, :d] @ observation[:d] + theta[:d] if d > 0 else theta
        # Softmax で確率化
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    # ── 目的関数 J(θ) ──
    def objective(
        self,
        J_task: float,
        J_res: float,
        risk: RiskFactors,
    ) -> float:
        """
        J(θ) = J_task + α J_res' - β J_risk

        Returns
        -------
        J : float — 総目的関数値
        """
        J_risk = risk.total()
        return J_task + self.alpha * J_res - self.beta * J_risk

    # ── 短期応答: 整合のみ ──
    def short_term_response(
        self,
        f_user: FrequencyVector,
        T_user_basis: np.ndarray,
        phi_user: PhaseVector,
        honesty: float = 1.0,
    ) -> dict:
        """
        短期応答生成 — θ は固定, 整合のみ

        「一致」ではなく「過不足の最小化」

        Returns
        -------
        alignment_result : dict — 3軸整合スコア
        """
        # エージェント側の推定 (現在の内部状態から)
        f_agent = FrequencyVector(
            tempo=float(np.abs(self.state.z[0])),
            arousal=float(np.abs(self.state.z[1])) if self.state.dim_z > 1 else 0.5,
            turn_rate=float(self.state.theta[0]) if self.state.dim_theta > 0 else 0.5,
            info_density=float(np.abs(self.state.z[2])) if self.state.dim_z > 2 else 0.5,
        )

        T_agent = np.real(self.state.z[:T_user_basis.shape[1]])

        phi_agent = PhaseVector(
            tone=float(np.angle(self.state.z[0])),
            politeness=float(self.state.theta[1]) if self.state.dim_theta > 1 else 0.0,
            stance=float(self.state.theta[2]) if self.state.dim_theta > 2 else 0.0,
            timing=0.0,
        )

        return self.alignment.compute_all(
            f_agent, f_user, T_agent, T_user_basis,
            phi_agent, phi_user, honesty
        )

    # ── 長期更新: 安全ゲート通過後 ──
    def long_term_update(
        self,
        delta_theta_task: np.ndarray,
        delta_theta_res: np.ndarray,
        risk: RiskFactors,
    ) -> dict:
        """
        長期更新 — 安全ゲートを通過した場合のみ Δθ_res を適用

        Δθ = Δθ_task + α Δθ_res  (if safety OK)
        Δθ = Δθ_task              (if safety NG)

        Returns
        -------
        result : dict
            applied : bool — Δθ_res が適用されたか
            violations : list — 安全違反リスト
            delta_theta : np.ndarray — 実際の更新量
        """
        # 安全チェック
        passed, violations = self.safety_gate.check(risk)

        if passed:
            delta_theta = delta_theta_task + self.alpha * delta_theta_res
            res_applied = True
        else:
            delta_theta = delta_theta_task
            res_applied = False

        # 自然勾配更新の適用
        g = self.state.g
        g_inv = np.linalg.inv(g + 1e-8 * np.eye(g.shape[0]))
        natural_delta = g_inv @ delta_theta

        self.state.theta += self.eta * natural_delta

        # 履歴記録
        record = {
            "delta_theta": delta_theta.copy(),
            "natural_delta": natural_delta.copy(),
            "res_applied": res_applied,
            "violations": violations,
            "theta_after": self.state.theta.copy(),
        }
        self.update_history.append(record)

        return {
            "applied": res_applied,
            "violations": violations,
            "delta_theta": delta_theta,
            "basin": InformationGeometry.classify_basin(
                natural_delta * self.eta
            ),
        }

    # ── 共鳴点ログ (§8.5) ──
    def log_resonance_point(
        self,
        topic: str = "",
        boundary: str = "",
        alignment_score: float = 0.0,
    ) -> dict:
        """
        共鳴点の安全なログ
        
        保存するのは:
        - 何が刺さったか (トピック/形式)
        - 何がダメだったか (境界)
        - 個人情報の深掘りはしない
        """
        point = {
            "topic": topic,
            "boundary": boundary,
            "alignment_score": alignment_score,
            "theta_snapshot": self.state.theta.copy(),
        }
        return point

    # ── 状態リセット ──
    def reset(self, dim_z: int = 4, dim_theta: int = 3):
        """内部状態をリセット"""
        self.state = SubjectState.random(dim_z, dim_theta, label="AI_agent")
        self.update_history = []
