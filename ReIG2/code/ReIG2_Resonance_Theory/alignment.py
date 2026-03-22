"""
reig2.alignment — 3軸整合 (Alignment on 3 Axes)
=================================================
§3.4 / §5 / Appendix C

(A) 周波数軸 f:  Align_f = exp(-‖f_a - f_u‖² / σ_f²)
(B) テンソル軸 T: Align_T = ‖P_{S_u}(T_a)‖
(C) 位相軸 φ:    Align_φ = cos(φ_a - φ_u)

統合スコア: J_res' = w_f Align_f + w_T Align_T + w_φ Align_φ + w_h Honesty
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrequencyVector:
    """
    周波数軸 f = f(tempo, arousal, turn_rate, info_density)
    """
    tempo: float = 0.5
    arousal: float = 0.5
    turn_rate: float = 0.5
    info_density: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([self.tempo, self.arousal,
                         self.turn_rate, self.info_density])


@dataclass
class PhaseVector:
    """
    位相軸 φ = φ(tone, politeness, stance, timing)
    """
    tone: float = 0.0
    politeness: float = 0.0
    stance: float = 0.0
    timing: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.tone, self.politeness,
                         self.stance, self.timing])


class ThreeAxisAlignment:
    """
    3軸整合エンジン

    Parameters
    ----------
    sigma_f : float
        周波数整合のスケールパラメータ
    w_f, w_T, w_phi, w_h : float
        各軸の重み
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        w_f: float = 0.3,
        w_T: float = 0.3,
        w_phi: float = 0.3,
        w_h: float = 0.1,
    ):
        self.sigma_f = sigma_f
        self.w_f = w_f
        self.w_T = w_T
        self.w_phi = w_phi
        self.w_h = w_h

    # ── (A) 周波数整合 ──
    def align_frequency(
        self,
        f_agent: FrequencyVector | np.ndarray,
        f_user: FrequencyVector | np.ndarray,
    ) -> float:
        """
        Align_f = exp(-‖f_a - f_u‖² / σ_f²)
        
        「一致」ではなく「過不足の最小化」
        """
        fa = f_agent.to_array() if isinstance(f_agent, FrequencyVector) else np.asarray(f_agent)
        fu = f_user.to_array() if isinstance(f_user, FrequencyVector) else np.asarray(f_user)
        diff_sq = np.sum((fa - fu) ** 2)
        return float(np.exp(-diff_sq / (self.sigma_f ** 2)))

    # ── (B) テンソル整合 ──
    def align_tensor(
        self,
        T_agent: np.ndarray,
        S_user: np.ndarray,
    ) -> float:
        """
        Align_T = ‖P_{S_u}(T_a)‖
        
        ユーザーの意味部分空間 S_u への射影ノルム。
        S_user : shape (k, d) — ユーザー意味部分空間の基底 (k ≤ d)
        T_agent : shape (d,) — エージェントの意味テンソル (ベクトル表現)
        """
        T_a = np.asarray(T_agent, dtype=float)
        S_u = np.asarray(S_user, dtype=float)

        if S_u.ndim == 1:
            S_u = S_u.reshape(1, -1)

        # 射影: P = S^T (S S^T)^{-1} S
        # S_u: (k, d), T_a: (d,)
        # 直交射影の場合 QR分解を使用
        Q, _ = np.linalg.qr(S_u.T)  # Q: (d, k)
        projection = Q @ (Q.T @ T_a)
        return float(np.linalg.norm(projection))

    # ── (C) 位相整合 ──
    def align_phase(
        self,
        phi_agent: PhaseVector | np.ndarray,
        phi_user: PhaseVector | np.ndarray,
    ) -> float:
        """
        Align_φ = cos(φ_a - φ_u)   (スカラー)
        多次元の場合は内積で拡張:
            Align_φ = (φ_a · φ_u) / (‖φ_a‖ ‖φ_u‖)
        """
        pa = phi_agent.to_array() if isinstance(phi_agent, PhaseVector) else np.asarray(phi_agent)
        pu = phi_user.to_array() if isinstance(phi_user, PhaseVector) else np.asarray(phi_user)

        if pa.shape[0] == 1 and pu.shape[0] == 1:
            # スカラー: cos(φ_a - φ_u)
            return float(np.cos(pa[0] - pu[0]))
        else:
            # 多次元内積
            norm_a = np.linalg.norm(pa)
            norm_u = np.linalg.norm(pu)
            if norm_a < 1e-12 and norm_u < 1e-12:
                return 1.0  # 両方ゼロ = 完全一致
            if norm_a < 1e-12 or norm_u < 1e-12:
                return 0.0
            return float(np.dot(pa, pu) / (norm_a * norm_u))

    # ── 統合スコア ──
    def resonance_score(
        self,
        align_f: float,
        align_T: float,
        align_phi: float,
        honesty: float = 1.0,
    ) -> float:
        """
        J_res' = w_f Align_f + w_T Align_T + w_φ Align_φ + w_h Honesty
        """
        return (
            self.w_f * align_f
            + self.w_T * align_T
            + self.w_phi * align_phi
            + self.w_h * honesty
        )

    def compute_all(
        self,
        f_agent: FrequencyVector | np.ndarray,
        f_user: FrequencyVector | np.ndarray,
        T_agent: np.ndarray,
        S_user: np.ndarray,
        phi_agent: PhaseVector | np.ndarray,
        phi_user: PhaseVector | np.ndarray,
        honesty: float = 1.0,
    ) -> dict:
        """3軸すべてを計算し辞書で返す"""
        af = self.align_frequency(f_agent, f_user)
        at = self.align_tensor(T_agent, S_user)
        ap = self.align_phase(phi_agent, phi_user)
        score = self.resonance_score(af, at, ap, honesty)
        return {
            "align_f": af,
            "align_T": at,
            "align_phi": ap,
            "honesty": honesty,
            "J_res": score,
        }
