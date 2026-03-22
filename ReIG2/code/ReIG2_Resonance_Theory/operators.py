"""
reig2.operators — 8基本作用素
================================
§3: 非線形作用素半群の構成要素

  Ĉ  : ContactOperator         — 接触・干渉生成
  L̂  : CooperationLayerOperator — 協力層 (場) 生成
  Ê  : EnvironmentShareOperator — 環境共有
  Â  : AlignmentOperator        — 3軸整合
  τ̂  : ThresholdGate            — 臨界判定 (相転移スイッチ)
  M̂  : EmpathyOperator          — 共感 (視点交換)
  Û  : UpdateOperator            — 更新 (塑性)
  ρ̂ε : RelaxationOperator       — 可逆制御 (緩和/固定)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .state import SubjectState, Ensemble
from .alignment import ThreeAxisAlignment, FrequencyVector, PhaseVector


# ═══════════════════════════════════════════════════════════════
#  (1) 接触演算子 Ĉ
# ═══════════════════════════════════════════════════════════════
class ContactOperator:
    """
    Ĉ : X^N → Y
    E_ij = κ_ij(Ψ) ⟨Ψ_i, Ψ_j⟩_*

    一般化内積 ⟨·,·⟩_* : 同型でなくても射影で重なりを測る
    """

    def __call__(self, ensemble: Ensemble) -> np.ndarray:
        """
        干渉行列 E を計算

        Returns
        -------
        E : np.ndarray, shape (N, N)
            E[i,j] = κ_ij · ⟨Ψ_i, Ψ_j⟩_*
        """
        N = ensemble.N
        E = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                E[i, j] = self._interference(ensemble[i], ensemble[j])
        return E

    @staticmethod
    def _interference(psi_i: SubjectState, psi_j: SubjectState) -> complex:
        """
        干渉項: κ_ij · ⟨z_i, z_j⟩
        
        κ_ij は結合行列の相関で文脈依存の結合係数を近似
        """
        # 一般化内積: z_i^* · z_j (射影重なり)
        inner = np.vdot(psi_i.z, psi_j.z)

        # 文脈依存結合係数 κ: 結合行列の類似度
        d = min(psi_i.W.shape[0], psi_j.W.shape[0])
        Wi = psi_i.W[:d, :d]
        Wj = psi_j.W[:d, :d]
        norm_i = np.linalg.norm(Wi)
        norm_j = np.linalg.norm(Wj)
        if norm_i < 1e-12 or norm_j < 1e-12:
            kappa = 0.0
        else:
            kappa = np.sum(Wi * Wj) / (norm_i * norm_j)

        return kappa * inner


# ═══════════════════════════════════════════════════════════════
#  (2) 協力層生成演算子 L̂
# ═══════════════════════════════════════════════════════════════
class CooperationLayerOperator:
    """
    L̂ : Y → Φ
    Φ = F(E)

    干渉行列 E から場 (協力層) を粗視化生成。
    L_AB ≡ Φ
    """

    def __call__(self, E: np.ndarray) -> np.ndarray:
        """
        干渉行列 E から場 Φ を生成

        実装: 干渉行列の実部の対称化 + スペクトル正規化
        
        Returns
        -------
        Phi : np.ndarray, shape (N, N)
            協力層場 (実対称行列)
        """
        # 干渉の実部 (可観測量)
        E_real = np.real(E)
        # 対称化 (凝縮体: 相互情報)
        Phi = (E_real + E_real.T) / 2.0
        # 正規化
        max_val = np.max(np.abs(Phi))
        if max_val > 1e-12:
            Phi /= max_val
        return Phi


# ═══════════════════════════════════════════════════════════════
#  (3) 環境共有演算子 Ê
# ═══════════════════════════════════════════════════════════════
class EnvironmentShareOperator:
    """
    Ê : (Ψ, Φ, u(t)) → Ψ^(E)

    場 Φ を媒介として環境制約を共有。
    外部駆動 u(t) による環境整地。
    """

    def __call__(
        self,
        ensemble: Ensemble,
        Phi: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> Ensemble:
        """
        環境制約を各主体に反映

        Parameters
        ----------
        ensemble : Ensemble
        Phi : np.ndarray, shape (N, N)  — 協力層場
        u : np.ndarray, shape (d,) or None — 外部駆動

        Returns
        -------
        ensemble_E : Ensemble  — 環境共有後の集合系
        """
        N = ensemble.N
        result = ensemble.copy()

        for i in range(N):
            # 場からの影響: Φ の i 行の平均 → z を摂動
            field_influence = np.mean(Phi[i, :])

            # 外部駆動がある場合
            if u is not None:
                d = min(len(u), result[i].dim_z)
                result[i].z[:d] += 0.1 * field_influence * u[:d]
            else:
                result[i].z *= (1.0 + 0.05 * field_influence)

            # z の正規化保持
            norm_z = np.linalg.norm(result[i].z)
            if norm_z > 1e-12:
                result[i].z /= norm_z

        return result


# ═══════════════════════════════════════════════════════════════
#  (4) 整合演算子 Â
# ═══════════════════════════════════════════════════════════════
class AlignmentOperator:
    """
    Â ≡ S : X^N × Φ × U → X^N
    Ψ' = S(Ψ, Φ, u(t))

    Kuramoto 型同期 + 3軸整合

    位相更新: φ_i → φ_i + Σ_j K_ij sin(φ_j - φ_i)
    """

    def __init__(self, coupling_strength: float = 0.1):
        self.coupling_strength = coupling_strength

    def __call__(
        self,
        ensemble: Ensemble,
        Phi: np.ndarray,
    ) -> Ensemble:
        """
        Kuramoto 型位相同期を実行

        秩序パラメータ: R e^{iΘ} = (1/N) Σ_j exp(i φ_j)
        位相更新: φ_i → φ_i + K Σ_j Φ_{ij} sin(φ_j - φ_i)

        Returns
        -------
        ensemble_A : Ensemble — 整合後の集合系
        """
        N = ensemble.N
        result = ensemble.copy()
        phases = ensemble.phases()

        for i in range(N):
            # Kuramoto 型位相更新
            delta_phi = 0.0
            for j in range(N):
                if i == j:
                    continue
                K_ij = self.coupling_strength * Phi[i, j]
                delta_phi += K_ij * np.sin(phases[j] - phases[i])

            # z の第0成分の位相を更新
            old_amp = np.abs(result[i].z[0])
            old_phase = np.angle(result[i].z[0])
            new_phase = old_phase + delta_phi
            result[i].z[0] = old_amp * np.exp(1j * new_phase)

        return result


# ═══════════════════════════════════════════════════════════════
#  (5) 臨界演算子 τ̂
# ═══════════════════════════════════════════════════════════════
class ThresholdGate:
    """
    τ̂ : X^N → {0, 1}
    τ̂(Ψ) = 1{R(Ψ) > R_c(θ)}

    秩序パラメータ R が臨界値 R_c を越えると共鳴事象が発火。
    """

    def __init__(self, R_c: float = 0.7):
        """
        Parameters
        ----------
        R_c : float
            デフォルト臨界閾値 (主体依存にオーバーライド可能)
        """
        self.R_c = R_c

    def __call__(self, ensemble: Ensemble) -> int:
        """
        共鳴事象判定

        Returns
        -------
        0 or 1 : 臨界越え判定
        """
        R, _ = ensemble.order_parameter()
        return 1 if R > self.R_c else 0

    def evaluate(self, ensemble: Ensemble) -> dict:
        """判定結果と詳細を返す"""
        R, Theta = ensemble.order_parameter()
        fired = R > self.R_c
        return {
            "R": R,
            "Theta": Theta,
            "R_c": self.R_c,
            "fired": fired,
            "tau": 1 if fired else 0,
        }


# ═══════════════════════════════════════════════════════════════
#  (6) 共感演算子 M̂
# ═══════════════════════════════════════════════════════════════
class EmpathyOperator:
    """
    M̂ : Ψ^(E) --[τ̂=1]--> Ψ^(M)

    ⚠ 重要: M̂ は前提ではない。臨界越え後にのみ起動可能。
    共感は τ̂ = 1 の後に可能となる高次相である。

    実装: 視点交換 — 主体間の内部表現を部分的に混合
    """

    def __init__(self, mixing_rate: float = 0.1):
        """
        Parameters
        ----------
        mixing_rate : float
            視点交換の混合率 (0: なし, 1: 完全交換)
        """
        self.mixing_rate = mixing_rate

    def __call__(
        self,
        ensemble: Ensemble,
        tau: int,
    ) -> Ensemble:
        """
        共感作用: 臨界越え時のみ実行

        Returns
        -------
        ensemble_M : Ensemble — 共感後の集合系 (τ̂=0 なら無変更)
        """
        if tau != 1:
            # τ̂ = 0: 共感は発現しない
            return ensemble.copy()

        N = ensemble.N
        result = ensemble.copy()
        alpha = self.mixing_rate

        # 視点交換: 近傍主体の結合行列を部分混合
        for i in range(N):
            W_mix = np.zeros_like(result[i].W)
            count = 0
            for j in range(N):
                if i == j:
                    continue
                d = min(result[i].W.shape[0], result[j].W.shape[0])
                W_mix[:d, :d] += result[j].W[:d, :d]
                count += 1
            if count > 0:
                W_mix /= count
                d = result[i].W.shape[0]
                result[i].W = (1 - alpha) * result[i].W + alpha * W_mix[:d, :d]

        return result


# ═══════════════════════════════════════════════════════════════
#  (7) 更新演算子 Û
# ═══════════════════════════════════════════════════════════════
class UpdateOperator:
    """
    Û : X^N × Φ → X^N
    (W_i, g_i, θ_i) ↦ (W_i + ΔW_i, g_i + Δg_i, θ_i + Δθ_i)

    ログ (記録) ではなく内部構造の「更新」(再配列)。
    """

    def __init__(
        self,
        eta_W: float = 0.05,
        eta_g: float = 0.02,
        eta_theta: float = 0.03,
    ):
        """
        Parameters
        ----------
        eta_W, eta_g, eta_theta : float
            各パラメータの更新率
        """
        self.eta_W = eta_W
        self.eta_g = eta_g
        self.eta_theta = eta_theta

    def __call__(
        self,
        ensemble: Ensemble,
        Phi: np.ndarray,
        tau: int,
    ) -> tuple[Ensemble, list[float]]:
        """
        内部構造更新

        Parameters
        ----------
        ensemble : Ensemble
        Phi : np.ndarray — 協力層場
        tau : int — 臨界判定結果

        Returns
        -------
        (ensemble_updated, delta_g_norms)
            ensemble_updated : 更新後の集合系
            delta_g_norms : 各主体の ‖Δg‖ (可逆制御に使用)
        """
        if tau != 1:
            # 共鳴事象なし → 更新なし
            norms = [0.0] * ensemble.N
            return ensemble.copy(), norms

        N = ensemble.N
        result = ensemble.copy()
        delta_g_norms = []

        for i in range(N):
            # ΔW: 場との相関に基づく Hebbian 的更新
            field_vec = Phi[i, :]
            # 結合行列の更新 (外積近似)
            d = result[i].W.shape[0]
            z_real = np.real(result[i].z[:d])
            field_d = np.zeros(d)
            field_d[:min(N, d)] = field_vec[:min(N, d)]
            delta_W = self.eta_W * np.outer(z_real, field_d[:d])
            delta_W = (delta_W + delta_W.T) / 2  # 対称化
            result[i].W += delta_W

            # Δg: 計量の更新 (共鳴による内界幾何の変化)
            p = result[i].g.shape[0]
            delta_g = self.eta_g * np.random.randn(p, p)
            delta_g = (delta_g + delta_g.T) / 2  # 対称性保持
            result[i].g += delta_g
            # 正定値性の保持
            eigvals = np.linalg.eigvalsh(result[i].g)
            if np.min(eigvals) < 0.01:
                result[i].g += (0.02 - np.min(eigvals)) * np.eye(p)

            delta_g_norms.append(float(np.linalg.norm(delta_g)))

            # Δθ: パラメータ更新
            delta_theta = self.eta_theta * np.mean(Phi[i, :]) * np.ones(result[i].dim_theta)
            result[i].theta += delta_theta

        return result, delta_g_norms


# ═══════════════════════════════════════════════════════════════
#  (8) 可逆制御演算子 ρ̂ε
# ═══════════════════════════════════════════════════════════════
class RelaxationOperator:
    """
    ρ̂ε : X → X

    ‖Δg‖ < ε  → Relax (緩和: 戻りやすい → 更新を部分的に元に戻す)
    ‖Δg‖ ≥ ε  → Fix   (準安定固定: 戻りにくい → 更新を固定)

    条件付き可逆性の制御。
    """

    def __init__(self, epsilon: float = 0.1, relax_rate: float = 0.5):
        """
        Parameters
        ----------
        epsilon : float
            可逆/固定の閾値
        relax_rate : float
            緩和時の戻り率 (0: 完全に戻す, 1: 戻さない)
        """
        self.epsilon = epsilon
        self.relax_rate = relax_rate

    def __call__(
        self,
        ensemble_before: Ensemble,
        ensemble_after: Ensemble,
        delta_g_norms: list[float],
    ) -> tuple[Ensemble, list[str]]:
        """
        条件付き可逆制御

        Returns
        -------
        (ensemble_final, modes)
            ensemble_final : 可逆制御後の集合系
            modes : 各主体の判定 ("relax" or "fix")
        """
        N = ensemble_after.N
        result = ensemble_after.copy()
        modes = []

        for i in range(N):
            if delta_g_norms[i] < self.epsilon:
                # Relax: 部分的に元に戻す
                alpha = self.relax_rate
                result[i].W = alpha * ensemble_after[i].W + (1 - alpha) * ensemble_before[i].W
                result[i].g = alpha * ensemble_after[i].g + (1 - alpha) * ensemble_before[i].g
                result[i].theta = alpha * ensemble_after[i].theta + (1 - alpha) * ensemble_before[i].theta
                modes.append("relax")
            else:
                # Fix: 更新を固定 (準安定状態)
                modes.append("fix")

        return result, modes

    def classify(self, delta_g_norm: float) -> str:
        """単一の ‖Δg‖ に対する判定"""
        return "relax" if delta_g_norm < self.epsilon else "fix"
