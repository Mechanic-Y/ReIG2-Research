"""
ReIG2/twinRIG 第4章
相転移生成演算子
Phase Transition Generation Operator (G)

Mechanic-Y / Yasuyuki Wakita
2025年12月

連続的時間発展から離散的状態遷移への転換
G = P ∘ E ∘ R（位相ジャンプ・拡張・ねじれの三成分構造）
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# 定数
# =============================================================================

HBAR = 1.0


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class PhaseState:
    """相状態"""
    state_vector: np.ndarray
    phase_index: int
    coherence: float
    stability: float


@dataclass
class TransitionResult:
    """相転移の結果"""
    initial_state: PhaseState
    final_state: PhaseState
    transition_probability: float
    operator_applied: str


# =============================================================================
# 三成分構造: R（ねじれ）, E（拡張）, P（位相ジャンプ）
# =============================================================================

class TwistOperator:
    """
    ねじれ演算子 R
    
    位相空間の連続的回転を実行
    """
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def operator(self, theta: float) -> np.ndarray:
        """
        回転演算子を生成
        
        R(θ) = exp(-i θ J_z)
        """
        if self.dim == 2:
            # 2準位系: σ_z による回転
            J_z = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        else:
            # 一般次元: 対角行列
            J_z = np.diag(np.arange(self.dim) - (self.dim - 1) / 2)
        
        return expm(-1j * theta * J_z)
    
    def apply(self, state: np.ndarray, theta: float) -> np.ndarray:
        """状態にねじれを適用"""
        R = self.operator(theta)
        return R @ state


class ExtensionOperator:
    """
    拡張演算子 E
    
    状態空間の次元拡大を実行
    位相コヒーレンス条件に基づく選択的拡張
    """
    
    def __init__(self, dim_in: int = 2, dim_out: int = 4):
        self.dim_in = dim_in
        self.dim_out = dim_out
    
    def operator(self, coherence_threshold: float = 0.5) -> np.ndarray:
        """
        拡張演算子を生成
        
        低次元状態を高次元空間に埋め込む
        """
        E = np.zeros((self.dim_out, self.dim_in), dtype=complex)
        
        # 単純な埋め込み（最初のdim_in次元に射影）
        for i in range(self.dim_in):
            E[i, i] = 1.0
        
        return E
    
    def apply(self, state: np.ndarray, coherence_threshold: float = 0.5) -> np.ndarray:
        """状態を拡張"""
        E = self.operator(coherence_threshold)
        extended = E @ state
        
        # 正規化
        norm = np.linalg.norm(extended)
        if norm > 0:
            extended = extended / norm
        
        return extended
    
    def reverse(self) -> np.ndarray:
        """逆演算（射影）"""
        return self.operator().T.conj()


class PhaseJumpOperator:
    """
    位相ジャンプ演算子 P
    
    離散的な状態遷移を実行
    確率的な位相変化を含む
    """
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def transition_matrix(self, p_jump: float) -> np.ndarray:
        """
        遷移行列を生成
        
        確率 p_jump で隣接状態に遷移
        """
        P = np.eye(self.dim, dtype=complex)
        
        if self.dim == 2:
            # 2準位系: ビットフリップ的な遷移
            P = np.array([
                [1 - p_jump, p_jump],
                [p_jump, 1 - p_jump]
            ], dtype=complex)
        else:
            # 一般次元: 隣接遷移
            for i in range(self.dim - 1):
                P[i, i] = 1 - p_jump
                P[i, i+1] = p_jump * 0.5
                P[i+1, i] = p_jump * 0.5
            P[-1, -1] = 1 - p_jump * 0.5
        
        return P
    
    def apply(self, state: np.ndarray, p_jump: float) -> np.ndarray:
        """状態に位相ジャンプを適用"""
        P = self.transition_matrix(p_jump)
        return P @ state


# =============================================================================
# 相転移生成演算子 G
# =============================================================================

class PhaseTransitionGenerator:
    """
    相転移生成演算子 G = P ∘ E ∘ R
    
    三段階の操作を合成：
    1. R: ねじれ（位相回転）
    2. E: 拡張（次元拡大）
    3. P: 位相ジャンプ（離散遷移）
    """
    
    def __init__(
        self,
        dim: int = 2,
        dim_extended: int = 4,
        enable_extension: bool = False
    ):
        """
        Args:
            dim: 基本次元
            dim_extended: 拡張後の次元
            enable_extension: 次元拡張を有効にするか
        """
        self.dim = dim
        self.dim_extended = dim_extended
        self.enable_extension = enable_extension
        
        self.R = TwistOperator(dim)
        self.E = ExtensionOperator(dim, dim_extended)
        self.P = PhaseJumpOperator(dim_extended if enable_extension else dim)
    
    def apply(
        self,
        state: np.ndarray,
        theta: float,
        p_jump: float,
        coherence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        G = P ∘ E ∘ R を適用
        
        Args:
            state: 入力状態
            theta: ねじれ角
            p_jump: 位相ジャンプ確率
            coherence_threshold: コヒーレンス閾値
        
        Returns:
            (output_state, info)
        """
        info = {"stages": []}
        
        # Stage 1: Twist (R)
        state_r = self.R.apply(state, theta)
        info["stages"].append({"name": "R", "theta": theta})
        
        # Stage 2: Extension (E)
        if self.enable_extension:
            state_e = self.E.apply(state_r, coherence_threshold)
        else:
            state_e = state_r
        info["stages"].append({"name": "E", "enabled": self.enable_extension})
        
        # Stage 3: Phase Jump (P)
        state_p = self.P.apply(state_e, p_jump)
        info["stages"].append({"name": "P", "p_jump": p_jump})
        
        # 正規化
        output = state_p / np.linalg.norm(state_p)
        
        return output, info
    
    def iterate(
        self,
        initial_state: np.ndarray,
        n_iterations: int,
        theta_func: Callable[[int], float],
        p_jump_func: Callable[[int], float]
    ) -> List[np.ndarray]:
        """
        G を繰り返し適用
        
        Args:
            initial_state: 初期状態
            n_iterations: 反復回数
            theta_func: 反復番号からねじれ角を計算する関数
            p_jump_func: 反復番号から位相ジャンプ確率を計算する関数
        
        Returns:
            状態の履歴
        """
        history = [initial_state.copy()]
        state = initial_state.copy()
        
        for n in range(n_iterations):
            theta = theta_func(n)
            p_jump = p_jump_func(n)
            state, _ = self.apply(state, theta, p_jump)
            history.append(state.copy())
        
        return history


# =============================================================================
# 相転移ハザード率モデル
# =============================================================================

def transition_hazard_rate(
    p_n: float,
    gamma: float = 0.5,
    beta: float = 0.3,
    omega: float = 1.0,
    tau: float = 0.0
) -> float:
    """
    相転移ハザード率の動態方程式
    
    dp_n/dt = -γ p_n (1 - p_n) + β sin(ω τ)
    
    Args:
        p_n: 現在の相転移確率
        gamma: 減衰率
        beta: 振動振幅
        omega: 振動周波数
        tau: 時間パラメータ
    
    Returns:
        dp_n/dt
    """
    return -gamma * p_n * (1 - p_n) + beta * np.sin(omega * tau)


def evolve_hazard_rate(
    p_0: float,
    tau_range: np.ndarray,
    gamma: float = 0.5,
    beta: float = 0.3,
    omega: float = 1.0
) -> np.ndarray:
    """
    ハザード率の時間発展（Euler法）
    """
    p_history = [p_0]
    p = p_0
    
    dt = tau_range[1] - tau_range[0] if len(tau_range) > 1 else 0.01
    
    for tau in tau_range[1:]:
        dp = transition_hazard_rate(p, gamma, beta, omega, tau)
        p = p + dp * dt
        p = max(0, min(1, p))  # [0, 1] に制限
        p_history.append(p)
    
    return np.array(p_history)


# =============================================================================
# 連続発展との接続
# =============================================================================

def continuous_to_discrete_limit(
    U_continuous: np.ndarray,
    threshold: float = 0.9
) -> np.ndarray:
    """
    連続発展演算子から離散遷移への極限
    
    コヒーレンス条件: C(ρ) ≥ threshold のとき連続、
    そうでなければ離散遷移
    """
    # コヒーレンスの計算（非対角成分の大きさ）
    coherence = np.sum(np.abs(U_continuous - np.diag(np.diag(U_continuous))))
    
    if coherence >= threshold:
        # 連続的発展を維持
        return U_continuous
    else:
        # 離散化：対角成分のみを保持
        return np.diag(np.diag(U_continuous))


def interpolate_continuous_discrete(
    U_continuous: np.ndarray,
    G_discrete: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    連続発展と離散遷移の補間
    
    U_interp = (1 - α) U_continuous + α G_discrete
    
    α = 0: 完全連続
    α = 1: 完全離散
    """
    return (1 - alpha) * U_continuous + alpha * G_discrete


# =============================================================================
# 待機光子安定化モデル
# =============================================================================

class StandbyPhotonStabilizer:
    """
    待機光子安定化モデル
    
    相転移の安定化を光子場との相互作用でモデル化
    """
    
    def __init__(self, n_photons: int = 10, coupling: float = 0.1):
        """
        Args:
            n_photons: 光子数カットオフ
            coupling: 結合定数
        """
        self.n_photons = n_photons
        self.coupling = coupling
        
        # 光子場の次元
        self.dim_field = n_photons + 1
    
    def creation_operator(self) -> np.ndarray:
        """生成演算子 a†"""
        a_dag = np.zeros((self.dim_field, self.dim_field), dtype=complex)
        for n in range(self.n_photons):
            a_dag[n+1, n] = np.sqrt(n + 1)
        return a_dag
    
    def annihilation_operator(self) -> np.ndarray:
        """消滅演算子 a"""
        return self.creation_operator().T.conj()
    
    def number_operator(self) -> np.ndarray:
        """数演算子 n = a†a"""
        a_dag = self.creation_operator()
        a = self.annihilation_operator()
        return a_dag @ a
    
    def stabilization_hamiltonian(self, system_dim: int = 2) -> np.ndarray:
        """
        安定化ハミルトニアン
        
        H_stab = g (a† ⊗ σ_- + a ⊗ σ_+)
        """
        a_dag = self.creation_operator()
        a = self.annihilation_operator()
        
        # システムの昇降演算子
        if system_dim == 2:
            sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
            sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        else:
            sigma_plus = np.zeros((system_dim, system_dim), dtype=complex)
            sigma_minus = np.zeros((system_dim, system_dim), dtype=complex)
            for i in range(system_dim - 1):
                sigma_plus[i, i+1] = 1
                sigma_minus[i+1, i] = 1
        
        # テンソル積
        H = self.coupling * (
            np.kron(a_dag, sigma_minus) + np.kron(a, sigma_plus)
        )
        
        return H


# =============================================================================
# デモ
# =============================================================================

def demo():
    """相転移生成演算子のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第4章")
    print("相転移生成演算子 G = P ∘ E ∘ R")
    print("=" * 60)
    
    # 初期状態
    psi_0 = np.array([1, 0], dtype=complex)
    
    # 1. 個別演算子のテスト
    print("\n[1] 個別演算子")
    
    # Twist
    R = TwistOperator(dim=2)
    state_r = R.apply(psi_0, theta=np.pi/4)
    print(f"    R(π/4)|0⟩: P(|1⟩) = {np.abs(state_r[1])**2:.3f}")
    
    # Phase Jump
    P = PhaseJumpOperator(dim=2)
    state_p = P.apply(psi_0, p_jump=0.3)
    print(f"    P(0.3)|0⟩: P(|1⟩) = {np.abs(state_p[1])**2:.3f}")
    
    # 2. 合成演算子 G
    print("\n[2] 合成演算子 G = P ∘ E ∘ R")
    G = PhaseTransitionGenerator(dim=2, enable_extension=False)
    
    test_cases = [
        (0, 0),
        (np.pi/4, 0.1),
        (np.pi/2, 0.3),
        (np.pi, 0.5),
    ]
    
    for theta, p_jump in test_cases:
        result, info = G.apply(psi_0, theta, p_jump)
        P_1 = np.abs(result[1])**2
        print(f"    θ={theta:.2f}, p={p_jump:.1f}: P(|1⟩) = {P_1:.3f}")
    
    # 3. 反復適用
    print("\n[3] G の反復適用")
    
    def theta_func(n): return np.pi / 10
    def p_jump_func(n): return 0.1 * (1 - np.exp(-n / 5))
    
    history = G.iterate(psi_0, n_iterations=10, theta_func=theta_func, p_jump_func=p_jump_func)
    
    for i, state in enumerate(history[::2]):  # 2ステップごと
        P_1 = np.abs(state[1])**2
        print(f"    n={i*2}: P(|1⟩) = {P_1:.3f}")
    
    # 4. ハザード率の時間発展
    print("\n[4] 相転移ハザード率")
    tau_range = np.linspace(0, 10, 50)
    p_history = evolve_hazard_rate(0.1, tau_range, gamma=0.3, beta=0.2, omega=1.0)
    
    for i in [0, 12, 24, 49]:
        print(f"    τ={tau_range[i]:.1f}: p = {p_history[i]:.3f}")
    
    # 5. 連続-離散補間
    print("\n[5] 連続-離散補間")
    
    # 連続演算子（回転）
    U_cont = expm(-1j * np.pi/4 * np.array([[1, 0], [0, -1]]))
    
    # 離散演算子
    G_disc = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=complex)
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        U_interp = interpolate_continuous_discrete(U_cont, G_disc, alpha)
        result = U_interp @ psi_0
        P_1 = np.abs(result[1])**2
        print(f"    α={alpha:.2f}: P(|1⟩) = {P_1:.3f}")
    
    # 6. 待機光子安定化
    print("\n[6] 待機光子安定化モデル")
    stabilizer = StandbyPhotonStabilizer(n_photons=5, coupling=0.1)
    
    n_op = stabilizer.number_operator()
    print(f"    光子数演算子の次元: {n_op.shape}")
    print(f"    ⟨n⟩ (真空): {np.trace(n_op @ np.outer([1,0,0,0,0,0], [1,0,0,0,0,0])):.1f}")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
