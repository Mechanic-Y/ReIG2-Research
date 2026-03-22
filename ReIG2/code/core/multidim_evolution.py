"""
ReIG2/twinRIG 第3章
多次元時間発展演算子
Multidimensional Time Evolution Operator (Û_multi)

Mechanic-Y / Yasuyuki Wakita
2025年12月

複数の時間軸（物理的・文化的・社会的・個人的）を同時に扱う時間発展
Trotter-Suzuki分解による非可換ハミルトニアンの取り扱い
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# 定数
# =============================================================================

HBAR = 1.0  # 自然単位系


# =============================================================================
# 時間軸の定義
# =============================================================================

class TimeAxis(Enum):
    """時間軸の種類"""
    PHYSICAL = "physical"      # 物理的時間
    CULTURAL = "cultural"      # 文化的時間
    SOCIAL = "social"          # 社会的時間
    PERSONAL = "personal"      # 個人的時間


@dataclass
class TimeAxisConfig:
    """時間軸の設定"""
    axis: TimeAxis
    hamiltonian: np.ndarray
    weight_func: Callable[[float, float, float], float]  # f_k(τ, ε, PFH)
    alpha: float = 1.0   # 重み係数
    beta: float = 1.0    # べき指数
    sigma: float = 1.0   # ガウス幅
    gamma: float = 1.0   # PFH係数


# =============================================================================
# 重み付け関数
# =============================================================================

def physical_weight(tau: float, epsilon: float, PFH: float) -> float:
    """物理的時間軸の重み: f_1 = 1 + τ"""
    return 1 + tau


def cultural_weight(tau: float, epsilon: float, PFH: float, sigma_c: float = 1.0) -> float:
    """文化的時間軸の重み: f_2 = τ · exp(-ε²/2σ_c²)"""
    return tau * np.exp(-epsilon**2 / (2 * sigma_c**2))


def social_weight(tau: float, epsilon: float, PFH: float) -> float:
    """社会的時間軸の重み: f_3 = PFH · τ^(1/2)"""
    return PFH * np.sqrt(max(tau, 0))


def personal_weight(tau: float, epsilon: float, PFH: float, omega_p: float = 1.0) -> float:
    """個人的時間軸の重み: f_4 = ε · sin(ω_p τ)"""
    return epsilon * np.sin(omega_p * tau)


def general_weight(
    tau: float, epsilon: float, PFH: float,
    alpha: float, beta: float, sigma: float, gamma: float
) -> float:
    """
    一般的な重み付け関数
    
    f_k(τ, ε, PFH) = α · τ^β · exp(-ε²/2σ²) · (1 + γ·PFH)
    """
    return alpha * (tau ** beta) * np.exp(-epsilon**2 / (2 * sigma**2)) * (1 + gamma * PFH)


# =============================================================================
# Trotter-Suzuki分解
# =============================================================================

def trotter_first_order(
    hamiltonians: List[np.ndarray],
    weights: List[float],
    dt: float,
    M: int = 1
) -> np.ndarray:
    """
    1次Trotter分解
    
    U(t) ≈ [∏_k exp(-i H_k f_k dt/M)]^M
    
    Args:
        hamiltonians: ハミルトニアンのリスト
        weights: 各ハミルトニアンの重み f_k
        dt: 時間刻み
        M: 分割数
    
    Returns:
        近似された時間発展演算子
    """
    dim = hamiltonians[0].shape[0]
    U_step = np.eye(dim, dtype=complex)
    
    dt_M = dt / M
    
    for H, f in zip(hamiltonians, weights):
        U_step = expm(-1j * H * f * dt_M / HBAR) @ U_step
    
    # M乗
    U = np.linalg.matrix_power(U_step, M)
    
    return U


def trotter_second_order(
    hamiltonians: List[np.ndarray],
    weights: List[float],
    dt: float,
    M: int = 1
) -> np.ndarray:
    """
    2次対称Trotter分解（Strang分割）
    
    U(t) ≈ [∏_{k=1}^K exp(-i H_k f_k dt/2M) ∏_{k=K}^1 exp(-i H_k f_k dt/2M)]^M
    
    誤差: O(dt³)
    """
    dim = hamiltonians[0].shape[0]
    dt_M = dt / M
    
    # 前半: 順方向
    U_forward = np.eye(dim, dtype=complex)
    for H, f in zip(hamiltonians, weights):
        U_forward = expm(-1j * H * f * dt_M / (2 * HBAR)) @ U_forward
    
    # 後半: 逆方向
    U_backward = np.eye(dim, dtype=complex)
    for H, f in zip(reversed(hamiltonians), reversed(weights)):
        U_backward = expm(-1j * H * f * dt_M / (2 * HBAR)) @ U_backward
    
    # 1ステップ
    U_step = U_backward @ U_forward
    
    # M乗
    U = np.linalg.matrix_power(U_step, M)
    
    return U


def trotter_fourth_order(
    hamiltonians: List[np.ndarray],
    weights: List[float],
    dt: float,
    M: int = 1
) -> np.ndarray:
    """
    4次Suzuki分解
    
    誤差: O(dt⁵)
    """
    # Suzukiの係数
    s = 1 / (4 - 4**(1/3))
    
    # 4次分解
    U1 = trotter_second_order(hamiltonians, weights, s * dt, M)
    U2 = trotter_second_order(hamiltonians, weights, s * dt, M)
    U3 = trotter_second_order(hamiltonians, weights, (1 - 4*s) * dt, M)
    
    return U1 @ U2 @ U3 @ U2 @ U1


# =============================================================================
# 非可換性の解析
# =============================================================================

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """交換子 [A, B] = AB - BA"""
    return A @ B - B @ A


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """交換子のノルム ||[A, B]||"""
    return np.linalg.norm(commutator(A, B), ord=2)


def estimate_trotter_error(
    hamiltonians: List[np.ndarray],
    weights: List[float],
    t: float,
    M: int
) -> float:
    """
    Trotter分解の誤差を推定
    
    ||U_exact - U_Trotter|| ≤ C · t²/M · Σ_{k<k'} ||[H_k, H_k']||
    """
    total_commutator = 0.0
    K = len(hamiltonians)
    
    for i in range(K):
        for j in range(i + 1, K):
            H_i = weights[i] * hamiltonians[i]
            H_j = weights[j] * hamiltonians[j]
            total_commutator += commutator_norm(H_i, H_j)
    
    C = 0.5  # 定数（保守的な推定）
    error = C * (t ** 2) / M * total_commutator
    
    return error


# =============================================================================
# 多次元時間発展演算子 Û_multi
# =============================================================================

class MultidimensionalEvolutionOperator:
    """
    多次元時間発展演算子 Û_multi
    
    Û_multi(τ, ε, PFH) = exp(-i Σ_k H_k f_k(τ, ε, PFH) / ℏ)
    """
    
    def __init__(self, dim: int = 2):
        """
        Args:
            dim: ヒルベルト空間の次元
        """
        self.dim = dim
        self.axes: List[TimeAxisConfig] = []
        self._setup_default_axes()
    
    def _setup_default_axes(self):
        """デフォルトの時間軸を設定"""
        # Pauli行列（2次元の場合）
        if self.dim == 2:
            sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            
            # 物理的時間軸
            self.add_axis(TimeAxisConfig(
                axis=TimeAxis.PHYSICAL,
                hamiltonian=sigma_z,
                weight_func=physical_weight
            ))
            
            # 文化的時間軸
            self.add_axis(TimeAxisConfig(
                axis=TimeAxis.CULTURAL,
                hamiltonian=0.5 * sigma_x,
                weight_func=cultural_weight
            ))
            
            # 社会的時間軸
            self.add_axis(TimeAxisConfig(
                axis=TimeAxis.SOCIAL,
                hamiltonian=0.3 * (sigma_x + sigma_z) / np.sqrt(2),
                weight_func=social_weight
            ))
            
            # 個人的時間軸
            self.add_axis(TimeAxisConfig(
                axis=TimeAxis.PERSONAL,
                hamiltonian=0.2 * sigma_y,
                weight_func=personal_weight
            ))
    
    def add_axis(self, config: TimeAxisConfig):
        """時間軸を追加"""
        self.axes.append(config)
    
    def get_weights(self, tau: float, epsilon: float, PFH: float) -> List[float]:
        """各時間軸の重みを計算"""
        weights = []
        for axis in self.axes:
            w = axis.weight_func(tau, epsilon, PFH)
            weights.append(w)
        return weights
    
    def get_hamiltonians(self) -> List[np.ndarray]:
        """ハミルトニアンのリストを取得"""
        return [axis.hamiltonian for axis in self.axes]
    
    def total_hamiltonian(self, tau: float, epsilon: float, PFH: float) -> np.ndarray:
        """全ハミルトニアンを計算"""
        hamiltonians = self.get_hamiltonians()
        weights = self.get_weights(tau, epsilon, PFH)
        
        H_total = np.zeros((self.dim, self.dim), dtype=complex)
        for H, f in zip(hamiltonians, weights):
            H_total += H * f
        
        return H_total
    
    def operator_exact(self, t: float, tau: float, epsilon: float, PFH: float) -> np.ndarray:
        """厳密な時間発展演算子（全ハミルトニアンを直接指数化）"""
        H_total = self.total_hamiltonian(tau, epsilon, PFH)
        return expm(-1j * H_total * t / HBAR)
    
    def operator_trotter(
        self,
        t: float,
        tau: float,
        epsilon: float,
        PFH: float,
        M: int = 10,
        order: int = 2
    ) -> np.ndarray:
        """
        Trotter分解による時間発展演算子
        
        Args:
            t: 時間
            tau, epsilon, PFH: 共鳴パラメータ
            M: 分割数
            order: Trotterの次数 (1, 2, or 4)
        """
        hamiltonians = self.get_hamiltonians()
        weights = self.get_weights(tau, epsilon, PFH)
        
        if order == 1:
            return trotter_first_order(hamiltonians, weights, t, M)
        elif order == 2:
            return trotter_second_order(hamiltonians, weights, t, M)
        elif order == 4:
            return trotter_fourth_order(hamiltonians, weights, t, M)
        else:
            raise ValueError(f"Unsupported Trotter order: {order}")
    
    def evolve(
        self,
        initial_state: np.ndarray,
        t: float,
        tau: float,
        epsilon: float,
        PFH: float,
        method: str = "exact"
    ) -> Tuple[np.ndarray, Dict]:
        """
        状態を時間発展させる
        
        Args:
            initial_state: 初期状態
            t: 時間
            tau, epsilon, PFH: 共鳴パラメータ
            method: "exact" or "trotter"
        
        Returns:
            (final_state, info)
        """
        if method == "exact":
            U = self.operator_exact(t, tau, epsilon, PFH)
        else:
            U = self.operator_trotter(t, tau, epsilon, PFH)
        
        final_state = U @ initial_state
        final_state = final_state / np.linalg.norm(final_state)
        
        info = {
            "weights": self.get_weights(tau, epsilon, PFH),
            "method": method
        }
        
        return final_state, info
    
    def analyze_noncommutativity(self, tau: float, epsilon: float, PFH: float) -> Dict:
        """非可換性の解析"""
        hamiltonians = self.get_hamiltonians()
        weights = self.get_weights(tau, epsilon, PFH)
        
        results = {
            "commutator_norms": [],
            "axis_pairs": []
        }
        
        K = len(hamiltonians)
        for i in range(K):
            for j in range(i + 1, K):
                H_i = weights[i] * hamiltonians[i]
                H_j = weights[j] * hamiltonians[j]
                norm = commutator_norm(H_i, H_j)
                
                results["commutator_norms"].append(norm)
                results["axis_pairs"].append(
                    (self.axes[i].axis.value, self.axes[j].axis.value)
                )
        
        results["total_noncommutativity"] = sum(results["commutator_norms"])
        
        return results


# =============================================================================
# 幾何学的解釈
# =============================================================================

def parameter_space_metric(
    operator: MultidimensionalEvolutionOperator,
    tau: float, epsilon: float, PFH: float,
    delta: float = 1e-5
) -> np.ndarray:
    """
    パラメータ空間の計量テンソルを数値的に計算
    
    g_μν = Σ_k (∂f_k/∂x^μ)(∂f_k/∂x^ν) ||H_k||²
    """
    params = [tau, epsilon, PFH]
    param_names = ["tau", "epsilon", "PFH"]
    
    g = np.zeros((3, 3))
    
    hamiltonians = operator.get_hamiltonians()
    H_norms = [np.linalg.norm(H, ord=2)**2 for H in hamiltonians]
    
    for mu in range(3):
        for nu in range(3):
            for k, axis in enumerate(operator.axes):
                # 数値微分で ∂f_k/∂x^μ を計算
                params_plus_mu = params.copy()
                params_plus_mu[mu] += delta
                
                params_minus_mu = params.copy()
                params_minus_mu[mu] -= delta
                
                f_plus_mu = axis.weight_func(*params_plus_mu)
                f_minus_mu = axis.weight_func(*params_minus_mu)
                df_dmu = (f_plus_mu - f_minus_mu) / (2 * delta)
                
                # ∂f_k/∂x^ν
                params_plus_nu = params.copy()
                params_plus_nu[nu] += delta
                
                params_minus_nu = params.copy()
                params_minus_nu[nu] -= delta
                
                f_plus_nu = axis.weight_func(*params_plus_nu)
                f_minus_nu = axis.weight_func(*params_minus_nu)
                df_dnu = (f_plus_nu - f_minus_nu) / (2 * delta)
                
                g[mu, nu] += df_dmu * df_dnu * H_norms[k]
    
    return g


# =============================================================================
# デモ
# =============================================================================

def demo():
    """多次元時間発展演算子のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第3章")
    print("多次元時間発展演算子 Û_multi")
    print("=" * 60)
    
    # 演算子の初期化
    U_multi = MultidimensionalEvolutionOperator(dim=2)
    
    print(f"\n[1] 時間軸の構成")
    for axis in U_multi.axes:
        print(f"    {axis.axis.value}: H_norm = {np.linalg.norm(axis.hamiltonian):.3f}")
    
    # 初期状態
    psi_0 = np.array([1, 0], dtype=complex)  # |0⟩
    
    # 2. 重み付け関数
    print("\n[2] 重み付け関数 f_k(τ, ε, PFH)")
    tau, epsilon, PFH = 0.5, 0.3, 0.2
    weights = U_multi.get_weights(tau, epsilon, PFH)
    
    for axis, w in zip(U_multi.axes, weights):
        print(f"    {axis.axis.value}: f = {w:.3f}")
    
    # 3. 非可換性解析
    print("\n[3] 非可換性解析")
    noncomm = U_multi.analyze_noncommutativity(tau, epsilon, PFH)
    
    for pair, norm in zip(noncomm["axis_pairs"], noncomm["commutator_norms"]):
        print(f"    [{pair[0]}, {pair[1]}]: ||[H_i, H_j]|| = {norm:.4f}")
    print(f"    総非可換性: {noncomm['total_noncommutativity']:.4f}")
    
    # 4. 厳密解 vs Trotter
    print("\n[4] 厳密解 vs Trotter分解")
    t = np.pi
    
    U_exact = U_multi.operator_exact(t, tau, epsilon, PFH)
    
    for M in [1, 5, 10, 50]:
        U_trotter = U_multi.operator_trotter(t, tau, epsilon, PFH, M=M, order=2)
        error = np.linalg.norm(U_exact - U_trotter, ord=2)
        print(f"    M={M:2d}: ||U_exact - U_trotter|| = {error:.6f}")
    
    # 5. 時間発展
    print("\n[5] 時間発展")
    
    param_sets = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.3, 0.0),
        (0.5, 0.3, 0.5),
    ]
    
    for tau, epsilon, PFH in param_sets:
        final_state, _ = U_multi.evolve(psi_0, np.pi/2, tau, epsilon, PFH)
        P_1 = np.abs(final_state[1])**2
        print(f"    τ={tau:.1f}, ε={epsilon:.1f}, PFH={PFH:.1f}: P(|1⟩) = {P_1:.3f}")
    
    # 6. Trotter誤差推定
    print("\n[6] Trotter誤差推定")
    hamiltonians = U_multi.get_hamiltonians()
    weights = U_multi.get_weights(0.5, 0.3, 0.2)
    
    for M in [1, 10, 100]:
        error = estimate_trotter_error(hamiltonians, weights, np.pi, M)
        print(f"    M={M:3d}: 推定誤差上界 = {error:.6f}")
    
    # 7. パラメータ空間の計量
    print("\n[7] パラメータ空間の計量テンソル")
    g = parameter_space_metric(U_multi, 0.5, 0.3, 0.2)
    print(f"    g_ττ = {g[0,0]:.4f}")
    print(f"    g_εε = {g[1,1]:.4f}")
    print(f"    g_PP = {g[2,2]:.4f}")
    print(f"    det(g) = {np.linalg.det(g):.6f}")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
