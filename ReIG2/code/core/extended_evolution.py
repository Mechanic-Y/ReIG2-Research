"""
ReIG2/twinRIG 第2章
拡張時間発展演算子
Extended Time Evolution Operator (Û_res)

Mechanic-Y / Yasuyuki Wakita
2025年12月

標準量子力学の時間発展演算子を3パラメータ（τ, ε, PFH）で拡張
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass

# =============================================================================
# 定数・Pauli行列
# =============================================================================

# Pauli行列
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# 物理定数（自然単位系: ℏ = 1）
HBAR = 1.0


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class ResonanceParameters:
    """共鳴パラメータ"""
    tau: float      # 時間共鳴（未来寄与度）
    epsilon: float  # エントロピー共鳴（揺らぎ度合い）
    PFH: float      # 哲学的共鳴（倫理・調和係数）
    
    def to_dict(self) -> Dict:
        return {"tau": self.tau, "epsilon": self.epsilon, "PFH": self.PFH}


@dataclass
class EvolutionResult:
    """時間発展の結果"""
    initial_state: np.ndarray
    final_state: np.ndarray
    operator: np.ndarray
    parameters: ResonanceParameters
    time: float
    observables: Dict[str, float]


# =============================================================================
# ハミルトニアン構成
# =============================================================================

def build_base_hamiltonian(
    omega_0: float = 1.0,
    dim: int = 2
) -> np.ndarray:
    """
    基本ハミルトニアン H_0 を構築
    
    2準位系: H_0 = ω_0 σ_z
    
    Args:
        omega_0: 基本周波数
        dim: ヒルベルト空間の次元（2 or higher）
    
    Returns:
        H_0: 基本ハミルトニアン
    """
    if dim == 2:
        return omega_0 * SIGMA_Z
    else:
        # 高次元の場合：対角行列
        eigenvalues = np.linspace(-omega_0, omega_0, dim)
        return np.diag(eigenvalues)


def build_future_hamiltonian(
    omega_f: float = 0.5,
    dim: int = 2
) -> np.ndarray:
    """
    未来寄与項 H_future を構築
    
    未来の可能性を現在に反映する項
    2準位系: H_future = ω_f σ_x（重ね合わせ生成）
    
    Args:
        omega_f: 未来結合強度
        dim: 次元
    
    Returns:
        H_future
    """
    if dim == 2:
        return omega_f * SIGMA_X
    else:
        # 高次元：隣接準位間の結合
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(dim - 1):
            H[i, i+1] = omega_f
            H[i+1, i] = omega_f
        return H


def build_entropy_hamiltonian(
    omega_e: float = 0.3,
    dim: int = 2
) -> np.ndarray:
    """
    エントロピー項 H_entropy を構築
    
    揺らぎ・不確定性を導入する項
    2準位系: H_entropy = ω_e σ_y（位相回転）
    
    Args:
        omega_e: エントロピー結合強度
        dim: 次元
    
    Returns:
        H_entropy
    """
    if dim == 2:
        return omega_e * SIGMA_Y
    else:
        # 高次元：反対称結合
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(dim - 1):
            H[i, i+1] = -1j * omega_e
            H[i+1, i] = 1j * omega_e
        return H


def build_ethics_hamiltonian(
    V_eth: float = 0.2,
    dim: int = 2
) -> np.ndarray:
    """
    倫理項 H_ethics を構築
    
    調和・不調和のポテンシャル
    
    Args:
        V_eth: 倫理ポテンシャル強度
        dim: 次元
    
    Returns:
        H_ethics
    """
    if dim == 2:
        # スカラーポテンシャルとして対角成分に加算
        return V_eth * IDENTITY
    else:
        return V_eth * np.eye(dim, dtype=complex)


def build_extended_hamiltonian(
    t: float,
    params: ResonanceParameters,
    omega_0: float = 1.0,
    omega_f: float = 0.5,
    omega_e: float = 0.3,
    V_eth: float = 0.2,
    dim: int = 2
) -> np.ndarray:
    """
    拡張ハミルトニアンを構築
    
    H(t, τ, ε, PFH) = H_0(t) + τ·H_future + ε·H_entropy + PFH·H_ethics
    
    Args:
        t: 時刻
        params: 共鳴パラメータ
        omega_0, omega_f, omega_e, V_eth: 各項の結合定数
        dim: 次元
    
    Returns:
        H: 拡張ハミルトニアン
    """
    H_0 = build_base_hamiltonian(omega_0, dim)
    H_future = build_future_hamiltonian(omega_f, dim)
    H_entropy = build_entropy_hamiltonian(omega_e, dim)
    H_ethics = build_ethics_hamiltonian(V_eth, dim)
    
    # 拡張ハミルトニアン
    H = H_0 + params.tau * H_future + params.epsilon * H_entropy + params.PFH * H_ethics
    
    return H


# =============================================================================
# 拡張時間発展演算子 Û_res
# =============================================================================

class ExtendedEvolutionOperator:
    """
    拡張時間発展演算子 Û_res
    
    Û_res(t; τ, ε, PFH) = exp(-i H(t, τ, ε, PFH) t / ℏ)
    """
    
    def __init__(
        self,
        omega_0: float = 1.0,
        omega_f: float = 0.5,
        omega_e: float = 0.3,
        V_eth: float = 0.2,
        dim: int = 2
    ):
        """
        Args:
            omega_0: 基本周波数
            omega_f: 未来結合強度
            omega_e: エントロピー結合強度
            V_eth: 倫理ポテンシャル
            dim: ヒルベルト空間の次元
        """
        self.omega_0 = omega_0
        self.omega_f = omega_f
        self.omega_e = omega_e
        self.V_eth = V_eth
        self.dim = dim
    
    def hamiltonian(self, t: float, params: ResonanceParameters) -> np.ndarray:
        """ハミルトニアンを取得"""
        return build_extended_hamiltonian(
            t, params, self.omega_0, self.omega_f, self.omega_e, self.V_eth, self.dim
        )
    
    def operator(self, t: float, params: ResonanceParameters) -> np.ndarray:
        """
        時間発展演算子を計算
        
        Û_res = exp(-i H t / ℏ)
        """
        H = self.hamiltonian(t, params)
        U = expm(-1j * H * t / HBAR)
        return U
    
    def evolve(
        self,
        initial_state: np.ndarray,
        t: float,
        params: ResonanceParameters
    ) -> EvolutionResult:
        """
        状態を時間発展させる
        
        |Ψ(t)⟩ = Û_res |Ψ(0)⟩
        
        Args:
            initial_state: 初期状態
            t: 発展時間
            params: 共鳴パラメータ
        
        Returns:
            EvolutionResult
        """
        U = self.operator(t, params)
        final_state = U @ initial_state
        
        # 正規化
        final_state = final_state / np.linalg.norm(final_state)
        
        # 観測量の計算
        observables = self._compute_observables(final_state)
        
        return EvolutionResult(
            initial_state=initial_state,
            final_state=final_state,
            operator=U,
            parameters=params,
            time=t,
            observables=observables
        )
    
    def evolve_trajectory(
        self,
        initial_state: np.ndarray,
        t_final: float,
        params: ResonanceParameters,
        n_steps: int = 100
    ) -> List[EvolutionResult]:
        """
        時間発展の軌跡を計算
        """
        trajectory = []
        dt = t_final / n_steps
        state = initial_state.copy()
        
        for i in range(n_steps + 1):
            t = i * dt
            result = self.evolve(state, dt if i > 0 else 0, params)
            result.time = t
            trajectory.append(result)
            state = result.final_state
        
        return trajectory
    
    def _compute_observables(self, state: np.ndarray) -> Dict[str, float]:
        """観測量を計算"""
        rho = np.outer(state, state.conj())
        
        observables = {}
        
        if self.dim == 2:
            # Pauli期待値
            observables["sigma_x"] = np.real(np.trace(rho @ SIGMA_X))
            observables["sigma_y"] = np.real(np.trace(rho @ SIGMA_Y))
            observables["sigma_z"] = np.real(np.trace(rho @ SIGMA_Z))
            
            # 確率
            observables["P_0"] = np.abs(state[0])**2
            observables["P_1"] = np.abs(state[1])**2
        
        # エントロピー
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        observables["entropy"] = -np.sum(eigenvalues * np.log(eigenvalues))
        
        return observables


# =============================================================================
# ユニタリ性の検証
# =============================================================================

def verify_unitarity(U: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    演算子のユニタリ性を検証
    
    U†U = I であることを確認
    """
    product = U.conj().T @ U
    identity = np.eye(U.shape[0])
    
    return np.allclose(product, identity, atol=tolerance)


def verify_hermiticity(H: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    ハミルトニアンのエルミート性を検証
    
    H† = H であることを確認
    """
    return np.allclose(H, H.conj().T, atol=tolerance)


# =============================================================================
# パラメータ依存性の解析
# =============================================================================

def parameter_scan(
    operator: ExtendedEvolutionOperator,
    initial_state: np.ndarray,
    t: float,
    param_name: str,
    param_range: np.ndarray,
    fixed_params: Dict[str, float]
) -> Dict[str, List]:
    """
    パラメータをスキャンして観測量の変化を調べる
    
    Args:
        operator: 時間発展演算子
        initial_state: 初期状態
        t: 時間
        param_name: スキャンするパラメータ名
        param_range: パラメータの範囲
        fixed_params: 固定するパラメータ
    
    Returns:
        各パラメータ値での観測量
    """
    results = {
        "param_values": list(param_range),
        "P_1": [],
        "sigma_z": [],
        "entropy": []
    }
    
    for val in param_range:
        params_dict = fixed_params.copy()
        params_dict[param_name] = val
        params = ResonanceParameters(**params_dict)
        
        result = operator.evolve(initial_state, t, params)
        
        results["P_1"].append(result.observables.get("P_1", 0))
        results["sigma_z"].append(result.observables.get("sigma_z", 0))
        results["entropy"].append(result.observables.get("entropy", 0))
    
    return results


# =============================================================================
# デモ
# =============================================================================

def demo():
    """拡張時間発展演算子のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第2章")
    print("拡張時間発展演算子 Û_res")
    print("=" * 60)
    
    # 演算子の初期化
    U_res = ExtendedEvolutionOperator(
        omega_0=1.0, omega_f=0.5, omega_e=0.3, V_eth=0.2
    )
    
    # 初期状態: |+⟩ = (|0⟩ + |1⟩)/√2
    psi_0 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    
    print(f"\n[1] 初期状態: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"    |0⟩成分: {np.abs(psi_0[0])**2:.3f}")
    print(f"    |1⟩成分: {np.abs(psi_0[1])**2:.3f}")
    
    # 2. ハミルトニアンの検証
    print("\n[2] ハミルトニアンの構造")
    params = ResonanceParameters(tau=0.5, epsilon=0.3, PFH=0.2)
    H = U_res.hamiltonian(0, params)
    
    print(f"    パラメータ: τ={params.tau}, ε={params.epsilon}, PFH={params.PFH}")
    print(f"    エルミート性: {verify_hermiticity(H)}")
    print(f"    固有値: {np.linalg.eigvalsh(H)}")
    
    # 3. 時間発展
    print("\n[3] 時間発展")
    
    test_cases = [
        {"tau": 0, "epsilon": 0, "PFH": 0},
        {"tau": 0.5, "epsilon": 0, "PFH": 0},
        {"tau": 0, "epsilon": 0.5, "PFH": 0},
        {"tau": 0, "epsilon": 0, "PFH": 0.5},
        {"tau": 0.5, "epsilon": 0.3, "PFH": 0.2},
    ]
    
    t = np.pi / 2
    
    for case in test_cases:
        params = ResonanceParameters(**case)
        result = U_res.evolve(psi_0, t, params)
        
        print(f"    τ={case['tau']:.1f}, ε={case['epsilon']:.1f}, PFH={case['PFH']:.1f}: "
              f"P(|1⟩)={result.observables['P_1']:.3f}, "
              f"⟨σ_z⟩={result.observables['sigma_z']:.3f}")
    
    # 4. ユニタリ性検証
    print("\n[4] ユニタリ性検証")
    params = ResonanceParameters(tau=1.0, epsilon=0.5, PFH=0.5)
    U = U_res.operator(np.pi, params)
    print(f"    U†U = I: {verify_unitarity(U)}")
    
    # 5. パラメータスキャン
    print("\n[5] τパラメータスキャン (t=π)")
    
    tau_range = np.linspace(0, 2, 5)
    scan_results = parameter_scan(
        U_res, psi_0, np.pi,
        param_name="tau",
        param_range=tau_range,
        fixed_params={"epsilon": 0.3, "PFH": 0.2}
    )
    
    for i, tau in enumerate(tau_range):
        print(f"    τ={tau:.2f}: P(|1⟩)={scan_results['P_1'][i]:.3f}")
    
    # 6. 時間発展軌跡
    print("\n[6] 時間発展軌跡")
    params = ResonanceParameters(tau=0.5, epsilon=0.3, PFH=0.2)
    trajectory = U_res.evolve_trajectory(psi_0, 2*np.pi, params, n_steps=4)
    
    for result in trajectory:
        print(f"    t={result.time:.2f}: P(|1⟩)={result.observables['P_1']:.3f}")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
