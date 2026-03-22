"""
ReIG2/twinRIG 第5章
世界演算子の統合実装
World Generation Operator (T̂_World)

Mechanic-Y / Yasuyuki Wakita
2025年12月

多部分系の量子フレームワークと世界構築演算子の統合
T̂_World = T̂_I ∘ T̂_R ∘ T̂_C ∘ Û_multi ∘ Û_res
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# 定数・ヒルベルト空間
# =============================================================================

HBAR = 1.0


class HilbertSpace(Enum):
    """ヒルベルト空間の種類"""
    MEANING = "M"       # 意味空間 H_M
    CONTEXT = "C"       # 文脈空間 H_C
    ETHICS = "E"        # 倫理空間 H_E
    FUTURE = "F"        # 未来空間 H_F
    STABILITY = "S"     # 安定性空間 H_S
    OBSERVATION = "O"   # 観察空間 H_O
    QUESTION = "Q"      # 質問空間 H_Q


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class WorldState:
    """世界状態"""
    state_vector: np.ndarray
    subspace_dims: Dict[str, int]
    coherence: float
    iteration: int


@dataclass
class TransformResult:
    """変換結果"""
    input_state: np.ndarray
    output_state: np.ndarray
    transform_name: str
    fidelity: float


# =============================================================================
# サブシステムハミルトニアン
# =============================================================================

def build_meaning_hamiltonian(dim: int = 4) -> np.ndarray:
    """
    意味ハミルトニアン H_M
    
    意味の階層構造を表現
    """
    H = np.zeros((dim, dim), dtype=complex)
    
    # 意味の階層エネルギー
    for i in range(dim):
        H[i, i] = i * 1.0
    
    # 隣接意味間の結合
    for i in range(dim - 1):
        H[i, i+1] = 0.5
        H[i+1, i] = 0.5
    
    return H


def build_context_hamiltonian(dim: int = 4) -> np.ndarray:
    """
    文脈ハミルトニアン H_C
    
    文脈の遷移を表現
    """
    H = np.zeros((dim, dim), dtype=complex)
    
    # 文脈間の結合（周期境界）
    for i in range(dim):
        H[i, (i+1) % dim] = 0.3
        H[(i+1) % dim, i] = 0.3
    
    return H


def build_ethics_hamiltonian(dim: int = 4) -> np.ndarray:
    """
    倫理ハミルトニアン H_E
    
    倫理的調和のポテンシャル
    """
    H = np.zeros((dim, dim), dtype=complex)
    
    # 対角成分：倫理レベル
    for i in range(dim):
        H[i, i] = -np.cos(2 * np.pi * i / dim)  # 周期的ポテンシャル
    
    return H


def build_future_hamiltonian(dim: int = 4) -> np.ndarray:
    """
    未来ハミルトニアン H_F
    
    未来可能性の生成
    """
    H = np.zeros((dim, dim), dtype=complex)
    
    # 非対角成分：未来への遷移
    for i in range(dim - 1):
        H[i, i+1] = 0.4 * np.exp(-i / dim)  # 減衰する結合
        H[i+1, i] = 0.4 * np.exp(-i / dim)
    
    return H


def build_stability_hamiltonian(dim: int = 4) -> np.ndarray:
    """
    安定性ハミルトニアン H_S
    
    系の安定化
    """
    H = np.zeros((dim, dim), dtype=complex)
    
    # 対角成分：安定点
    for i in range(dim):
        H[i, i] = (i - dim/2) ** 2 / dim  # 調和ポテンシャル
    
    return H


# =============================================================================
# 世界生成テンソル体系
# =============================================================================

class WorldTensorSystem:
    """
    世界生成テンソル体系
    
    H_sys = H_M ⊗ H_C ⊗ H_E ⊗ H_F ⊗ H_S
    """
    
    def __init__(
        self,
        dim_M: int = 2,
        dim_C: int = 2,
        dim_E: int = 2,
        dim_F: int = 2,
        dim_S: int = 2
    ):
        """
        Args:
            dim_M: 意味空間の次元
            dim_C: 文脈空間の次元
            dim_E: 倫理空間の次元
            dim_F: 未来空間の次元
            dim_S: 安定性空間の次元
        """
        self.dims = {
            "M": dim_M, "C": dim_C, "E": dim_E, "F": dim_F, "S": dim_S
        }
        self.total_dim = dim_M * dim_C * dim_E * dim_F * dim_S
        
        # サブシステムハミルトニアン
        self.H_M = build_meaning_hamiltonian(dim_M)
        self.H_C = build_context_hamiltonian(dim_C)
        self.H_E = build_ethics_hamiltonian(dim_E)
        self.H_F = build_future_hamiltonian(dim_F)
        self.H_S = build_stability_hamiltonian(dim_S)
    
    def total_hamiltonian(self) -> np.ndarray:
        """
        全体ハミルトニアン
        
        H_sys = H_M ⊗ I ⊗ I ⊗ I ⊗ I + I ⊗ H_C ⊗ I ⊗ I ⊗ I + ...
        """
        H_total = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        
        # 各サブシステムの寄与
        I_M = np.eye(self.dims["M"])
        I_C = np.eye(self.dims["C"])
        I_E = np.eye(self.dims["E"])
        I_F = np.eye(self.dims["F"])
        I_S = np.eye(self.dims["S"])
        
        # H_M ⊗ I ⊗ I ⊗ I ⊗ I
        H_total += np.kron(np.kron(np.kron(np.kron(self.H_M, I_C), I_E), I_F), I_S)
        
        # I ⊗ H_C ⊗ I ⊗ I ⊗ I
        H_total += np.kron(np.kron(np.kron(np.kron(I_M, self.H_C), I_E), I_F), I_S)
        
        # I ⊗ I ⊗ H_E ⊗ I ⊗ I
        H_total += np.kron(np.kron(np.kron(np.kron(I_M, I_C), self.H_E), I_F), I_S)
        
        # I ⊗ I ⊗ I ⊗ H_F ⊗ I
        H_total += np.kron(np.kron(np.kron(np.kron(I_M, I_C), I_E), self.H_F), I_S)
        
        # I ⊗ I ⊗ I ⊗ I ⊗ H_S
        H_total += np.kron(np.kron(np.kron(np.kron(I_M, I_C), I_E), I_F), self.H_S)
        
        return H_total
    
    def partial_trace(self, rho: np.ndarray, subsystems_to_trace: List[str]) -> np.ndarray:
        """
        部分トレース
        
        指定されたサブシステムをトレースアウト
        """
        # 簡略化された実装（完全な実装は複雑）
        # ここでは2サブシステムの場合のみ
        raise NotImplementedError("Full partial trace not implemented")
    
    def random_state(self) -> np.ndarray:
        """ランダムな正規化状態を生成"""
        state = np.random.randn(self.total_dim) + 1j * np.random.randn(self.total_dim)
        return state / np.linalg.norm(state)
    
    def product_state(self, *substates) -> np.ndarray:
        """積状態を生成"""
        result = substates[0]
        for s in substates[1:]:
            result = np.kron(result, s)
        return result / np.linalg.norm(result)


# =============================================================================
# 変換演算子 T̂_C, T̂_R, T̂_I
# =============================================================================

class CognitionTransform:
    """
    認知変換 T̂_C
    
    観察から認知への変換
    """
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def operator(self, alpha: float = 0.5) -> np.ndarray:
        """
        認知変換演算子
        
        注意機構を模倣した選択的変換
        """
        T = np.eye(self.dim, dtype=complex)
        
        # 対角成分：認知の強さ
        for i in range(self.dim):
            T[i, i] = np.exp(-alpha * i / self.dim)
        
        # 正規化
        T = T / np.linalg.norm(T, ord=2)
        
        return T
    
    def apply(self, state: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """変換を適用"""
        T = self.operator(alpha)
        result = T @ state
        return result / np.linalg.norm(result)


class RecognitionTransform:
    """
    認識変換 T̂_R
    
    認知から認識への変換（パターン認識）
    """
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def operator(self, beta: float = 0.3) -> np.ndarray:
        """
        認識変換演算子
        
        パターンマッチングを模倣
        """
        T = np.zeros((self.dim, self.dim), dtype=complex)
        
        # パターン認識行列
        for i in range(self.dim):
            for j in range(self.dim):
                T[i, j] = np.exp(-beta * abs(i - j))
        
        # 正規化
        T = T / np.linalg.norm(T, ord=2)
        
        return T
    
    def apply(self, state: np.ndarray, beta: float = 0.3) -> np.ndarray:
        """変換を適用"""
        T = self.operator(beta)
        result = T @ state
        return result / np.linalg.norm(result)


class IntegrationTransform:
    """
    統合変換 T̂_I
    
    認識から統合的理解への変換
    """
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def operator(self, gamma: float = 0.2) -> np.ndarray:
        """
        統合変換演算子
        
        情報の統合を模倣
        """
        # 固定点への収束を促す演算子
        T = np.eye(self.dim, dtype=complex)
        
        # 重心への収縮
        center = np.ones((self.dim, self.dim), dtype=complex) / self.dim
        T = (1 - gamma) * T + gamma * center
        
        return T
    
    def apply(self, state: np.ndarray, gamma: float = 0.2) -> np.ndarray:
        """変換を適用"""
        T = self.operator(gamma)
        result = T @ state
        return result / np.linalg.norm(result)


# =============================================================================
# 世界構築演算子 T̂_World
# =============================================================================

class WorldOperator:
    """
    世界構築演算子 T̂_World
    
    T̂_World = T̂_I ∘ T̂_R ∘ T̂_C ∘ Û_multi ∘ Û_res
    
    完全な世界生成プロセスを実行
    """
    
    def __init__(self, dim: int = 8):
        """
        Args:
            dim: 状態空間の次元
        """
        self.dim = dim
        
        # 各変換の初期化
        self.T_C = CognitionTransform(dim)
        self.T_R = RecognitionTransform(dim)
        self.T_I = IntegrationTransform(dim)
        
        # 時間発展演算子のパラメータ
        self.tau = 0.5
        self.epsilon = 0.3
        self.PFH = 0.2
    
    def U_res(self, t: float = 1.0) -> np.ndarray:
        """拡張時間発展演算子"""
        # 簡略化されたハミルトニアン
        H = np.diag(np.arange(self.dim, dtype=complex))
        H = H + self.tau * self._off_diagonal(self.dim, 0.5)
        H = H + self.epsilon * self._antisymmetric(self.dim, 0.3)
        H = H + self.PFH * np.eye(self.dim, dtype=complex) * 0.2
        
        return expm(-1j * H * t / HBAR)
    
    def U_multi(self, t: float = 1.0) -> np.ndarray:
        """多次元時間発展演算子"""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        
        # 複数の時間軸
        H += (1 + self.tau) * np.diag(np.arange(self.dim, dtype=float))
        H += self.tau * np.exp(-self.epsilon**2) * self._off_diagonal(self.dim, 0.3)
        H += self.PFH * np.sqrt(self.tau) * self._off_diagonal(self.dim, 0.2)
        
        return expm(-1j * H * t / HBAR)
    
    def _off_diagonal(self, dim: int, strength: float) -> np.ndarray:
        """非対角成分を生成"""
        M = np.zeros((dim, dim), dtype=complex)
        for i in range(dim - 1):
            M[i, i+1] = strength
            M[i+1, i] = strength
        return M
    
    def _antisymmetric(self, dim: int, strength: float) -> np.ndarray:
        """反対称成分を生成"""
        M = np.zeros((dim, dim), dtype=complex)
        for i in range(dim - 1):
            M[i, i+1] = 1j * strength
            M[i+1, i] = -1j * strength
        return M
    
    def operator(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        t: float = 1.0
    ) -> np.ndarray:
        """
        完全な世界構築演算子
        
        T̂_World = T̂_I ∘ T̂_R ∘ T̂_C ∘ Û_multi ∘ Û_res
        """
        # 各演算子
        U_r = self.U_res(t)
        U_m = self.U_multi(t)
        T_c = self.T_C.operator(alpha)
        T_r = self.T_R.operator(beta)
        T_i = self.T_I.operator(gamma)
        
        # 合成
        T_world = T_i @ T_r @ T_c @ U_m @ U_r
        
        return T_world
    
    def apply(
        self,
        state: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        t: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        世界構築を実行
        
        Returns:
            (output_state, info)
        """
        T_world = self.operator(alpha, beta, gamma, t)
        output = T_world @ state
        output = output / np.linalg.norm(output)
        
        # 情報
        info = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "t": t,
            "operator_norm": np.linalg.norm(T_world, ord=2)
        }
        
        return output, info
    
    def set_resonance_params(self, tau: float, epsilon: float, PFH: float):
        """共鳴パラメータを設定"""
        self.tau = tau
        self.epsilon = epsilon
        self.PFH = PFH


# =============================================================================
# 観測量
# =============================================================================

def meaning_observable(dim: int) -> np.ndarray:
    """
    意味観測量 O_M
    
    KLダイバージェンスに対応
    """
    O_M = np.diag(np.arange(dim, dtype=float))
    return O_M


def question_observable(dim: int) -> np.ndarray:
    """
    質問観測量 O_Q
    
    予測誤差に対応
    """
    O_Q = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        O_Q[i, i+1] = 1
        O_Q[i+1, i] = 1
    return O_Q


def compute_expectation(state: np.ndarray, observable: np.ndarray) -> float:
    """期待値を計算"""
    rho = np.outer(state, state.conj())
    return np.real(np.trace(rho @ observable))


# =============================================================================
# デモ
# =============================================================================

def demo():
    """世界演算子のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第5章")
    print("世界構築演算子 T̂_World")
    print("=" * 60)
    
    # 1. 世界テンソル系
    print("\n[1] 世界生成テンソル体系")
    tensor_sys = WorldTensorSystem(dim_M=2, dim_C=2, dim_E=2, dim_F=2, dim_S=2)
    print(f"    総次元: {tensor_sys.total_dim}")
    print(f"    サブシステム次元: {tensor_sys.dims}")
    
    H_total = tensor_sys.total_hamiltonian()
    eigenvalues = np.linalg.eigvalsh(H_total)
    print(f"    H_sys の固有値範囲: [{eigenvalues.min():.2f}, {eigenvalues.max():.2f}]")
    
    # 2. 変換演算子
    print("\n[2] 変換演算子 T̂_C, T̂_R, T̂_I")
    dim = 8
    
    T_C = CognitionTransform(dim)
    T_R = RecognitionTransform(dim)
    T_I = IntegrationTransform(dim)
    
    print(f"    T̂_C ノルム: {np.linalg.norm(T_C.operator(), ord=2):.3f}")
    print(f"    T̂_R ノルム: {np.linalg.norm(T_R.operator(), ord=2):.3f}")
    print(f"    T̂_I ノルム: {np.linalg.norm(T_I.operator(), ord=2):.3f}")
    
    # 3. 世界構築演算子
    print("\n[3] 世界構築演算子 T̂_World")
    
    world_op = WorldOperator(dim=8)
    world_op.set_resonance_params(tau=0.5, epsilon=0.3, PFH=0.2)
    
    T_world = world_op.operator()
    print(f"    T̂_World の次元: {T_world.shape}")
    print(f"    T̂_World のノルム: {np.linalg.norm(T_world, ord=2):.3f}")
    
    # 固有値
    eigenvalues = np.linalg.eigvals(T_world)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"    最大固有値の絶対値: {max_eigenvalue:.3f}")
    
    # 4. 状態の変換
    print("\n[4] 状態の変換")
    
    # 初期状態
    psi_0 = np.zeros(8, dtype=complex)
    psi_0[0] = 1.0
    
    # 変換
    psi_final, info = world_op.apply(psi_0)
    
    print(f"    初期状態: |0⟩")
    print(f"    最終状態の確率分布:")
    for i in range(min(4, dim)):
        prob = np.abs(psi_final[i])**2
        print(f"        P(|{i}⟩) = {prob:.3f}")
    
    # 5. 観測量
    print("\n[5] 観測量")
    
    O_M = meaning_observable(dim)
    O_Q = question_observable(dim)
    
    exp_M = compute_expectation(psi_final, O_M)
    exp_Q = compute_expectation(psi_final, O_Q)
    
    print(f"    ⟨O_M⟩ (意味): {exp_M:.3f}")
    print(f"    ⟨O_Q⟩ (質問): {exp_Q:.3f}")
    
    # 6. パラメータ依存性
    print("\n[6] パラメータ依存性")
    
    for tau in [0.0, 0.5, 1.0]:
        world_op.set_resonance_params(tau=tau, epsilon=0.3, PFH=0.2)
        psi, _ = world_op.apply(psi_0)
        exp_M = compute_expectation(psi, O_M)
        print(f"    τ={tau:.1f}: ⟨O_M⟩ = {exp_M:.3f}")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
