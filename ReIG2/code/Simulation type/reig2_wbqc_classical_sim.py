"""
ReIG2/twinRIG: World-Building Quantum Channels
Classical Simulation Module - NumPy/SciPyベース古典シミュレーション

論文セクション対応: §3, §4, §5

このモジュールは以下を提供します：
- 共鳴時間発展演算子 Eres
- 多次元時間発展演算子 Emulti
- 相転移生成器 G = P ∘ E ∘ R
- 世界構築チャネル W
- 自己参照的不動点の計算
- 疎行列版（v2）大次元対応

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Complete
"""

import numpy as np
from scipy.linalg import expm, sqrtm, logm
from scipy.optimize import minimize, fixed_point
from scipy.integrate import odeint
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import warnings

# Sparse matrix support for large dimensions
try:
    from scipy.sparse import csr_matrix, issparse, diags
    from scipy.sparse.linalg import expm_multiply, eigsh
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    warnings.warn("scipy.sparse not fully available. Large dimension support limited.")

from reig2_wbqc_core import (
    WorldState, Hamiltonian, HamiltonianSystem, ResonanceParameters,
    CPTPMap, ThresholdConfig, CostWeights, HBAR,
    PAULI_X, PAULI_Y, PAULI_Z, PAULI_I, _matrix_exp
)


# ==============================================================================
# 共鳴時間発展演算子 Eres（§3）
# ==============================================================================

class ResonantTimeEvolution(CPTPMap):
    """
    共鳴時間発展演算子 Eres
    
    論文 §3 より：
    τパラメータは未来への寄与度（前方向の時間的延長）を制御する。
    
    修正された時間発展:
    Eres(ρ) = U_eff(t) ρ U_eff(t)†
    
    ここで U_eff は τ による修正を受けた実効ユニタリ
    """
    
    def __init__(self, 
                 hamiltonian: Hamiltonian,
                 params: ResonanceParameters,
                 time: float = 1.0):
        """
        Args:
            hamiltonian: 基底ハミルトニアン
            params: 共鳴パラメータ θ = (τ, ε, PFH)
            time: 発展時間
        """
        self.hamiltonian = hamiltonian
        self.params = params
        self.time = time
        self.dim = hamiltonian.dim
        
        # 実効ユニタリを計算
        self._compute_effective_unitary()
    
    def _compute_effective_unitary(self):
        """
        実効ユニタリ演算子を計算
        
        τによる時間スケーリング：
        - τ → 0: 即時応答（現在重視）
        - τ → ∞: 長期計画（未来重視）
        """
        # τによる実効時間スケール
        tau_scale = 1 + self.params.tau
        effective_time = self.time * tau_scale
        
        # 実効ハミルトニアン（εによるエントロピー的揺らぎを含む）
        H_eff = self._get_effective_hamiltonian()
        
        # ユニタリ演算子
        self._unitary = expm(-1j * H_eff * effective_time / HBAR)
    
    def _get_effective_hamiltonian(self) -> np.ndarray:
        """εを考慮した実効ハミルトニアンを計算"""
        H = self.hamiltonian.matrix.copy()
        
        # εによる確率的揺らぎ（ランダム摂動）
        if self.params.epsilon > 0:
            eps_scale = self.params.epsilon / (1 + self.params.epsilon)
            # ランダムエルミート行列を生成
            random_H = np.random.randn(self.dim, self.dim) + \
                      1j * np.random.randn(self.dim, self.dim)
            random_H = (random_H + random_H.conj().T) / 2
            random_H = random_H / np.linalg.norm(random_H) * eps_scale
            H = H + random_H * np.linalg.norm(H)
        
        return H
    
    def apply(self, state: WorldState) -> WorldState:
        """共鳴時間発展を適用"""
        rho_new = self._unitary @ state.rho @ self._unitary.conj().T
        return WorldState(rho_new)
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """Kraus演算子を取得（ユニタリなので単一）"""
        return [self._unitary]


# ==============================================================================
# 多次元時間発展演算子 Emulti（§3）
# ==============================================================================

class MultidimensionalTimeEvolution(CPTPMap):
    """
    多次元時間発展演算子 Emulti
    
    論文 §3.1 幾何学的解釈より：
    多次元時間の構造は以下の幾何学的解釈を持つ：
    - τ軸：未来への寄与度
    - ε軸：エントロピー的揺らぎの許容度
    - PFH軸：哲学的・倫理的重みづけ
    """
    
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 params: ResonanceParameters,
                 time: float = 1.0,
                 n_dimensions: int = 3):
        """
        Args:
            hamiltonian: 基底ハミルトニアン
            params: 共鳴パラメータ
            time: 発展時間
            n_dimensions: 時間次元数
        """
        self.hamiltonian = hamiltonian
        self.params = params
        self.time = time
        self.n_dimensions = n_dimensions
        self.dim = hamiltonian.dim
        
        # 各次元の発展演算子を計算
        self._compute_evolution_operators()
    
    def _compute_evolution_operators(self):
        """各時間次元の発展演算子を計算"""
        H = self.hamiltonian.matrix
        
        # 次元1: 通常時間発展（τ修正）
        tau_scale = 1 / (1 + self.params.tau)
        self._U_tau = expm(-1j * H * self.time * tau_scale / HBAR)
        
        # 次元2: エントロピー発展（ε修正）
        # 散逸成分を含むLindblad的発展を近似
        self._lindblad_rate = self.params.epsilon / (1 + self.params.epsilon)
        
        # 次元3: PFH方向（倫理的収束）
        # 射影的収縮を表現
        self._pfh_damping = 1 / (1 + self.params.PFH)
    
    def apply(self, state: WorldState) -> WorldState:
        """多次元時間発展を適用"""
        rho = state.rho.copy()
        
        # 次元1: ユニタリ発展
        rho = self._U_tau @ rho @ self._U_tau.conj().T
        
        # 次元2: 散逸（デコヒーレンス）
        if self._lindblad_rate > 0:
            rho = self._apply_decoherence(rho)
        
        # 次元3: PFH収縮（倫理的フィルタリング）
        rho = self._apply_pfh_contraction(rho)
        
        return WorldState(rho)
    
    def _apply_decoherence(self, rho: np.ndarray) -> np.ndarray:
        """デコヒーレンス（位相減衰）を適用"""
        # 対角成分は保存、非対角成分を減衰
        decay = np.exp(-self._lindblad_rate * self.time)
        
        rho_new = rho.copy()
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    rho_new[i, j] *= decay
        
        return rho_new
    
    def _apply_pfh_contraction(self, rho: np.ndarray) -> np.ndarray:
        """PFH収縮（倫理的状態への収束）を適用"""
        # 最大混合状態への部分的収縮
        rho_mixed = np.eye(self.dim, dtype=complex) / self.dim
        return self._pfh_damping * rho + (1 - self._pfh_damping) * rho_mixed
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """Kraus演算子を取得"""
        # 完全なKraus分解は複雑なので近似
        return [self._U_tau]


# ==============================================================================
# 相転移生成器 G = P ∘ E ∘ R（§4）
# ==============================================================================

class PhaseTransitionGenerator(CPTPMap):
    """
    相転移生成器 G = P ∘ E ∘ R
    
    論文 §4 より：
    - P: 相転移演算子
    - E: 発展演算子
    - R: 再構成演算子
    """
    
    def __init__(self,
                 dim: int,
                 params: ResonanceParameters,
                 transition_strength: float = 0.5,
                 reconstruction_rate: float = 0.3):
        """
        Args:
            dim: 状態空間次元
            params: 共鳴パラメータ
            transition_strength: 相転移強度
            reconstruction_rate: 再構成率
        """
        self.dim = dim
        self.params = params
        self.transition_strength = transition_strength
        self.reconstruction_rate = reconstruction_rate
        
        # 各演算子を構築
        self._build_operators()
    
    def _build_operators(self):
        """P, E, R 演算子を構築"""
        # R: 再構成演算子（Kraus形式）
        self._R_kraus = self._build_reconstruction()
        
        # E: 発展演算子
        self._E_kraus = self._build_evolution()
        
        # P: 相転移演算子
        self._P_kraus = self._build_phase_transition()
    
    def _build_reconstruction(self) -> List[np.ndarray]:
        """再構成演算子 R を構築"""
        # 状態を混合化する演算子
        rate = self.reconstruction_rate
        K0 = np.sqrt(1 - rate) * np.eye(self.dim, dtype=complex)
        K1 = np.sqrt(rate / self.dim) * np.ones((self.dim, self.dim), dtype=complex)
        return [K0, K1]
    
    def _build_evolution(self) -> List[np.ndarray]:
        """発展演算子 E を構築"""
        # ランダムユニタリによる発展
        # Haar測度からのサンプリングを近似
        H_random = np.random.randn(self.dim, self.dim) + \
                  1j * np.random.randn(self.dim, self.dim)
        H_random = (H_random + H_random.conj().T) / 2
        U = expm(-1j * H_random * 0.1)
        return [U]
    
    def _build_phase_transition(self) -> List[np.ndarray]:
        """相転移演算子 P を構築"""
        # 臨界的な状態変化を表現
        strength = self.transition_strength
        
        # 単位行列と射影の組み合わせ
        # 基底状態への部分的射影
        proj_ground = np.zeros((self.dim, self.dim), dtype=complex)
        proj_ground[0, 0] = 1
        
        K0 = np.sqrt(1 - strength) * np.eye(self.dim, dtype=complex)
        K1 = np.sqrt(strength) * proj_ground
        
        return [K0, K1]
    
    def apply(self, state: WorldState) -> WorldState:
        """G = P ∘ E ∘ R を適用"""
        rho = state.rho.copy()
        
        # R を適用
        rho = self._apply_kraus(rho, self._R_kraus)
        
        # E を適用
        rho = self._apply_kraus(rho, self._E_kraus)
        
        # P を適用
        rho = self._apply_kraus(rho, self._P_kraus)
        
        return WorldState(rho)
    
    def _apply_kraus(self, rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
        """Kraus演算子を適用"""
        rho_new = np.zeros_like(rho)
        for K in kraus_ops:
            rho_new += K @ rho @ K.conj().T
        return rho_new
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """全体のKraus演算子を取得（近似）"""
        # 合成Kraus演算子は組み合わせ的に増加するため、主要項のみ返す
        return self._P_kraus


# ==============================================================================
# 世界構築チャネル W（§5）
# ==============================================================================

class WorldBuildingChannel(CPTPMap):
    """
    世界構築チャネル W
    
    論文 §5 より：
    世界構築チャネル W と自己参照的不動点 ρ* を定義する。
    
    定理 5.1 (収束定理): 
    適切な条件下で、世界状態は一意な定常状態 ρ* に収束する。
    """
    
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 params: ResonanceParameters,
                 E_res: Optional[ResonantTimeEvolution] = None,
                 E_multi: Optional[MultidimensionalTimeEvolution] = None,
                 G: Optional[PhaseTransitionGenerator] = None):
        """
        Args:
            hamiltonian: 基底ハミルトニアン
            params: 共鳴パラメータ
            E_res: 共鳴時間発展演算子
            E_multi: 多次元時間発展演算子
            G: 相転移生成器
        """
        self.hamiltonian = hamiltonian
        self.params = params
        self.dim = hamiltonian.dim
        
        # コンポーネント演算子
        self.E_res = E_res or ResonantTimeEvolution(hamiltonian, params)
        self.E_multi = E_multi or MultidimensionalTimeEvolution(hamiltonian, params)
        self.G = G or PhaseTransitionGenerator(self.dim, params)
        
        # 不動点キャッシュ
        self._fixed_point = None
    
    def apply(self, state: WorldState) -> WorldState:
        """
        世界構築チャネルを適用
        
        W = G ∘ Emulti ∘ Eres
        """
        # Eres
        state = self.E_res.apply(state)
        
        # Emulti
        state = self.E_multi.apply(state)
        
        # G
        state = self.G.apply(state)
        
        return state
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """Kraus演算子を取得（近似）"""
        return self.E_res.get_kraus_operators()
    
    def iterate(self, initial_state: WorldState, n_iterations: int) -> List[WorldState]:
        """
        チャネルを繰り返し適用
        
        ρn+1 = W(ρn)
        """
        states = [initial_state]
        current = initial_state
        
        for _ in range(n_iterations):
            current = self.apply(current)
            states.append(current)
        
        return states
    
    def find_fixed_point(self, 
                        initial_guess: Optional[WorldState] = None,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-8,
                        method: str = "iterative") -> WorldState:
        """
        自己参照的不動点 ρ* を求める
        
        W(ρ*) = ρ*
        
        Args:
            initial_guess: 初期推定状態
            max_iterations: 最大反復回数
            tolerance: 収束判定閾値
            method: "iterative" (反復法) or "optimize" (最適化法)
        
        Returns:
            不動点状態 ρ*
        """
        if initial_guess is None:
            # 最大混合状態から開始
            initial_guess = WorldState.maximally_mixed(self.dim)
        
        if method == "optimize":
            return self._find_fixed_point_optimize(initial_guess, max_iterations, tolerance)
        else:
            return self._find_fixed_point_iterative(initial_guess, max_iterations, tolerance)
    
    def _find_fixed_point_iterative(self,
                                    initial_guess: WorldState,
                                    max_iterations: int,
                                    tolerance: float) -> WorldState:
        """反復法による不動点探索（各ステップで正規化を強制）"""
        current = initial_guess
        
        for i in range(max_iterations):
            next_state = self.apply(current)
            
            # 正規化を各ステップで強制（数値安定性）
            rho_normalized = next_state.rho.copy()
            trace = np.trace(rho_normalized)
            if np.abs(trace) > 1e-10:
                rho_normalized = rho_normalized / trace
            # エルミート性を強制
            rho_normalized = (rho_normalized + rho_normalized.conj().T) / 2
            next_state = WorldState(rho_normalized, validate=False)
            
            # 収束判定
            distance = current.trace_distance(next_state)
            if distance < tolerance:
                print(f"Fixed point found after {i+1} iterations")
                self._fixed_point = next_state
                return next_state
            
            current = next_state
        
        warnings.warn(f"Did not converge after {max_iterations} iterations")
        self._fixed_point = current
        return current
    
    def _find_fixed_point_optimize(self,
                                   initial_guess: WorldState,
                                   max_iterations: int,
                                   tolerance: float) -> WorldState:
        """
        最適化による不動点探索（複素数対応）
        
        複素密度行列を実数ベクトルに変換（real/imag別ベクトル化）
        目的関数: ||W(ρ) - ρ||_F² を最小化
        """
        dim = self.dim
        
        def rho_to_real_vec(rho: np.ndarray) -> np.ndarray:
            """複素密度行列を実数ベクトルに変換（上三角部分のみ）"""
            real_parts = []
            imag_parts = []
            for i in range(dim):
                for j in range(i, dim):
                    real_parts.append(rho[i, j].real)
                    if i != j:
                        imag_parts.append(rho[i, j].imag)
            return np.concatenate([real_parts, imag_parts])
        
        def real_vec_to_rho(vec: np.ndarray) -> np.ndarray:
            """実数ベクトルを複素密度行列に復元"""
            rho = np.zeros((dim, dim), dtype=complex)
            n_diag = dim
            n_off_diag = dim * (dim - 1) // 2
            
            idx_real = 0
            idx_imag = n_diag + n_off_diag
            
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        rho[i, j] = vec[idx_real]
                    else:
                        rho[i, j] = vec[idx_real] + 1j * vec[idx_imag]
                        rho[j, i] = vec[idx_real] - 1j * vec[idx_imag]
                        idx_imag += 1
                    idx_real += 1
            
            return rho
        
        def normalize_rho(rho: np.ndarray) -> np.ndarray:
            """密度行列を正規化（トレース=1、エルミート、正半定値）"""
            # エルミート化
            rho = (rho + rho.conj().T) / 2
            # トレース正規化
            trace = np.trace(rho)
            if np.abs(trace) > 1e-10:
                rho = rho / trace
            return rho
        
        def objective(vec: np.ndarray) -> float:
            """目的関数: ||W(ρ) - ρ||_F² + penalties"""
            rho = real_vec_to_rho(vec)
            rho = normalize_rho(rho)
            
            # 正半定値性ペナルティ
            eigvals = np.linalg.eigvalsh(rho)
            penalty = np.sum(np.maximum(-eigvals, 0)) * 100
            
            # W(ρ)を計算
            state = WorldState(rho, validate=False)
            next_state = self.apply(state)
            
            # ||W(ρ) - ρ||_F²
            diff = next_state.rho - rho
            distance = np.real(np.sum(diff * diff.conj()))
            
            return distance + penalty
        
        # 初期値
        x0 = rho_to_real_vec(initial_guess.rho)
        
        # 最適化
        result = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        # 結果を密度行列に復元
        rho_opt = real_vec_to_rho(result.x)
        rho_opt = (rho_opt + rho_opt.conj().T) / 2
        trace = np.trace(rho_opt)
        if np.abs(trace) > 1e-10:
            rho_opt = rho_opt / trace
        
        # 正半定値性の強制
        eigvals, eigvecs = np.linalg.eigh(rho_opt)
        eigvals = np.maximum(eigvals, 0)
        if np.sum(eigvals) > 0:
            eigvals = eigvals / np.sum(eigvals)
        rho_opt = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        fixed_point = WorldState(rho_opt)
        
        if result.success:
            print(f"Fixed point found via optimization (iterations: {result.nit})")
        else:
            warnings.warn(f"Optimization did not fully converge: {result.message}")
        
        self._fixed_point = fixed_point
        return fixed_point
    
    def check_contraction(self, 
                         state1: WorldState, 
                         state2: WorldState) -> Tuple[float, bool]:
        """
        縮小写像性を確認
        
        D(W(ρ1), W(ρ2)) ≤ γ D(ρ1, ρ2) for some γ < 1
        """
        d_before = state1.trace_distance(state2)
        
        W_state1 = self.apply(state1)
        W_state2 = self.apply(state2)
        
        d_after = W_state1.trace_distance(W_state2)
        
        if d_before > 1e-10:
            gamma = d_after / d_before
        else:
            gamma = 0.0
        
        is_contraction = gamma < 1.0
        
        return gamma, is_contraction


# ==============================================================================
# 疎行列版 世界構築チャネル（v2 - 大次元対応）
# ==============================================================================

class SparseWorldBuildingChannel:
    """
    疎行列版 世界構築チャネル（大次元対応 v2）
    
    論文 §5 より：
    dim > 100 の場合にメモリ効率と計算速度を改善
    
    特徴:
    - 疎行列演算による効率的な時間発展
    - Krylov部分空間法による行列指数関数近似
    - 自動的な疎/密切り替え
    """
    
    def __init__(self,
                 hamiltonian_matrix: np.ndarray,
                 params: ResonanceParameters,
                 sparsity_threshold: float = 0.1):
        """
        Args:
            hamiltonian_matrix: ハミルトニアン行列
            params: 共鳴パラメータ
            sparsity_threshold: 疎行列化閾値（非零要素比がこの値以下で疎行列使用）
        """
        self.params = params
        self.dim = hamiltonian_matrix.shape[0]
        self.sparsity_threshold = sparsity_threshold
        
        if not SPARSE_AVAILABLE:
            warnings.warn("Sparse support not available. Using dense matrices.")
            self._use_sparse = False
            self.H = hamiltonian_matrix.astype(complex)
        else:
            # 疎行列性チェック
            nnz_ratio = np.count_nonzero(hamiltonian_matrix) / hamiltonian_matrix.size
            self._use_sparse = nnz_ratio < sparsity_threshold
            
            if self._use_sparse:
                self.H = csr_matrix(hamiltonian_matrix.astype(complex))
            else:
                self.H = hamiltonian_matrix.astype(complex)
        
        # パラメータに基づく係数
        self.tau_scale = 1 / (1 + params.tau)
        self.epsilon_scale = params.epsilon / (1 + params.epsilon)
        self.pfh_damping = 1 / (1 + params.PFH)
        
        self._fixed_point = None
    
    def apply(self, rho: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        世界構築チャネルを疎行列演算で適用
        
        Args:
            rho: 密度行列
            dt: 時間ステップ
        
        Returns:
            更新された密度行列
        """
        # τによる時間スケーリング
        effective_dt = dt * self.tau_scale
        
        if self._use_sparse and SPARSE_AVAILABLE:
            # 疎行列版の時間発展
            rho_new = self._sparse_evolution(rho, effective_dt)
        else:
            # 密行列版
            U = expm(-1j * self.H * effective_dt / HBAR)
            rho_new = U @ rho @ U.conj().T
        
        # εによるデコヒーレンス（位相減衰）
        if self.epsilon_scale > 0:
            decay = np.exp(-self.epsilon_scale * dt)
            diag = np.diag(np.diag(rho_new))
            off_diag = rho_new - diag
            rho_new = diag + decay * off_diag
        
        # PFH収縮（最大混合状態への部分的緩和）
        rho_mixed = np.eye(self.dim, dtype=complex) / self.dim
        rho_new = self.pfh_damping * rho_new + (1 - self.pfh_damping) * rho_mixed
        
        # 正規化を強制
        trace = np.trace(rho_new)
        if np.abs(trace) > 1e-10:
            rho_new = rho_new / trace
        
        # エルミート性を強制
        rho_new = (rho_new + rho_new.conj().T) / 2
        
        return rho_new
    
    def _sparse_evolution(self, rho: np.ndarray, dt: float) -> np.ndarray:
        """
        疎行列による時間発展
        
        Krylov部分空間法 (expm_multiply) を使用
        """
        if not SPARSE_AVAILABLE:
            U = expm(-1j * self.H * dt / HBAR)
            return U @ rho @ U.conj().T
        
        # 疎行列に変換
        H_sparse = self.H if issparse(self.H) else csr_matrix(self.H)
        
        # 密度行列の各列に対して exp(-iHt)|ρ_j⟩ を計算
        # これは U @ rho を列ごとに計算することに相当
        rho_new = np.zeros_like(rho)
        
        for j in range(self.dim):
            col_j = rho[:, j].copy()
            # expm_multiply: exp(A) @ v を効率的に計算
            evolved_col = expm_multiply(-1j * H_sparse * dt / HBAR, col_j)
            rho_new[:, j] = evolved_col
        
        # U @ rho @ U† = (U @ rho) @ U† を計算
        # 行に対しても同様の処理
        result = np.zeros_like(rho_new)
        for i in range(self.dim):
            row_i = rho_new[i, :].copy()
            # U† = conj(U.T) なので、conj(expm(-iH*t) @ v) = expm(iH*t) @ conj(v)
            evolved_row = expm_multiply(1j * H_sparse * dt / HBAR, row_i.conj())
            result[i, :] = evolved_row.conj()
        
        return result
    
    def iterate(self, 
                initial_rho: np.ndarray, 
                n_iterations: int,
                dt: float = 1.0) -> List[np.ndarray]:
        """チャネルを繰り返し適用"""
        states = [initial_rho.copy()]
        current = initial_rho.copy()
        
        for _ in range(n_iterations):
            current = self.apply(current, dt)
            states.append(current.copy())
        
        return states
    
    def find_fixed_point(self, 
                        initial_rho: Optional[np.ndarray] = None,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-8,
                        dt: float = 1.0) -> np.ndarray:
        """
        不動点を探索（各ステップで正規化強制）
        """
        if initial_rho is None:
            initial_rho = np.eye(self.dim, dtype=complex) / self.dim
        
        current = initial_rho.copy()
        
        for i in range(max_iterations):
            next_rho = self.apply(current, dt)
            
            # 収束判定（フロベニウスノルム）
            diff = next_rho - current
            distance = np.sqrt(np.real(np.trace(diff.conj().T @ diff)))
            
            if distance < tolerance:
                print(f"Sparse fixed point found after {i+1} iterations")
                self._fixed_point = next_rho
                return next_rho
            
            current = next_rho
        
        warnings.warn(f"Did not converge after {max_iterations} iterations")
        self._fixed_point = current
        return current
    
    @property
    def is_sparse(self) -> bool:
        """疎行列モードかどうか"""
        return self._use_sparse
    
    def memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量の推定"""
        dense_size = self.dim ** 2 * 16  # complex128 = 16 bytes
        if self._use_sparse and SPARSE_AVAILABLE and issparse(self.H):
            sparse_size = (self.H.data.nbytes + 
                          self.H.indices.nbytes + 
                          self.H.indptr.nbytes)
            return {
                'dense_equivalent_bytes': dense_size,
                'actual_bytes': sparse_size,
                'savings_ratio': 1 - sparse_size / dense_size,
                'mode': 'sparse'
            }
        return {
            'dense_equivalent_bytes': dense_size,
            'actual_bytes': dense_size,
            'savings_ratio': 0.0,
            'mode': 'dense'
        }


# ==============================================================================
# Lindblad方程式シミュレーション
# ==============================================================================

class LindbladEvolution:
    """
    Lindblad方程式による時間発展
    
    dρ/dt = -i[H, ρ]/ℏ + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    """
    
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 lindblad_operators: List[np.ndarray],
                 gamma: List[float]):
        """
        Args:
            hamiltonian: システムハミルトニアン
            lindblad_operators: Lindblad演算子のリスト
            gamma: 各演算子の係数
        """
        self.H = hamiltonian.matrix
        self.L_ops = lindblad_operators
        self.gamma = gamma
        self.dim = hamiltonian.dim
    
    def _lindblad_rhs(self, rho_flat: np.ndarray, t: float) -> np.ndarray:
        """Lindblad方程式の右辺"""
        rho = rho_flat.reshape(self.dim, self.dim)
        
        # ユニタリ部分: -i[H, ρ]/ℏ
        drho = -1j * (self.H @ rho - rho @ self.H) / HBAR
        
        # 散逸部分
        for L, g in zip(self.L_ops, self.gamma):
            LdL = L.conj().T @ L
            drho += g * (L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
        
        return drho.flatten()
    
    def evolve(self, 
               initial_state: WorldState, 
               times: np.ndarray) -> List[WorldState]:
        """時間発展を計算"""
        rho0 = initial_state.rho.flatten()
        
        # 実部と虚部に分離してODE積分
        rho0_real = np.concatenate([rho0.real, rho0.imag])
        
        def rhs_real(y, t):
            rho = y[:self.dim**2] + 1j * y[self.dim**2:]
            drho = self._lindblad_rhs(rho, t)
            return np.concatenate([drho.real, drho.imag])
        
        solution = odeint(rhs_real, rho0_real, times)
        
        states = []
        for sol in solution:
            rho = sol[:self.dim**2] + 1j * sol[self.dim**2:]
            rho = rho.reshape(self.dim, self.dim)
            states.append(WorldState(rho))
        
        return states


# ==============================================================================
# シミュレーションランナー
# ==============================================================================

class ClassicalSimulationRunner:
    """
    古典シミュレーションの統合ランナー
    """
    
    def __init__(self,
                 dim: int = 4,
                 params: Optional[ResonanceParameters] = None):
        """
        Args:
            dim: 状態空間次元
            params: 共鳴パラメータ
        """
        self.dim = dim
        self.params = params or ResonanceParameters()
        
        # ハミルトニアンを生成
        self._setup_hamiltonian()
        
        # 世界構築チャネル
        self.W = WorldBuildingChannel(self.hamiltonian, self.params)
    
    def _setup_hamiltonian(self):
        """ハミルトニアンを設定"""
        # ランダムエルミート行列を生成
        H = np.random.randn(self.dim, self.dim) + \
            1j * np.random.randn(self.dim, self.dim)
        H = (H + H.conj().T) / 2
        self.hamiltonian = Hamiltonian(H, "H_system")
    
    def run_world_evolution(self,
                           initial_state: Optional[WorldState] = None,
                           n_steps: int = 100) -> Dict[str, Any]:
        """
        世界発展シミュレーションを実行
        """
        if initial_state is None:
            # ランダム純粋状態から開始
            psi = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
            initial_state = WorldState.pure_state(psi)
        
        # 発展を計算
        states = self.W.iterate(initial_state, n_steps)
        
        # 物理量を計算
        purities = [s.purity() for s in states]
        entropies = [s.von_neumann_entropy() for s in states]
        
        # 不動点との距離（もし計算済みなら）
        if self.W._fixed_point is not None:
            distances = [s.trace_distance(self.W._fixed_point) for s in states]
        else:
            distances = None
        
        return {
            'states': states,
            'purities': np.array(purities),
            'entropies': np.array(entropies),
            'distances': np.array(distances) if distances else None
        }
    
    def find_fixed_point(self) -> WorldState:
        """不動点を求める"""
        return self.W.find_fixed_point()
    
    def analyze_contraction(self, n_pairs: int = 10) -> Dict[str, Any]:
        """縮小写像性を分析"""
        gammas = []
        
        for _ in range(n_pairs):
            # ランダムな状態ペアを生成
            psi1 = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
            psi2 = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
            
            state1 = WorldState.pure_state(psi1)
            state2 = WorldState.pure_state(psi2)
            
            gamma, _ = self.W.check_contraction(state1, state2)
            gammas.append(gamma)
        
        return {
            'gammas': np.array(gammas),
            'mean_gamma': np.mean(gammas),
            'max_gamma': np.max(gammas),
            'is_contraction': np.max(gammas) < 1.0
        }


# ==============================================================================
# 連続/離散ハイブリッドモデル（§4.1）
# ==============================================================================

class HybridEvolution:
    """
    連続時間発展と離散相転移のハイブリッドモデル
    
    論文 §4.1 より：
    連続時間発展と離散相転移を組み合わせたハイブリッドモデル
    """
    
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 params: ResonanceParameters,
                 transition_times: List[float],
                 transition_strength: float = 0.5):
        """
        Args:
            hamiltonian: システムハミルトニアン
            params: 共鳴パラメータ
            transition_times: 相転移が起こる時刻のリスト
            transition_strength: 相転移の強度
        """
        self.hamiltonian = hamiltonian
        self.params = params
        self.transition_times = sorted(transition_times)
        self.transition_strength = transition_strength
        self.dim = hamiltonian.dim
        
        # 相転移演算子
        self.G = PhaseTransitionGenerator(
            dim=self.dim,
            params=params,
            transition_strength=transition_strength
        )
    
    def evolve(self, 
               initial_state: WorldState,
               t_final: float,
               dt: float = 0.01) -> Tuple[np.ndarray, List[WorldState]]:
        """
        ハイブリッド時間発展を実行
        
        Returns:
            times: 時刻の配列
            states: 各時刻の状態
        """
        times = [0.0]
        states = [initial_state]
        
        current_state = initial_state
        current_time = 0.0
        
        # 相転移時刻のインデックス
        transition_idx = 0
        
        while current_time < t_final:
            # 次の時刻を決定
            if transition_idx < len(self.transition_times) and \
               self.transition_times[transition_idx] < current_time + dt:
                # 相転移時刻
                next_time = self.transition_times[transition_idx]
                is_transition = True
                transition_idx += 1
            else:
                next_time = min(current_time + dt, t_final)
                is_transition = False
            
            # 連続時間発展
            if next_time > current_time:
                delta_t = next_time - current_time
                U = expm(-1j * self.hamiltonian.matrix * delta_t / HBAR)
                rho_new = U @ current_state.rho @ U.conj().T
                current_state = WorldState(rho_new)
            
            # 離散相転移
            if is_transition:
                current_state = self.G.apply(current_state)
            
            current_time = next_time
            times.append(current_time)
            states.append(current_state)
        
        return np.array(times), states


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReIG2/twinRIG Classical Simulation Module Test")
    print("=" * 60)
    
    # パラメータ設定
    params = ResonanceParameters(tau=0.5, epsilon=0.1, PFH=1.0)
    dim = 4
    
    # ハミルトニアン生成
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2
    hamiltonian = Hamiltonian(H, "H_test")
    
    # 初期状態
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    initial_state = WorldState.pure_state(psi0)
    
    print(f"\nInitial state: purity = {initial_state.purity():.4f}")
    
    # [1] 共鳴時間発展テスト
    print("\n[1] Resonant Time Evolution Test:")
    E_res = ResonantTimeEvolution(hamiltonian, params, time=1.0)
    evolved = E_res.apply(initial_state)
    print(f"  After Eres: purity = {evolved.purity():.4f}")
    
    # [2] 多次元時間発展テスト
    print("\n[2] Multidimensional Time Evolution Test:")
    E_multi = MultidimensionalTimeEvolution(hamiltonian, params)
    evolved = E_multi.apply(initial_state)
    print(f"  After Emulti: purity = {evolved.purity():.4f}")
    
    # [3] 相転移生成器テスト
    print("\n[3] Phase Transition Generator Test:")
    G = PhaseTransitionGenerator(dim, params)
    evolved = G.apply(initial_state)
    print(f"  After G: purity = {evolved.purity():.4f}")
    
    # [4] 世界構築チャネルテスト
    print("\n[4] World Building Channel Test:")
    W = WorldBuildingChannel(hamiltonian, params)
    
    # 反復適用
    states = W.iterate(initial_state, n_iterations=20)
    print(f"  Iterations: {len(states)}")
    print(f"  Initial purity: {states[0].purity():.4f}")
    print(f"  Final purity: {states[-1].purity():.4f}")
    
    # [5] 不動点探索テスト
    print("\n[5] Fixed Point Search Test:")
    fixed_point = W.find_fixed_point(max_iterations=100)
    print(f"  Fixed point purity: {fixed_point.purity():.4f}")
    print(f"  Fixed point entropy: {fixed_point.von_neumann_entropy():.4f}")
    
    # 不動点の検証
    W_fixed = W.apply(fixed_point)
    distance = fixed_point.trace_distance(W_fixed)
    print(f"  Distance W(ρ*) to ρ*: {distance:.2e}")
    
    # [6] 縮小写像性テスト
    print("\n[6] Contraction Property Test:")
    gamma, is_contraction = W.check_contraction(initial_state, WorldState.maximally_mixed(dim))
    print(f"  Contraction factor γ: {gamma:.4f}")
    print(f"  Is contraction: {is_contraction}")
    
    # [7] シミュレーションランナーテスト
    print("\n[7] Simulation Runner Test:")
    runner = ClassicalSimulationRunner(dim=4, params=params)
    results = runner.run_world_evolution(n_steps=50)
    
    print(f"  Purity evolution: {results['purities'][0]:.4f} → {results['purities'][-1]:.4f}")
    print(f"  Entropy evolution: {results['entropies'][0]:.4f} → {results['entropies'][-1]:.4f}")
    
    # [8] ハイブリッド発展テスト
    print("\n[8] Hybrid Evolution Test:")
    hybrid = HybridEvolution(
        hamiltonian=hamiltonian,
        params=params,
        transition_times=[1.0, 3.0, 5.0],
        transition_strength=0.3
    )
    
    times, hybrid_states = hybrid.evolve(initial_state, t_final=6.0, dt=0.1)
    print(f"  Time points: {len(times)}")
    print(f"  Final purity: {hybrid_states[-1].purity():.4f}")
    
    print("\n" + "=" * 60)
    print("All classical simulation tests completed!")
    print("=" * 60)
