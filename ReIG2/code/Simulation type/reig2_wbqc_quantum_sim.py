"""
ReIG2/twinRIG: World-Building Quantum Channels
Quantum Simulation Module - Qiskitベース量子シミュレーション

論文セクション対応: §2.2.1, §4.1.1

このモジュールは以下を提供します：
- Qiskitを用いた量子回路シミュレーション
- 2×2システムの量子実装
- Trotter分解による時間発展
- CPTP写像の量子回路表現

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Complete
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, DensityMatrix, Operator
    from qiskit.quantum_info import partial_trace, state_fidelity
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import UnitaryGate
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Placeholder types for type hints when Qiskit is unavailable
    QuantumCircuit = None
    QuantumRegister = None
    ClassicalRegister = None
    UnitaryGate = None
    warnings.warn("Qiskit not available. Using numpy fallback.")

from reig2_wbqc_core import (
    WorldState, Hamiltonian, HamiltonianSystem, ResonanceParameters,
    CPTPMap, HBAR, PAULI_X, PAULI_Y, PAULI_Z, PAULI_I,
    KET_0, KET_1, _matrix_exp
)


# ==============================================================================
# 量子回路ベースの世界状態
# ==============================================================================

class QuantumWorldState:
    """
    量子回路で表現される世界状態
    
    Qiskitの密度行列シミュレーションを使用
    """
    
    def __init__(self, n_qubits: int = 2, initial_state: Optional[np.ndarray] = None):
        """
        Args:
            n_qubits: 量子ビット数
            initial_state: 初期状態ベクトルまたは密度行列
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
        if not QISKIT_AVAILABLE:
            # フォールバック：numpy実装
            if initial_state is None:
                self._rho = np.eye(self.dim, dtype=complex) / self.dim
            elif initial_state.ndim == 1:
                psi = initial_state / np.linalg.norm(initial_state)
                self._rho = np.outer(psi, psi.conj())
            else:
                self._rho = initial_state
            self._density_matrix = None
        else:
            if initial_state is None:
                # 最大混合状態で初期化
                self._density_matrix = DensityMatrix(np.eye(self.dim) / self.dim)
            elif initial_state.ndim == 1:
                # 状態ベクトルから
                self._density_matrix = DensityMatrix(Statevector(initial_state))
            else:
                # 密度行列から
                self._density_matrix = DensityMatrix(initial_state)
            self._rho = self._density_matrix.data
    
    @property
    def rho(self) -> np.ndarray:
        """密度行列を取得"""
        if QISKIT_AVAILABLE and self._density_matrix is not None:
            return np.array(self._density_matrix.data)
        return self._rho.copy()
    
    def to_world_state(self) -> WorldState:
        """通常のWorldStateに変換"""
        return WorldState(self.rho)
    
    @classmethod
    def from_world_state(cls, state: WorldState) -> 'QuantumWorldState':
        """WorldStateから変換"""
        n_qubits = int(np.log2(state.dim))
        return cls(n_qubits=n_qubits, initial_state=state.rho)
    
    def purity(self) -> float:
        """純度を計算"""
        if QISKIT_AVAILABLE and self._density_matrix is not None:
            return float(self._density_matrix.purity())
        return np.real(np.trace(self._rho @ self._rho))
    
    def fidelity(self, other: 'QuantumWorldState') -> float:
        """忠実度を計算"""
        if QISKIT_AVAILABLE:
            return state_fidelity(self._density_matrix, other._density_matrix)
        return self.to_world_state().fidelity(other.to_world_state())
    
    def partial_trace_subsystem(self, keep_qubits: List[int]) -> 'QuantumWorldState':
        """指定した量子ビットを残して部分トレース"""
        if QISKIT_AVAILABLE:
            trace_out = [i for i in range(self.n_qubits) if i not in keep_qubits]
            reduced = partial_trace(self._density_matrix, trace_out)
            return QuantumWorldState(
                n_qubits=len(keep_qubits),
                initial_state=reduced.data
            )
        else:
            # numpy フォールバック
            raise NotImplementedError("Partial trace requires Qiskit")
    
    def apply_operator(self, operator: np.ndarray) -> 'QuantumWorldState':
        """演算子を適用"""
        rho_new = operator @ self.rho @ operator.conj().T
        return QuantumWorldState(n_qubits=self.n_qubits, initial_state=rho_new)
    
    def __repr__(self) -> str:
        return f"QuantumWorldState(n_qubits={self.n_qubits}, purity={self.purity():.4f})"


# ==============================================================================
# Trotter分解による時間発展（§4.1.1）
# ==============================================================================

@dataclass
class TrotterConfig:
    """
    Trotter分解の設定
    
    論文 §4.1.1 より：
    1次Trotter分解: exp(-i(HA+HB)Δt/ℏ) ≈ exp(-iHAΔt/ℏ)exp(-iHBΔt/ℏ) + O(Δt²)
    2次対称Trotter分解: 誤差はO(Δt³)に改善
    """
    order: int = 2           # Trotterの次数（1 or 2）
    n_steps: int = 100       # ステップ数
    delta_t: float = 0.01    # 時間刻み（自然単位）


class TrotterEvolution:
    """
    Trotter-鈴木分解による時間発展シミュレーション
    
    論文 §4.1.1 より：
    【誤差上界】
    ||U_exact - U_Trotter|| ≤ (Δt²/2ℏ²)||[HA, HB]||
    """
    
    def __init__(self, 
                 H_A: np.ndarray, 
                 H_B: np.ndarray, 
                 config: Optional[TrotterConfig] = None):
        """
        Args:
            H_A: 第一ハミルトニアン
            H_B: 第二ハミルトニアン
            config: Trotter分解設定
        """
        self.H_A = np.array(H_A, dtype=complex)
        self.H_B = np.array(H_B, dtype=complex)
        self.config = config or TrotterConfig()
        
        # 交換子の計算（誤差評価用）
        self.commutator = self.H_A @ self.H_B - self.H_B @ self.H_A
        self.commutator_norm = np.max(np.abs(np.linalg.eigvalsh(
            (self.commutator + self.commutator.conj().T) / 2
        )))
    
    def first_order_step(self, dt: float) -> np.ndarray:
        """
        1次Trotter分解による1ステップ
        
        U1(Δt) = exp(-iHA·Δt/ℏ) × exp(-iHB·Δt/ℏ)
        """
        U_A = _matrix_exp(-1j * self.H_A * dt / HBAR)
        U_B = _matrix_exp(-1j * self.H_B * dt / HBAR)
        return U_A @ U_B
    
    def second_order_step(self, dt: float) -> np.ndarray:
        """
        2次対称Trotter分解による1ステップ
        
        U2(Δt) = exp(-iHA·Δt/2ℏ) × exp(-iHB·Δt/ℏ) × exp(-iHA·Δt/2ℏ)
        """
        U_A_half = _matrix_exp(-1j * self.H_A * dt / (2 * HBAR))
        U_B = _matrix_exp(-1j * self.H_B * dt / HBAR)
        return U_A_half @ U_B @ U_A_half
    
    def exact_evolution(self, total_time: float) -> np.ndarray:
        """厳密な時間発展演算子"""
        H_total = self.H_A + self.H_B
        return _matrix_exp(-1j * H_total * total_time / HBAR)
    
    def trotter_evolution(self, total_time: float) -> np.ndarray:
        """Trotter分解による時間発展演算子"""
        dt = total_time / self.config.n_steps
        
        if self.config.order == 1:
            U_step = self.first_order_step(dt)
        else:
            U_step = self.second_order_step(dt)
        
        U_total = np.eye(self.H_A.shape[0], dtype=complex)
        for _ in range(self.config.n_steps):
            U_total = U_step @ U_total
        
        return U_total
    
    def error_bound(self, total_time: float) -> float:
        """
        理論的誤差上界を計算
        
        1次: O(Δt²)
        2次: O(Δt³)
        """
        dt = total_time / self.config.n_steps
        
        if self.config.order == 1:
            # ||U_exact - U_Trotter|| ≤ (Δt²/2ℏ²)||[HA, HB]|| × n_steps
            single_step_error = (dt ** 2 / (2 * HBAR ** 2)) * self.commutator_norm
        else:
            # 2次の場合はO(Δt³)
            single_step_error = (dt ** 3 / (12 * HBAR ** 3)) * self.commutator_norm ** 2
        
        return single_step_error * self.config.n_steps
    
    def actual_error(self, total_time: float) -> float:
        """実際の誤差を計算"""
        U_exact = self.exact_evolution(total_time)
        U_trotter = self.trotter_evolution(total_time)
        diff = U_exact - U_trotter
        return np.max(np.abs(np.linalg.eigvalsh(diff.conj().T @ diff))) ** 0.5
    
    def evolve_state(self, state: QuantumWorldState, total_time: float) -> QuantumWorldState:
        """状態を時間発展させる"""
        U = self.trotter_evolution(total_time)
        return state.apply_operator(U)


# ==============================================================================
# 量子回路による実装
# ==============================================================================

class QuantumCircuitBuilder:
    """
    ReIG2/twinRIG用の量子回路を構築
    """
    
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        
        if not QISKIT_AVAILABLE:
            warnings.warn("Qiskit not available, circuit building limited")
    
    def build_unitary_evolution(self, 
                                hamiltonian: np.ndarray, 
                                time: float,
                                label: str = "U(t)") -> Optional[Any]:
        """
        ユニタリ時間発展回路を構築
        
        U(t) = exp(-iHt/ℏ)
        """
        if not QISKIT_AVAILABLE:
            return None
        
        U = _matrix_exp(-1j * hamiltonian * time / HBAR)
        
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr, name=label)
        
        # ユニタリゲートとして追加
        gate = UnitaryGate(U, label=label)
        circuit.append(gate, qr)
        
        return circuit
    
    def build_trotter_circuit(self,
                             H_A: np.ndarray,
                             H_B: np.ndarray,
                             total_time: float,
                             n_steps: int = 10,
                             order: int = 2) -> Optional[Any]:
        """
        Trotter分解回路を構築
        """
        if not QISKIT_AVAILABLE:
            return None
        
        dt = total_time / n_steps
        
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr, name=f"Trotter(t={total_time})")
        
        U_A = _matrix_exp(-1j * H_A * dt / HBAR)
        U_B = _matrix_exp(-1j * H_B * dt / HBAR)
        
        if order == 2:
            U_A_half = _matrix_exp(-1j * H_A * dt / (2 * HBAR))
        
        for step in range(n_steps):
            if order == 1:
                circuit.append(UnitaryGate(U_A, label=f"U_A_{step}"), qr)
                circuit.append(UnitaryGate(U_B, label=f"U_B_{step}"), qr)
            else:
                circuit.append(UnitaryGate(U_A_half, label=f"U_A/2_{step}"), qr)
                circuit.append(UnitaryGate(U_B, label=f"U_B_{step}"), qr)
                circuit.append(UnitaryGate(U_A_half, label=f"U_A/2_{step}'"), qr)
        
        return circuit
    
    def build_bell_state_circuit(self) -> Optional[Any]:
        """ベル状態準備回路"""
        if not QISKIT_AVAILABLE:
            return None
        
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr, name="Bell|Φ+⟩")
        
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        
        return circuit
    
    def build_measurement_circuit(self, 
                                 base_circuit: Any,
                                 measure_qubits: Optional[List[int]] = None) -> Any:
        """測定を追加した回路を構築"""
        if not QISKIT_AVAILABLE:
            return None
        
        if measure_qubits is None:
            measure_qubits = list(range(self.n_qubits))
        
        cr = ClassicalRegister(len(measure_qubits), 'c')
        circuit = base_circuit.copy()
        circuit.add_register(cr)
        
        for i, q in enumerate(measure_qubits):
            circuit.measure(q, cr[i])
        
        return circuit


# ==============================================================================
# 2×2 最小システムの量子実装（§2.2.1）
# ==============================================================================

class Minimal2x2QuantumSystem:
    """
    最小の2×2量子システム実装
    
    論文 §2.2.1 より：
    - Hself = ℏωs|1⟩⟨1|
    - Hother = ℏωo|1⟩⟨1|
    - Hint = g(σx ⊗ σx)
    """
    
    def __init__(self, omega_s: float = 1.0, omega_o: float = 1.0, g: float = 0.5):
        """
        Args:
            omega_s: 自己状態の特性周波数
            omega_o: 他者状態の特性周波数
            g: 相互作用強度
        """
        self.omega_s = omega_s
        self.omega_o = omega_o
        self.g = g
        
        # ハミルトニアン構成
        self.H_self = HBAR * omega_s * np.array([[0, 0], [0, 1]], dtype=complex)
        self.H_other = HBAR * omega_o * np.array([[0, 0], [0, 1]], dtype=complex)
        self.H_int = g * np.kron(PAULI_X, PAULI_X)
        
        # 全ハミルトニアン（4×4）
        I2 = np.eye(2, dtype=complex)
        self.H_total = (np.kron(self.H_self, I2) + 
                       np.kron(I2, self.H_other) + 
                       self.H_int)
        
        # 量子回路ビルダー
        self.circuit_builder = QuantumCircuitBuilder(n_qubits=2)
    
    def create_initial_state(self, state_type: str = "ground") -> QuantumWorldState:
        """
        初期状態を生成
        
        Args:
            state_type: "ground", "excited", "superposition", "bell", "thermal"
        """
        if state_type == "ground":
            psi = np.array([1, 0, 0, 0], dtype=complex)
        elif state_type == "excited":
            psi = np.array([0, 0, 0, 1], dtype=complex)
        elif state_type == "superposition":
            psi = np.array([1, 1, 1, 1], dtype=complex) / 2
        elif state_type == "bell":
            psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        elif state_type == "thermal":
            # 熱平衡状態（β=1）
            beta = 1.0
            rho = _matrix_exp(-beta * self.H_total)
            rho = rho / np.trace(rho)
            return QuantumWorldState(n_qubits=2, initial_state=rho)
        else:
            raise ValueError(f"Unknown state type: {state_type}")
        
        return QuantumWorldState(n_qubits=2, initial_state=psi)
    
    def evolve(self, 
               initial_state: QuantumWorldState, 
               total_time: float,
               method: str = "exact") -> QuantumWorldState:
        """
        時間発展を実行
        
        Args:
            initial_state: 初期状態
            total_time: 発展時間
            method: "exact" or "trotter"
        """
        if method == "exact":
            U = _matrix_exp(-1j * self.H_total * total_time / HBAR)
            return initial_state.apply_operator(U)
        
        elif method == "trotter":
            # 非相互作用部分と相互作用部分に分割
            I2 = np.eye(2, dtype=complex)
            H_free = np.kron(self.H_self, I2) + np.kron(I2, self.H_other)
            
            trotter = TrotterEvolution(
                H_A=H_free,
                H_B=self.H_int,
                config=TrotterConfig(order=2, n_steps=100, delta_t=total_time/100)
            )
            return trotter.evolve_state(initial_state, total_time)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_observables(self, state: QuantumWorldState) -> Dict[str, float]:
        """
        各種物理量を計算
        """
        rho = state.rho
        
        # エネルギー期待値
        energy = np.real(np.trace(rho @ self.H_total))
        
        # 各サブシステムの局所観測量
        # σz期待値（自己系）
        sigma_z_self = np.kron(PAULI_Z, np.eye(2))
        z_self = np.real(np.trace(rho @ sigma_z_self))
        
        # σz期待値（他者系）
        sigma_z_other = np.kron(np.eye(2), PAULI_Z)
        z_other = np.real(np.trace(rho @ sigma_z_other))
        
        # エンタングルメント（線形エントロピーで近似）
        # 部分トレースで自己系の縮約密度行列を取得
        rho_self = _partial_trace_2x2(rho, keep='first')
        entanglement = 1 - np.real(np.trace(rho_self @ rho_self))
        
        return {
            'energy': energy,
            'z_self': z_self,
            'z_other': z_other,
            'entanglement': entanglement,
            'purity': state.purity()
        }
    
    def run_simulation(self, 
                      initial_state: str = "ground",
                      total_time: float = 10.0,
                      n_time_points: int = 100) -> Dict[str, np.ndarray]:
        """
        シミュレーションを実行して時間発展を追跡
        """
        state = self.create_initial_state(initial_state)
        dt = total_time / n_time_points
        
        times = np.linspace(0, total_time, n_time_points + 1)
        results = {
            'times': times,
            'energy': [],
            'z_self': [],
            'z_other': [],
            'entanglement': [],
            'purity': []
        }
        
        # 初期状態の観測量
        obs = self.compute_observables(state)
        for key in ['energy', 'z_self', 'z_other', 'entanglement', 'purity']:
            results[key].append(obs[key])
        
        # 時間発展
        current_state = state
        for i in range(n_time_points):
            current_state = self.evolve(current_state, dt, method="exact")
            obs = self.compute_observables(current_state)
            for key in ['energy', 'z_self', 'z_other', 'entanglement', 'purity']:
                results[key].append(obs[key])
        
        # numpy配列に変換
        for key in results:
            results[key] = np.array(results[key])
        
        return results


# ==============================================================================
# 共鳴時間発展演算子（§3）
# ==============================================================================

class ResonantEvolutionOperator:
    """
    共鳴時間発展演算子 Eres
    
    論文 §3 より：
    τパラメータによって修正された時間発展
    """
    
    def __init__(self, 
                 hamiltonian: np.ndarray,
                 params: ResonanceParameters):
        """
        Args:
            hamiltonian: ベースハミルトニアン
            params: 共鳴パラメータ
        """
        self.H_base = hamiltonian
        self.params = params
        self.dim = hamiltonian.shape[0]
    
    def get_effective_hamiltonian(self) -> np.ndarray:
        """
        実効ハミルトニアンを計算
        
        τによる修正を含む
        """
        # τによる未来寄与の修正
        tau_factor = 1 / (1 + self.params.tau)
        
        # εによる揺らぎ成分の追加
        noise_scale = self.params.epsilon / (1 + self.params.epsilon)
        noise = noise_scale * np.random.randn(self.dim, self.dim)
        noise = (noise + noise.T) / 2  # エルミート化
        
        return tau_factor * self.H_base + noise * HBAR
    
    def evolve(self, state: QuantumWorldState, time: float) -> QuantumWorldState:
        """共鳴時間発展を適用"""
        H_eff = self.get_effective_hamiltonian()
        U = _matrix_exp(-1j * H_eff * time / HBAR)
        return state.apply_operator(U)


# ==============================================================================
# ユーティリティ関数
# ==============================================================================

def _partial_trace_2x2(rho: np.ndarray, keep: str = 'first') -> np.ndarray:
    """
    2量子ビットシステムの部分トレース
    
    Args:
        rho: 4×4密度行列
        keep: 'first' or 'second'（残すサブシステム）
    """
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    
    if keep == 'first':
        # 第2サブシステムでトレース
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    else:
        # 第1サブシステムでトレース
        return np.trace(rho_reshaped, axis1=0, axis2=2)


def run_quantum_benchmark(n_runs: int = 5) -> Dict[str, Any]:
    """
    量子シミュレーションのベンチマーク
    """
    import time
    
    results = {
        'exact_times': [],
        'trotter_times': [],
        'trotter_errors': []
    }
    
    system = Minimal2x2QuantumSystem()
    initial = system.create_initial_state("superposition")
    
    for _ in range(n_runs):
        # 厳密解
        start = time.time()
        _ = system.evolve(initial, 1.0, method="exact")
        results['exact_times'].append(time.time() - start)
        
        # Trotter
        start = time.time()
        _ = system.evolve(initial, 1.0, method="trotter")
        results['trotter_times'].append(time.time() - start)
    
    # 誤差評価
    trotter = TrotterEvolution(
        H_A=np.kron(system.H_self, np.eye(2)) + np.kron(np.eye(2), system.H_other),
        H_B=system.H_int
    )
    results['trotter_error'] = trotter.actual_error(1.0)
    results['trotter_error_bound'] = trotter.error_bound(1.0)
    
    return results


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReIG2/twinRIG Quantum Simulation Module Test")
    print("=" * 60)
    
    print(f"\nQiskit available: {QISKIT_AVAILABLE}")
    
    # 2×2システムテスト
    print("\n[1] Minimal 2×2 Quantum System Test:")
    system = Minimal2x2QuantumSystem(omega_s=1.0, omega_o=0.8, g=0.3)
    
    # 各種初期状態
    for state_type in ["ground", "bell", "superposition"]:
        state = system.create_initial_state(state_type)
        print(f"  Initial state '{state_type}': {state}")
    
    # 時間発展テスト
    print("\n[2] Time Evolution Test:")
    initial = system.create_initial_state("superposition")
    evolved_exact = system.evolve(initial, 1.0, method="exact")
    evolved_trotter = system.evolve(initial, 1.0, method="trotter")
    
    fidelity = evolved_exact.fidelity(evolved_trotter)
    print(f"  Fidelity (exact vs trotter): {fidelity:.6f}")
    
    # 観測量テスト
    print("\n[3] Observable Computation Test:")
    obs = system.compute_observables(evolved_exact)
    for key, value in obs.items():
        print(f"  {key}: {value:.4f}")
    
    # Trotter誤差テスト
    print("\n[4] Trotter Error Analysis:")
    I2 = np.eye(2, dtype=complex)
    H_free = np.kron(system.H_self, I2) + np.kron(I2, system.H_other)
    
    trotter = TrotterEvolution(
        H_A=H_free,
        H_B=system.H_int,
        config=TrotterConfig(order=2, n_steps=100, delta_t=0.01)
    )
    
    actual_error = trotter.actual_error(1.0)
    error_bound = trotter.error_bound(1.0)
    print(f"  Commutator norm: {trotter.commutator_norm:.6f}")
    print(f"  Actual error: {actual_error:.2e}")
    print(f"  Error bound: {error_bound:.2e}")
    
    # シミュレーション実行テスト
    print("\n[5] Full Simulation Test:")
    results = system.run_simulation(
        initial_state="bell",
        total_time=5.0,
        n_time_points=50
    )
    print(f"  Time points: {len(results['times'])}")
    print(f"  Energy range: [{results['energy'].min():.4f}, {results['energy'].max():.4f}]")
    print(f"  Entanglement range: [{results['entanglement'].min():.4f}, {results['entanglement'].max():.4f}]")
    
    # 共鳴発展テスト
    print("\n[6] Resonant Evolution Test:")
    params = ResonanceParameters(tau=0.5, epsilon=0.1, PFH=1.0)
    resonant = ResonantEvolutionOperator(system.H_total, params)
    
    state0 = system.create_initial_state("ground")
    state1 = resonant.evolve(state0, 1.0)
    print(f"  Initial purity: {state0.purity():.4f}")
    print(f"  After resonant evolution: {state1.purity():.4f}")
    
    print("\n" + "=" * 60)
    print("All quantum simulation tests completed!")
    print("=" * 60)
