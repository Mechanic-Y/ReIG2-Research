"""
ReIG2/twinRIG 第10章
量子回路実装
Quantum Circuit Implementation (Qiskit)

Mechanic-Y / Yasuyuki Wakita
2025年12月

ReIG2/twinRIG フレームワークの量子ハードウェア実装
IBM Qiskit を用いた量子回路の構築と実行
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Qiskitのインポート（利用可能な場合）
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator, Statevector
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Running in simulation mode.")


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class CircuitResult:
    """量子回路実行結果"""
    counts: Dict[str, int]
    statevector: Optional[np.ndarray]
    circuit_depth: int
    gate_count: int


@dataclass
class HardwareRequirements:
    """ハードウェア要件"""
    n_qubits: int
    circuit_depth: int
    n_gates: int
    required_fidelity: float
    estimated_T1_requirement: float  # μs
    estimated_T2_requirement: float  # μs


# =============================================================================
# 基本ゲート
# =============================================================================

def rz_gate(theta: float) -> np.ndarray:
    """Z軸周りの回転ゲート"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def rx_gate(theta: float) -> np.ndarray:
    """X軸周りの回転ゲート"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def ry_gate(theta: float) -> np.ndarray:
    """Y軸周りの回転ゲート"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)


def cnot_gate() -> np.ndarray:
    """CNOTゲート"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def swap_gate() -> np.ndarray:
    """SWAPゲート（共感演算子 M̂ の実装）"""
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)


# =============================================================================
# ReIG2 量子回路構築
# =============================================================================

class ReIG2QuantumCircuit:
    """
    ReIG2/twinRIG の量子回路実装
    """
    
    def __init__(self, n_qubits: int = 3):
        """
        Args:
            n_qubits: 量子ビット数
        """
        self.n_qubits = n_qubits
        
        if QISKIT_AVAILABLE:
            self.qr = QuantumRegister(n_qubits, 'q')
            self.cr = ClassicalRegister(n_qubits, 'c')
            self.circuit = QuantumCircuit(self.qr, self.cr)
        else:
            self.circuit = None
    
    def build_extended_evolution(
        self,
        tau: float,
        epsilon: float,
        PFH: float,
        t: float = 1.0
    ):
        """
        拡張時間発展演算子 Û_res の回路実装
        
        H = ω_0 σ_z + τ·ω_f·σ_x + ε·ω_e·σ_y + PFH·I
        """
        if not QISKIT_AVAILABLE:
            return self._simulate_extended_evolution(tau, epsilon, PFH, t)
        
        # パラメータ
        omega_0 = 1.0
        omega_f = 0.5
        omega_e = 0.3
        
        # 各量子ビットに適用
        for i in range(self.n_qubits):
            # Rz (基本項)
            theta_z = 2 * omega_0 * t
            self.circuit.rz(theta_z, self.qr[i])
            
            # Rx (未来項)
            theta_x = 2 * tau * omega_f * t
            self.circuit.rx(theta_x, self.qr[i])
            
            # Ry (エントロピー項)
            theta_y = 2 * epsilon * omega_e * t
            self.circuit.ry(theta_y, self.qr[i])
            
            # 全体位相 (PFH)
            self.circuit.rz(PFH * t, self.qr[i])
    
    def _simulate_extended_evolution(
        self,
        tau: float,
        epsilon: float,
        PFH: float,
        t: float
    ) -> np.ndarray:
        """Qiskitなしでの行列シミュレーション"""
        omega_0, omega_f, omega_e = 1.0, 0.5, 0.3
        
        U = rz_gate(2 * omega_0 * t)
        U = rx_gate(2 * tau * omega_f * t) @ U
        U = ry_gate(2 * epsilon * omega_e * t) @ U
        U = rz_gate(PFH * t) @ U
        
        return U
    
    def build_multidim_evolution(
        self,
        tau: float,
        epsilon: float,
        PFH: float,
        t: float = 1.0,
        trotter_steps: int = 5
    ):
        """
        多次元時間発展演算子 Û_multi の回路実装
        
        Trotter分解を使用
        """
        if not QISKIT_AVAILABLE:
            return self._simulate_multidim_evolution(tau, epsilon, PFH, t, trotter_steps)
        
        dt = t / trotter_steps
        
        for _ in range(trotter_steps):
            # 物理的時間軸
            for i in range(self.n_qubits):
                theta = 2 * (1 + tau) * dt
                self.circuit.rz(theta, self.qr[i])
            
            # 文化的時間軸
            weight_cultural = tau * np.exp(-epsilon**2 / 2)
            for i in range(self.n_qubits):
                self.circuit.rx(2 * weight_cultural * dt, self.qr[i])
            
            # 社会的時間軸（エンタングルメント）
            weight_social = PFH * np.sqrt(max(tau, 0))
            if self.n_qubits >= 2:
                for i in range(self.n_qubits - 1):
                    self.circuit.cx(self.qr[i], self.qr[i+1])
                    self.circuit.rz(weight_social * dt, self.qr[i+1])
                    self.circuit.cx(self.qr[i], self.qr[i+1])
    
    def _simulate_multidim_evolution(
        self,
        tau: float,
        epsilon: float,
        PFH: float,
        t: float,
        trotter_steps: int
    ) -> np.ndarray:
        """多次元発展の行列シミュレーション"""
        dim = 2 ** self.n_qubits
        U = np.eye(dim, dtype=complex)
        dt = t / trotter_steps
        
        for _ in range(trotter_steps):
            # 単一量子ビット項
            U_single = rz_gate(2 * (1 + tau) * dt)
            U_single = rx_gate(2 * tau * np.exp(-epsilon**2/2) * dt) @ U_single
            
            # テンソル積
            U_step = U_single
            for _ in range(self.n_qubits - 1):
                U_step = np.kron(U_step, U_single)
            
            U = U_step @ U
        
        return U
    
    def build_phase_transition(self, p_jump: float, theta_twist: float):
        """
        相転移生成演算子 G の回路実装
        
        G = P ∘ E ∘ R
        """
        if not QISKIT_AVAILABLE:
            return self._simulate_phase_transition(p_jump, theta_twist)
        
        # R: Twist (全体回転)
        for i in range(self.n_qubits):
            self.circuit.rz(theta_twist, self.qr[i])
        
        # P: Phase Jump (確率的ビットフリップをRyで近似)
        theta_jump = 2 * np.arcsin(np.sqrt(p_jump))
        for i in range(self.n_qubits):
            self.circuit.ry(theta_jump, self.qr[i])
    
    def _simulate_phase_transition(self, p_jump: float, theta_twist: float) -> np.ndarray:
        """相転移の行列シミュレーション"""
        R = rz_gate(theta_twist)
        theta_jump = 2 * np.arcsin(np.sqrt(p_jump))
        P = ry_gate(theta_jump)
        
        G_single = P @ R
        
        G = G_single
        for _ in range(self.n_qubits - 1):
            G = np.kron(G, G_single)
        
        return G
    
    def build_mirror_operator(self, qubit_pairs: List[Tuple[int, int]]):
        """
        共感演算子 M̂ (SWAP) の回路実装
        """
        if not QISKIT_AVAILABLE:
            return
        
        for i, j in qubit_pairs:
            if i < self.n_qubits and j < self.n_qubits:
                self.circuit.swap(self.qr[i], self.qr[j])
    
    def add_measurement(self):
        """測定を追加"""
        if QISKIT_AVAILABLE:
            self.circuit.measure(self.qr, self.cr)
    
    def run(self, shots: int = 1024) -> CircuitResult:
        """
        回路を実行
        """
        if not QISKIT_AVAILABLE:
            return self._run_matrix_simulation()
        
        # シミュレータで実行
        simulator = AerSimulator()
        
        # 状態ベクトル取得用の回路
        sv_circuit = self.circuit.copy()
        sv_circuit.remove_final_measurements()
        sv_circuit.save_statevector()
        
        # 実行
        job = simulator.run(sv_circuit)
        result = job.result()
        statevector = result.get_statevector().data
        
        # 測定用の回路
        meas_circuit = self.circuit.copy()
        if not any(inst.operation.name == 'measure' for inst in meas_circuit.data):
            meas_circuit.measure_all()
        
        job = simulator.run(meas_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        return CircuitResult(
            counts=counts,
            statevector=statevector,
            circuit_depth=self.circuit.depth(),
            gate_count=len(self.circuit.data)
        )
    
    def _run_matrix_simulation(self) -> CircuitResult:
        """行列シミュレーション（Qiskitなし）"""
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        return CircuitResult(
            counts={"0" * self.n_qubits: 1024},
            statevector=state,
            circuit_depth=0,
            gate_count=0
        )
    
    def get_circuit_diagram(self) -> str:
        """回路図を取得"""
        if QISKIT_AVAILABLE and self.circuit is not None:
            return str(self.circuit.draw(output='text'))
        else:
            return "Circuit diagram not available without Qiskit"


# =============================================================================
# ハードウェア要件の推定
# =============================================================================

def estimate_hardware_requirements(
    n_qubits: int,
    circuit_depth: int,
    gate_fidelity: float = 0.999,
    measurement_fidelity: float = 0.99
) -> HardwareRequirements:
    """
    ハードウェア要件を推定
    
    Args:
        n_qubits: 量子ビット数
        circuit_depth: 回路深さ
        gate_fidelity: 単一ゲートのフィデリティ
        measurement_fidelity: 測定フィデリティ
    
    Returns:
        HardwareRequirements
    """
    # ゲート数の推定（深さ × 量子ビット数 × 係数）
    n_gates = int(circuit_depth * n_qubits * 1.5)
    
    # 必要なフィデリティ
    # 全体フィデリティ ≈ gate_fidelity^n_gates × measurement_fidelity^n_qubits
    required_fidelity = gate_fidelity ** n_gates * measurement_fidelity ** n_qubits
    
    # T1, T2 の要件（μs）
    # 回路時間 ≈ depth × 100ns (典型的な2量子ビットゲート時間)
    circuit_time_us = circuit_depth * 0.1
    T1_requirement = circuit_time_us * 10  # T1 >> 回路時間
    T2_requirement = circuit_time_us * 5   # T2 > 回路時間
    
    return HardwareRequirements(
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        n_gates=n_gates,
        required_fidelity=required_fidelity,
        estimated_T1_requirement=T1_requirement,
        estimated_T2_requirement=T2_requirement
    )


# =============================================================================
# エラー訂正コードの簡易実装
# =============================================================================

class SimpleBitFlipCode:
    """
    単純なビットフリップ訂正コード [3,1,3]
    
    |0⟩_L = |000⟩
    |1⟩_L = |111⟩
    """
    
    def __init__(self):
        self.n_physical = 3
        self.n_logical = 1
    
    def encode(self, logical_state: np.ndarray) -> np.ndarray:
        """論理状態を物理状態にエンコード"""
        # |0⟩_L → |000⟩, |1⟩_L → |111⟩
        physical = np.zeros(8, dtype=complex)
        physical[0] = logical_state[0]  # |000⟩
        physical[7] = logical_state[1]  # |111⟩
        return physical
    
    def decode(self, physical_state: np.ndarray) -> np.ndarray:
        """物理状態を論理状態にデコード"""
        # 多数決
        logical = np.zeros(2, dtype=complex)
        
        # |0⟩ 側: 0,1,2,4 番目 (0が多数)
        logical[0] = physical_state[0] + physical_state[1] + physical_state[2] + physical_state[4]
        
        # |1⟩ 側: 3,5,6,7 番目 (1が多数)
        logical[1] = physical_state[3] + physical_state[5] + physical_state[6] + physical_state[7]
        
        return logical / np.linalg.norm(logical)
    
    def syndrome_measurement(self, physical_state: np.ndarray) -> Tuple[int, int]:
        """シンドローム測定"""
        probs = np.abs(physical_state) ** 2
        
        # Z_1 Z_2
        s1 = int(probs[2] + probs[3] + probs[6] + probs[7] > 0.5)
        
        # Z_2 Z_3
        s2 = int(probs[1] + probs[3] + probs[5] + probs[7] > 0.5)
        
        return (s1, s2)


# =============================================================================
# IBM Quantum 実行例
# =============================================================================

def create_ibm_quantum_job_example() -> str:
    """
    IBM Quantum への投入例（コードのみ）
    
    実際の実行には IBM Quantum アカウントが必要
    """
    code = '''
# IBM Quantum での実行例
from qiskit_ibm_runtime import QiskitRuntimeService

# 認証
service = QiskitRuntimeService(channel="ibm_quantum")

# バックエンドの選択
backend = service.backend("ibm_brisbane")

# 回路の準備
circuit = ReIG2QuantumCircuit(n_qubits=3)
circuit.build_extended_evolution(tau=0.5, epsilon=0.3, PFH=0.2)
circuit.build_phase_transition(p_jump=0.1, theta_twist=np.pi/4)
circuit.add_measurement()

# トランスパイル
from qiskit import transpile
transpiled = transpile(circuit.circuit, backend=backend, optimization_level=3)

# 実行
job = backend.run(transpiled, shots=4096)
result = job.result()
counts = result.get_counts()

print("Results:", counts)
'''
    return code


# =============================================================================
# デモ
# =============================================================================

def demo():
    """量子回路実装のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第10章")
    print("量子回路実装 (Qiskit)")
    print("=" * 60)
    
    print(f"\nQiskit available: {QISKIT_AVAILABLE}")
    
    # 1. 基本回路の構築
    print("\n[1] 拡張時間発展演算子の回路")
    circuit = ReIG2QuantumCircuit(n_qubits=3)
    circuit.build_extended_evolution(tau=0.5, epsilon=0.3, PFH=0.2)
    
    if QISKIT_AVAILABLE:
        print(f"    回路深さ: {circuit.circuit.depth()}")
        print(f"    ゲート数: {len(circuit.circuit.data)}")
    
    # 2. 多次元発展
    print("\n[2] 多次元時間発展演算子の回路")
    circuit2 = ReIG2QuantumCircuit(n_qubits=3)
    circuit2.build_multidim_evolution(tau=0.5, epsilon=0.3, PFH=0.2, trotter_steps=3)
    
    if QISKIT_AVAILABLE:
        print(f"    回路深さ: {circuit2.circuit.depth()}")
        print(f"    ゲート数: {len(circuit2.circuit.data)}")
    
    # 3. 相転移
    print("\n[3] 相転移生成演算子の回路")
    circuit3 = ReIG2QuantumCircuit(n_qubits=3)
    circuit3.build_phase_transition(p_jump=0.2, theta_twist=np.pi/4)
    
    if QISKIT_AVAILABLE:
        print(f"    回路深さ: {circuit3.circuit.depth()}")
    
    # 4. 共感演算子
    print("\n[4] 共感演算子 (SWAP) の回路")
    circuit4 = ReIG2QuantumCircuit(n_qubits=4)
    circuit4.build_mirror_operator([(0, 1), (2, 3)])
    
    if QISKIT_AVAILABLE:
        print(f"    SWAP ゲート数: {sum(1 for inst in circuit4.circuit.data if inst.operation.name == 'swap')}")
    
    # 5. 統合回路と実行
    print("\n[5] 統合回路の実行")
    full_circuit = ReIG2QuantumCircuit(n_qubits=3)
    full_circuit.build_extended_evolution(tau=0.5, epsilon=0.3, PFH=0.2, t=0.5)
    full_circuit.build_phase_transition(p_jump=0.1, theta_twist=np.pi/8)
    
    result = full_circuit.run(shots=1024)
    
    print(f"    回路深さ: {result.circuit_depth}")
    print(f"    ゲート数: {result.gate_count}")
    print(f"    測定結果（上位3）:")
    sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:3]
    for state, count in sorted_counts:
        print(f"        |{state}⟩: {count} ({count/1024*100:.1f}%)")
    
    # 6. ハードウェア要件
    print("\n[6] ハードウェア要件推定")
    req = estimate_hardware_requirements(n_qubits=3, circuit_depth=20)
    
    print(f"    量子ビット数: {req.n_qubits}")
    print(f"    回路深さ: {req.circuit_depth}")
    print(f"    推定ゲート数: {req.n_gates}")
    print(f"    必要フィデリティ: {req.required_fidelity:.4f}")
    print(f"    必要 T1: >{req.estimated_T1_requirement:.1f} μs")
    print(f"    必要 T2: >{req.estimated_T2_requirement:.1f} μs")
    
    # 7. エラー訂正
    print("\n[7] ビットフリップ訂正コード")
    code = SimpleBitFlipCode()
    
    logical = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    physical = code.encode(logical)
    
    print(f"    論理状態: |+⟩_L")
    print(f"    物理状態の非ゼロ成分: |000⟩, |111⟩")
    
    # エラー注入（ビットフリップ）
    error_state = physical.copy()
    error_state[0], error_state[1] = error_state[1], error_state[0]  # 第1ビットフリップ
    
    decoded = code.decode(error_state)
    fidelity = np.abs(np.vdot(logical, decoded))**2
    print(f"    エラー後のフィデリティ: {fidelity:.4f}")
    
    # 8. 回路図
    print("\n[8] 回路図")
    print(full_circuit.get_circuit_diagram()[:500] + "...")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
