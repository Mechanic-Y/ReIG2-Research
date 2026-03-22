"""
ReIG2/twinRIG 完全シミュレーション
元論文 Section 6.2 の数値例を再現
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm

# ==================== 基本設定 ====================

# Pauli行列
I2 = np.eye(2)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# 基底状態
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

def tensor(*ops):
    """複数の演算子のテンソル積"""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def normalize(vec):
    """ベクトルの規格化"""
    return vec / np.linalg.norm(vec)

# ==================== 3量子ビットシステム ====================

class ThreeQubitSystem:
    """H_M ⊗ H_C ⊗ H_O の3量子ビット系"""
    
    def __init__(self, omega_M=1.0, omega_C=0.7, omega_O=0.5):
        self.omega_M = omega_M
        self.omega_C = omega_C
        self.omega_O = omega_O
        
        # ハミルトニアン: H = ω_M σ_z^M ⊗ I ⊗ I + ω_C I ⊗ σ_z^C ⊗ I + ω_O I ⊗ I ⊗ σ_z^O
        self.H = (omega_M * tensor(sigma_z, I2, I2) +
                  omega_C * tensor(I2, sigma_z, I2) +
                  omega_O * tensor(I2, I2, sigma_z))
        
        # 観測演算子: P_obs = I ⊗ I ⊗ |1⟩⟨1|
        P1 = ket1 @ ket1.conj().T
        self.P_obs = tensor(I2, I2, P1)
        
    def U_res(self, t):
        """共鳴ユニタリ: U(t) = exp(-iHt)"""
        return expm(-1j * self.H * t)
    
    def evolve(self, psi, t):
        """状態を時刻tまで進化"""
        U = self.U_res(t)
        return U @ psi
    
    def observe_M(self, psi):
        """Meaning空間の観測量 O_M"""
        # トレースアウトC,O → ρ_M
        rho_full = psi @ psi.conj().T
        rho_M = self.partial_trace(rho_full, [1, 2])  # C,Oをトレースアウト
        return rho_M
    
    def partial_trace(self, rho, trace_out):
        """部分トレース (簡易実装)"""
        # 8x8行列 → 2x2 (Mのみ残す)
        if trace_out == [1, 2]:  # C,Oをトレースアウト
            rho_M = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    # M=[i,j], C=[0,1], O=[0,1]のブロックを合計
                    for c in range(2):
                        for o in range(2):
                            idx_i = i*4 + c*2 + o
                            idx_j = j*4 + c*2 + o
                            rho_M[i, j] += rho[idx_i, idx_j]
            return rho_M
        return rho
    
    def measure_population(self, psi, subsystem='M'):
        """部分系の|1⟩占有率"""
        rho_M = self.observe_M(psi)
        return np.real(rho_M[1, 1])

# ==================== 自己参照ループ ====================

class SelfReferenceLoop:
    """T_Self の反復適用"""
    
    def __init__(self, system):
        self.system = system
        
    def T_World(self, psi, t):
        """T_World = P_obs ∘ U_res"""
        psi_evolved = self.system.evolve(psi, t)
        # 射影測定 (観測して状態を更新)
        P = self.system.P_obs
        psi_projected = P @ psi_evolved
        norm = np.linalg.norm(psi_projected)
        if norm > 1e-10:
            return psi_projected / norm
        return psi_evolved
    
    def iterate(self, psi0, N, dt):
        """N回反復"""
        psi = psi0.copy()
        history = {'psi': [psi], 'O_M': [], 'L_world': []}
        
        for n in range(N):
            # 時間発展
            psi = self.T_World(psi, dt)
            
            # 観測量
            O_M = self.system.measure_population(psi, 'M')
            
            # コスト (簡易版: 目標状態を|111⟩と仮定)
            target = tensor(ket1, ket1, ket1)
            L_world = np.linalg.norm(psi - target)**2
            
            history['psi'].append(psi)
            history['O_M'].append(O_M)
            history['L_world'].append(L_world)
        
        return history

# ==================== シミュレーション実行 ====================

def run_simulation():
    """元論文 Section 6.2 の再現"""
    
    # システム初期化
    system = ThreeQubitSystem(omega_M=1.0, omega_C=0.7, omega_O=0.5)
    loop = SelfReferenceLoop(system)
    
    # 初期状態: |Ψ₀⟩ = 1/(2√2) Σ_{i,j,k∈{0,1}} |ijk⟩
    basis_states = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                ket_i = ket0 if i == 0 else ket1
                ket_j = ket0 if j == 0 else ket1
                ket_k = ket0 if k == 0 else ket1
                basis_states.append(tensor(ket_i, ket_j, ket_k))
    
    psi0 = sum(basis_states) / (2 * np.sqrt(2))
    psi0 = normalize(psi0)
    
    # 反復実行
    N = 100
    dt = 0.1
    print("シミュレーション開始...")
    print(f"パラメータ: N={N}, dt={dt}, ω_M={system.omega_M}, ω_C={system.omega_C}, ω_O={system.omega_O}")
    
    history = loop.iterate(psi0, N, dt)
    
    # 結果表示
    print(f"\n結果:")
    print(f"O_M(N=0):   {system.measure_population(psi0, 'M'):.4f}")
    print(f"O_M(N=50):  {history['O_M'][49]:.4f}")
    print(f"O_M(N=100): {history['O_M'][99]:.4f}")
    print(f"L_world(N=100): {history['L_world'][99]:.4f}")
    
    return history, system

def plot_results(history):
    """結果のプロット"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    iterations = range(len(history['O_M']))
    
    # O_M の変化
    axes[0].plot(iterations, history['O_M'], 'b-', linewidth=2, label='O_M (Meaning観測量)')
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='目標値')
    axes[0].set_ylabel('O_M', fontsize=12)
    axes[0].set_title('自己参照ループの収束 (ReIG2 Section 6.2)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L_world の変化
    axes[1].semilogy(iterations, history['L_world'], 'g-', linewidth=2, label='L_world (コスト)')
    axes[1].set_xlabel('Iteration N', fontsize=12)
    axes[1].set_ylabel('L_world (log scale)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reig2_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 拡張分析 ====================

def extended_analysis(history, system):
    """追加の分析"""
    print("\n=== 拡張分析 ===")
    
    # 固有値分析
    H = system.H
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    print(f"\nハミルトニアンの固有値:")
    for i, e in enumerate(eigenvalues):
        print(f"  λ_{i} = {e:.4f}")
    
    # 収束率の推定
    O_M = history['O_M']
    if len(O_M) > 10:
        convergence_rate = (O_M[-1] - O_M[-10]) / 10
        print(f"\n収束率 (最後の10ステップ): {convergence_rate:.6f}/step")
    
    # エンタングルメント推定 (簡易版)
    final_psi = history['psi'][-1]
    rho_full = final_psi @ final_psi.conj().T
    purity = np.real(np.trace(rho_full @ rho_full))
    print(f"\n最終状態の純粋度: {purity:.4f}")
    print(f"  (1.0 = 純粋状態, <1.0 = 混合状態)")

# ==================== メイン実行 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ReIG2/twinRIG 完全シミュレーション")
    print("元論文 Section 6.2 の数値例")
    print("=" * 60)
    
    history, system = run_simulation()
    plot_results(history)
    extended_analysis(history, system)
    
    print("\n✓ シミュレーション完了")
    print("  グラフ保存: reig2_simulation_results.png")