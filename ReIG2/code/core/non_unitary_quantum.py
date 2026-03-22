"""
非ユニタリ量子進化のシミュレーション
測定・デコヒーレンス・学習を含む
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# 基本演算子
I = np.eye(2)
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])

# 基底状態
ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])

def density_matrix(ket):
    """純粋状態から密度行列を生成"""
    return ket @ ket.conj().T

def unitary_evolution(rho, H, t):
    """ユニタリ進化: ρ(t) = U ρ U†"""
    U = expm(-1j * H * t)
    return U @ rho @ U.conj().T

def dephasing_channel(rho, gamma):
    """位相緩和チャネル (T2過程)"""
    K0 = np.sqrt(1 - gamma) * I
    K1 = np.sqrt(gamma) * sigma_z
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

def amplitude_damping(rho, gamma):
    """振幅減衰チャネル (T1過程)"""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

def projective_measurement(rho, projector):
    """射影測定"""
    p_success = np.real(np.trace(projector @ rho))
    if p_success > 1e-10:
        rho_measured = projector @ rho @ projector / p_success
    else:
        rho_measured = rho
    return rho_measured, p_success

def lindblad_step(rho, H, L_ops, gamma_list, dt):
    """Lindblad方程式の1ステップ (Euler法)"""
    # ユニタリ部分
    drho = -1j * (H @ rho - rho @ H)
    
    # 散逸部分
    for L, gamma in zip(L_ops, gamma_list):
        L_dag = L.conj().T
        drho += gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
    
    return rho + drho * dt

# ====================
# シミュレーション例
# ====================

def simulation_example():
    """完全なシミュレーション例"""
    
    # パラメータ設定
    omega = 1.0  # 共振周波数
    T = 10.0     # 総時間
    dt = 0.01    # 時間ステップ
    steps = int(T / dt)
    
    # デコヒーレンスパラメータ
    gamma_dephase = 0.01   # 位相緩和率
    gamma_decay = 0.005    # 振幅減衰率
    
    # ハミルトニアン
    H = omega * sigma_z
    
    # 初期状態: |+⟩ = (|0⟩ + |1⟩)/√2
    psi0 = (ket0 + ket1) / np.sqrt(2)
    rho = density_matrix(psi0)
    
    # 観測量の記録
    time_points = []
    populations = []  # |1⟩状態の占有率
    coherence = []    # 非対角成分の大きさ
    purity = []       # Tr(ρ²)
    
    # 測定演算子
    P_obs = density_matrix(ket1)  # |1⟩⟨1|
    
    # 時間発展
    for step in range(steps):
        t = step * dt
        
        # 1. ユニタリ進化
        rho = unitary_evolution(rho, H, dt)
        
        # 2. デコヒーレンス
        rho = dephasing_channel(rho, gamma_dephase * dt)
        rho = amplitude_damping(rho, gamma_decay * dt)
        
        # 3. 観測量の計算
        pop_1 = np.real(rho[1, 1])  # |1⟩の占有率
        coh = np.abs(rho[0, 1])     # コヒーレンス
        pur = np.real(np.trace(rho @ rho))  # 純粋度
        
        # 記録
        if step % 10 == 0:
            time_points.append(t)
            populations.append(pop_1)
            coherence.append(coh)
            purity.append(pur)
    
    # 結果のプロット
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # |1⟩の占有率
    axes[0].plot(time_points, populations, 'b-', linewidth=2)
    axes[0].set_ylabel('Population of |1⟩', fontsize=12)
    axes[0].set_title('Non-Unitary Quantum Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # コヒーレンス
    axes[1].plot(time_points, coherence, 'r-', linewidth=2)
    axes[1].set_ylabel('Coherence |ρ₀₁|', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 純粋度
    axes[2].plot(time_points, purity, 'g-', linewidth=2)
    axes[2].set_ylabel('Purity Tr(ρ²)', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Mixed state threshold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('non_unitary_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return time_points, populations, coherence, purity

# 実行
if __name__ == "__main__":
    print("=== 非ユニタリ量子進化シミュレーション ===\n")
    print("パラメータ:")
    print("- 共振周波数 ω = 1.0")
    print("- 位相緩和率 γ_dephase = 0.01")
    print("- 振幅減衰率 γ_decay = 0.005")
    print("- 初期状態: |+⟩ = (|0⟩ + |1⟩)/√2\n")
    
    t, pop, coh, pur = simulation_example()
    
    print(f"\n最終結果 (t={t[-1]:.2f}):")
    print(f"- |1⟩の占有率: {pop[-1]:.4f}")
    print(f"- コヒーレンス: {coh[-1]:.4f}")
    print(f"- 純粋度: {pur[-1]:.4f}")
    print(f"  (純粋状態=1, 最大混合状態=0.5)")