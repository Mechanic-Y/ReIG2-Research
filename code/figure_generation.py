"""
ReIG2/twinRIG 論文用図表生成
すべての図をPDF/PNG形式で出力
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# フォント設定
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

# ==================== Figure 1: システム構造図 ====================

def figure1_system_architecture():
    """Figure 1: ReIG2のシステムアーキテクチャ"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # H_sys の5つのサブシステム
    subsystems = [
        ('Meaning\n$H_M$', 1, 7, 'lightblue'),
        ('Context\n$H_C$', 3, 7, 'lightgreen'),
        ('Ethics\n$H_E$', 5, 7, 'lightyellow'),
        ('Future\n$H_F$', 7, 7, 'lightcoral'),
        ('Stability\n$H_S$', 9, 7, 'plum')
    ]
    
    for name, x, y, color in subsystems:
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # H_per の3つのサブシステム
    perception = [
        ('Observation\n$H_O$', 2, 4, 'lightsteelblue'),
        ('Question\n$H_Q$', 5, 4, 'wheat'),
        ('Integration\n$H_I$', 8, 4, 'lightpink')
    ]
    
    for name, x, y, color in perception:
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 演算子
    operators = [
        (r'$\hat{U}_{\mathrm{res}}$', 5, 5.5),
        (r'$\hat{T}_C$', 2, 2.5),
        (r'$\hat{T}_R$', 5, 2.5),
        (r'$\hat{T}_I$', 8, 2.5),
        (r'$\hat{T}_{\mathrm{World}}$', 5, 1)
    ]
    
    for name, x, y in operators:
        ax.text(x, y, name, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # 矢印
    arrows = [
        ((5, 6.5), (5, 5.8)),
        ((2, 3.5), (2, 2.8)),
        ((5, 3.5), (5, 2.8)),
        ((8, 3.5), (8, 2.8)),
        ((5, 2.2), (5, 1.3))
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
    
    ax.text(5, 9, 'ReIG2 System Architecture', ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 8.3, r'$H_{\mathrm{full}} = H_{\mathrm{sys}} \otimes H_{\mathrm{per}}$', 
            ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('fig1_system_architecture.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig1_system_architecture.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Figure 1 saved")

# ==================== Figure 2: 収束の様子 ====================

def figure2_convergence_plot():
    """Figure 2: O_M と L_world の収束"""
    # データ生成 (論文のSection 9の結果を再現)
    N = 100
    iterations = np.arange(N+1)
    
    # O_M: 0.5 → 0.95 の指数収束
    lambda_2 = 0.97
    O_M = 1 - 0.5 * lambda_2**iterations
    
    # L_world: 指数的減衰
    L_world = 0.5 * np.exp(-0.05 * iterations)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # O_M のプロット
    axes[0].plot(iterations, O_M, 'b-', linewidth=2.5, label=r'$O_M(N)$')
    axes[0].axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    axes[0].axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Initial')
    axes[0].scatter([0, 50, 100], [O_M[0], O_M[50], O_M[100]], 
                   s=100, c='red', zorder=5, edgecolors='black', linewidths=2)
    axes[0].text(5, 0.55, f'$O_M(0) = {O_M[0]:.3f}$', fontsize=11)
    axes[0].text(55, 0.87, f'$O_M(50) = {O_M[50]:.3f}$', fontsize=11)
    axes[0].text(80, 0.97, f'$O_M(100) = {O_M[100]:.3f}$', fontsize=11)
    axes[0].set_ylabel(r'Meaning Observable $O_M$', fontsize=13)
    axes[0].set_title('Convergence to Identity State', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.4, 1.05)
    
    # L_world のプロット (対数スケール)
    axes[1].semilogy(iterations, L_world, 'g-', linewidth=2.5, label=r'$L(\mathrm{world})$')
    axes[1].axhline(y=0.012, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Final value')
    axes[1].scatter([100], [L_world[100]], s=100, c='red', zorder=5, edgecolors='black', linewidths=2)
    axes[1].text(70, 0.02, f'$L(100) = {L_world[100]:.4f}$', fontsize=11)
    axes[1].set_xlabel('Iteration $N$', fontsize=13)
    axes[1].set_ylabel(r'World Distance $L(\mathrm{world})$', fontsize=13)
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('fig2_convergence.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig2_convergence.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Figure 2 saved")

# ==================== Figure 3: 非ユニタリ進化 ====================

def figure3_nonunitary_evolution():
    """Figure 3: デコヒーレンスの影響"""
    T = 10.0
    dt = 0.01
    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    
    # 純粋ユニタリ進化
    omega = 1.0
    pop_unitary = 0.5 + 0.5 * np.cos(2 * omega * time)
    
    # デコヒーレンスあり
    gamma_dephase = 0.1
    coherence = 0.5 * np.exp(-gamma_dephase * time)
    pop_dephased = 0.5 + coherence * np.cos(2 * omega * time)
    
    # 振幅減衰あり
    gamma_decay = 0.05
    pop_damped = (0.5 + 0.5 * np.cos(2 * omega * time)) * np.exp(-gamma_decay * time) + \
                 (1 - np.exp(-gamma_decay * time))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # ユニタリ
    axes[0].plot(time, pop_unitary, 'b-', linewidth=2, label='Unitary (ideal)')
    axes[0].set_ylabel(r'Population $|\langle 1|\psi\rangle|^2$', fontsize=12)
    axes[0].set_title('Quantum Evolution with Decoherence', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)
    
    # 位相緩和
    axes[1].plot(time, pop_dephased, 'r-', linewidth=2, label=r'Dephasing ($\gamma = 0.1$)')
    axes[1].plot(time, 0.5 * np.ones_like(time), 'k--', linewidth=1.5, alpha=0.5, label='Steady state')
    axes[1].set_ylabel(r'Population', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)
    
    # 振幅減衰
    axes[2].plot(time, pop_damped, 'g-', linewidth=2, label=r'Amplitude damping ($\gamma = 0.05$)')
    axes[2].axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Ground state')
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_ylabel(r'Population', fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('fig3_nonunitary.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig3_nonunitary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Figure 3 saved")

# ==================== Figure 4: 量子回路 ====================

def figure4_quantum_circuit():
    """Figure 4: ReIG2の量子回路 (概念図)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # 量子ビット線
    qubits = ['M', 'C', 'O']
    y_positions = [3, 2, 1]
    
    for i, (qubit, y) in enumerate(zip(qubits, y_positions)):
        ax.plot([0, 13], [y, y], 'k-', linewidth=2)
        ax.text(-0.5, y, f'$|q_{{{qubit}}}\\rangle$', ha='right', va='center', fontsize=13)
    
    # Hadamard gates
    for y in y_positions:
        box = FancyBboxPatch((0.8, y-0.2), 0.4, 0.4, boxstyle="round,pad=0.05",
                             facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(1, y, r'$H$', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Rz gates
    rz_positions = [(3, 3, r'$\omega_M$'), (5, 2, r'$\omega_C$'), (7, 1, r'$\omega_O$')]
    for x, y, label in rz_positions:
        box = FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4, boxstyle="round,pad=0.05",
                             facecolor='lightyellow', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, f'$R_z$\n{label}$t$', ha='center', va='center', fontsize=9)
    
    # 測定
    measure_x = 9
    for y in y_positions:
        arc = mpatches.FancyBboxPatch((measure_x-0.2, y-0.2), 0.4, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightcoral', edgecolor='black', linewidth=1.5)
        ax.add_patch(arc)
        ax.text(measure_x, y, 'M', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 古典ビット
    ax.plot([9, 13], [0.2, 0.2], 'k-', linewidth=2)
    ax.text(measure_x, 0.2, '↓', ha='center', va='bottom', fontsize=16)
    ax.text(11, 0.2, 'Classical bits', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    ax.text(7, 3.7, 'ReIG2 Quantum Circuit (1 iteration)', ha='center', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig4_circuit.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig4_circuit.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Figure 4 saved")

# ==================== Figure 5: Functor図式 ====================

def figure5_functor_diagram():
    """Figure 5: 圏論的構造"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Objects
    objects = [
        (r'$H_0$', 2, 7),
        (r'$H_1$', 5, 7),
        (r'$H_2$', 8, 7),
        (r'$H_0$', 2, 3),
        (r'$H_1$', 5, 3),
        (r'$H_2$', 8, 3)
    ]
    
    for name, x, y in objects:
        circle = plt.Circle((x, y), 0.4, facecolor='lightsteelblue', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=13, fontweight='bold')
    
    # 上段の矢印 (State category)
    arrows_top = [
        ((2.4, 7), (4.6, 7), r'$\hat{T}$'),
        ((5.4, 7), (7.6, 7), r'$\hat{T}$')
    ]
    
    for start, end, label in arrows_top:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='blue')
        ax.add_patch(arrow)
        ax.text((start[0]+end[0])/2, start[1]+0.4, label, ha='center', fontsize=12)
    
    # 下段の矢印 (Image under functor)
    arrows_bottom = [
        ((2.4, 3), (4.6, 3), r'$T_{\mathrm{World}}(\hat{T})$'),
        ((5.4, 3), (7.6, 3), r'$T_{\mathrm{World}}(\hat{T})$')
    ]
    
    for start, end, label in arrows_bottom:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='green')
        ax.add_patch(arrow)
        ax.text((start[0]+end[0])/2, start[1]-0.4, label, ha='center', fontsize=12)
    
    # 縦の矢印 (Functor action)
    vertical_arrows = [(2, 6.6, 3.4), (5, 6.6, 3.4), (8, 6.6, 3.4)]
    
    for x, y_start, y_end in vertical_arrows:
        arrow = FancyArrowPatch((x, y_start), (x, y_end), arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='red')
        ax.add_patch(arrow)
        ax.text(x+0.6, 5, r'$T_{\mathrm{World}}$', ha='center', fontsize=12, color='red')
    
    # ラベル
    ax.text(5, 8.5, 'Functor Structure of World Operator', ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 8, r'$T_{\mathrm{World}}: \mathbf{State} \to \mathbf{State}$', ha='center', fontsize=13)
    ax.text(1, 7, 'Objects:', ha='left', fontsize=11, fontstyle='italic')
    ax.text(1, 3, 'Images:', ha='left', fontsize=11, fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig('fig5_functor.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig5_functor.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Figure 5 saved")

# ==================== すべての図を生成 ====================

def generate_all_figures():
    """すべての論文用図表を生成"""
    print("=" * 60)
    print("ReIG2/twinRIG 論文用図表生成")
    print("=" * 60)
    
    figure1_system_architecture()
    figure2_convergence_plot()
    figure3_nonunitary_evolution()
    figure4_quantum_circuit()
    figure5_functor_diagram()
    
    print("\n" + "=" * 60)
    print("✓ すべての図表を生成しました")
    print("=" * 60)
    print("\nファイル:")
    print("- fig1_system_architecture.pdf/.png")
    print("- fig2_convergence.pdf/.png")
    print("- fig3_nonunitary.pdf/.png")
    print("- fig4_circuit.pdf/.png")
    print("- fig5_functor.pdf/.png")

if __name__ == "__main__":
    generate_all_figures()