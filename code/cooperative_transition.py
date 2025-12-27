"""
ReIG2/twinRIG 第14章（発展編）
協力相転移シミュレーション
Cooperative Phase Transition Simulation

Mechanic-Y / Yasuyuki Wakita
2025年12月
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# =============================================================================
# 基本行列定義
# =============================================================================

# Pauli行列
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# 共感演算子（SWAP演算子）
MIRROR_OPERATOR = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)


# =============================================================================
# 第14章 コア関数
# =============================================================================

def build_total_hamiltonian(
    lambda_empathy: float,
    gamma_trauma: float = 0.1,
    eta_asymmetry: float = 0.3,
    J_coupling: float = 0.5
) -> np.ndarray:
    """
    自己-他者全体ハミルトニアンを構築
    
    式(14.1): Ĥ_total = Ĥ_self ⊗ 1̂_other + 1̂_self ⊗ Ĥ_other + Ĥ_int
    
    Args:
        lambda_empathy: 共感強度パラメータ λ
        gamma_trauma: トラウマの深さ γ
        eta_asymmetry: 非対称性パラメータ η (0 < η < 1)
        J_coupling: 対称結合定数 J
    
    Returns:
        4x4 ハミルトニアン行列
    """
    # 自己・他者の基本ハミルトニアン
    H_self = np.kron(SIGMA_Z, IDENTITY)
    H_other = np.kron(IDENTITY, SIGMA_Z)
    
    # 対称相互作用項
    H_sym = J_coupling * (
        np.kron(SIGMA_X, SIGMA_X) + 
        np.kron(SIGMA_Y, SIGMA_Y) + 
        np.kron(SIGMA_Z, SIGMA_Z)
    )
    
    # 非対称項（トラウマ継承）
    # 式(14.2): Ĥ_asym = γ(σ^z_victim ⊗ 1̂ - η · 1̂ ⊗ σ^z_perpetrator)
    H_asym = gamma_trauma * (
        np.kron(SIGMA_Z, IDENTITY) - 
        eta_asymmetry * np.kron(IDENTITY, SIGMA_Z)
    )
    
    # 共感結合項
    H_emp = -lambda_empathy * MIRROR_OPERATOR
    
    # 全体ハミルトニアン
    H_total = H_self + H_other + H_sym + H_asym + H_emp
    
    return H_total


def compute_mutual_coherence(rho: np.ndarray) -> float:
    """
    相互コヒーレンスを計算
    
    式(14.8): C_mutual(ρ) = Tr[ρ · M̂]
    
    Args:
        rho: 密度行列 (4x4)
    
    Returns:
        相互コヒーレンス値 [0, 1]
    """
    return np.real(np.trace(rho @ MIRROR_OPERATOR))


def compute_mutual_free_energy(
    psi_self: np.ndarray,
    psi_other: np.ndarray,
    lambda_empathy: float
) -> float:
    """
    相互予測自由エネルギーの古典近似
    
    式(14.4): F_mutual = F_self + F_other - λ⟨ψ|M̂|φ⟩
    
    Args:
        psi_self: 自己の状態ベクトル
        psi_other: 他者の状態ベクトル
        lambda_empathy: 共感強度
    
    Returns:
        相互自由エネルギー
    """
    # 簡略化：エネルギー期待値として計算
    F_self = np.real(psi_self.conj() @ SIGMA_Z @ psi_self)
    F_other = np.real(psi_other.conj() @ SIGMA_Z @ psi_other)
    
    # 複合状態
    psi_combined = np.kron(psi_self, psi_other)
    M_expectation = np.real(psi_combined.conj() @ MIRROR_OPERATOR @ psi_combined)
    
    F_mutual = F_self + F_other - lambda_empathy * M_expectation
    
    return F_mutual


def critical_lambda(epsilon: float, tau: float, lambda_0: float = 1.0) -> float:
    """
    協力相転移の臨界値を計算
    
    式(14.6): λ_c(ε, τ) = λ₀ · exp(-ε/ε₀) · (1 + ατ)⁻¹
    
    Args:
        epsilon: 揺らぎパラメータ ε
        tau: 時間共鳴パラメータ τ
        lambda_0: 基準臨界値
    
    Returns:
        臨界値 λ_c
    """
    epsilon_0 = 0.1
    alpha = 0.5
    
    return lambda_0 * np.exp(-epsilon / epsilon_0) / (1 + alpha * tau)


def check_phase(lambda_val: float, PFH: float, epsilon: float, tau: float) -> str:
    """
    現在の相を判定
    
    式(14.5): λ + PFH > λ_c ⟹ 協力相
    
    Returns:
        "cooperative" or "competitive"
    """
    lambda_c = critical_lambda(epsilon, tau)
    
    if lambda_val + PFH > lambda_c:
        return "cooperative"
    else:
        return "competitive"


# =============================================================================
# シミュレーション
# =============================================================================

def simulate_time_evolution(
    lambda_empathy: float,
    N_iterations: int = 100,
    dt: float = 0.1,
    initial_state: str = "asymmetric"
) -> Tuple[np.ndarray, List[float]]:
    """
    時間発展シミュレーション
    
    Args:
        lambda_empathy: 共感強度
        N_iterations: 反復回数
        dt: 時間刻み
        initial_state: "asymmetric" (|01⟩) or "symmetric" ((|01⟩+|10⟩)/√2)
    
    Returns:
        (最終状態, 相互コヒーレンスの履歴)
    """
    # 初期状態
    if initial_state == "asymmetric":
        psi = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
    elif initial_state == "symmetric":
        psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)  # (|01⟩+|10⟩)/√2
    else:
        psi = np.array([0, 1, 0, 0], dtype=complex)
    
    # ハミルトニアン構築
    H = build_total_hamiltonian(lambda_empathy)
    
    # 時間発展演算子
    U = expm(-1j * H * dt)
    
    # 相互コヒーレンスの履歴
    coherence_history = []
    
    for _ in range(N_iterations):
        # 時間発展
        psi = U @ psi
        psi = psi / np.linalg.norm(psi)  # 正規化
        
        # 密度行列
        rho = np.outer(psi, psi.conj())
        
        # 相互コヒーレンス
        C = compute_mutual_coherence(rho)
        coherence_history.append(C)
    
    return psi, coherence_history


def scan_lambda_transition(
    lambda_range: np.ndarray,
    N_iterations: int = 100
) -> List[float]:
    """
    λ をスキャンして相転移を観測
    
    Args:
        lambda_range: λ の範囲
        N_iterations: 各点での反復回数
    
    Returns:
        最終相互コヒーレンスのリスト
    """
    final_coherences = []
    
    for lam in lambda_range:
        _, history = simulate_time_evolution(lam, N_iterations)
        final_coherences.append(history[-1])
    
    return final_coherences


# =============================================================================
# 古典（LLM）実装関数
# =============================================================================

def mirror_operator_classical(
    self_embedding: np.ndarray,
    other_embedding: np.ndarray,
    W_Q: np.ndarray = None,
    W_K: np.ndarray = None,
    W_V: np.ndarray = None
) -> np.ndarray:
    """
    共感演算子の古典実装（Cross-Attention）
    
    Args:
        self_embedding: 自己の埋め込みベクトル
        other_embedding: 他者の埋め込みベクトル
        W_Q, W_K, W_V: 重み行列（Noneの場合は恒等）
    
    Returns:
        視点変換後の埋め込み
    """
    d = len(self_embedding)
    
    # 重み行列がない場合は恒等行列
    if W_Q is None:
        W_Q = np.eye(d)
    if W_K is None:
        W_K = np.eye(d)
    if W_V is None:
        W_V = np.eye(d)
    
    # Query, Key, Value
    Q = self_embedding @ W_Q
    K = other_embedding @ W_K
    V = other_embedding @ W_V
    
    # Attention
    d_k = np.sqrt(len(Q))
    attention = np.exp(np.dot(Q, K) / d_k)
    attention = attention / np.sum(attention)
    
    # 視点変換
    self_as_other = attention * V
    
    return self_as_other


def asymmetric_context_weighting(
    context_history: List[Dict],
    eta: float = 0.3
) -> List[Dict]:
    """
    トラウマ継承モデルの古典実装
    
    Args:
        context_history: [{"role": "victim"|"perpetrator", "content": str}, ...]
        eta: 非対称性パラメータ
    
    Returns:
        重み付けされた履歴
    """
    weighted_history = []
    
    for entry in context_history:
        if entry["role"] == "victim":
            weight = 1.0
        else:
            weight = eta
        
        weighted_history.append({
            "content": entry["content"],
            "weight": weight,
            "original_role": entry["role"]
        })
    
    return weighted_history


def adjust_generation_params(phase: str) -> Dict:
    """
    相に応じた生成パラメータの調整
    
    Args:
        phase: "cooperative" or "competitive"
    
    Returns:
        生成パラメータ辞書
    """
    if phase == "cooperative":
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.1
        }
    else:
        return {
            "temperature": 1.2,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }


# =============================================================================
# 可視化
# =============================================================================

def plot_phase_transition(lambda_range: np.ndarray, coherences: List[float]):
    """
    相転移グラフをプロット
    """
    plt.figure(figsize=(10, 6))
    
    # 臨界点を計算（ε=0.1, τ=0.3, PFH=0.5 として）
    lambda_c = critical_lambda(0.1, 0.3) - 0.5
    
    plt.plot(lambda_range, coherences, 'b-', linewidth=2, label='C_mutual')
    plt.axvline(x=lambda_c, color='r', linestyle='--', label=f'λ_c = {lambda_c:.2f}')
    
    plt.xlabel('λ (Empathy Strength)', fontsize=12)
    plt.ylabel('C_mutual (Mutual Coherence)', fontsize=12)
    plt.title('Cooperative Phase Transition', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 相のラベル
    plt.text(lambda_c - 0.3, 0.8, 'Competitive\nPhase', fontsize=10, ha='center')
    plt.text(lambda_c + 0.3, 0.8, 'Cooperative\nPhase', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('phase_transition.png', dpi=150)
    plt.close()


def plot_coherence_evolution(history: List[float], lambda_val: float):
    """
    相互コヒーレンスの時間発展をプロット
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('C_mutual', fontsize=12)
    plt.title(f'Mutual Coherence Evolution (λ = {lambda_val})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'coherence_evolution_lambda{lambda_val}.png', dpi=150)
    plt.close()


# =============================================================================
# メイン実行
# =============================================================================

def main():
    """
    第14章シミュレーションのデモ実行
    """
    print("=" * 60)
    print("ReIG2/twinRIG 第14章")
    print("協力相転移シミュレーション")
    print("=" * 60)
    
    # 1. 相転移スキャン
    print("\n[1] 相転移スキャン...")
    lambda_range = np.linspace(0, 2, 50)
    coherences = scan_lambda_transition(lambda_range, N_iterations=100)
    
    # 臨界点
    lambda_c = critical_lambda(0.1, 0.3)
    print(f"    臨界値 λ_c = {lambda_c:.3f}")
    
    # 相判定例
    for lam in [0.3, 0.8, 1.5]:
        phase = check_phase(lam, PFH=0.5, epsilon=0.1, tau=0.3)
        print(f"    λ = {lam}: {phase}")
    
    # 2. 時間発展の詳細
    print("\n[2] 時間発展シミュレーション...")
    
    # 競争相での発展
    psi_comp, hist_comp = simulate_time_evolution(lambda_empathy=0.3, N_iterations=100)
    print(f"    λ=0.3 (競争相): 最終 C_mutual = {hist_comp[-1]:.3f}")
    
    # 協力相での発展
    psi_coop, hist_coop = simulate_time_evolution(lambda_empathy=1.5, N_iterations=100)
    print(f"    λ=1.5 (協力相): 最終 C_mutual = {hist_coop[-1]:.3f}")
    
    # 3. 最終状態の確認
    print("\n[3] 最終状態の確認...")
    
    # 協力相の最終状態
    print(f"    協力相最終状態:")
    print(f"    |00⟩ 成分: {np.abs(psi_coop[0])**2:.3f}")
    print(f"    |01⟩ 成分: {np.abs(psi_coop[1])**2:.3f}")
    print(f"    |10⟩ 成分: {np.abs(psi_coop[2])**2:.3f}")
    print(f"    |11⟩ 成分: {np.abs(psi_coop[3])**2:.3f}")
    
    # 4. 可視化
    print("\n[4] グラフ生成...")
    try:
        plot_phase_transition(lambda_range, coherences)
        print("    phase_transition.png を生成しました")
        
        plot_coherence_evolution(hist_coop, lambda_val=1.5)
        print("    coherence_evolution_lambda1.5.png を生成しました")
    except Exception as e:
        print(f"    グラフ生成スキップ: {e}")
    
    # 5. 古典実装デモ
    print("\n[5] 古典（LLM）実装デモ...")
    
    # 生成パラメータの調整
    params_coop = adjust_generation_params("cooperative")
    params_comp = adjust_generation_params("competitive")
    print(f"    協力相パラメータ: {params_coop}")
    print(f"    競争相パラメータ: {params_comp}")
    
    # 非対称重み付け
    sample_history = [
        {"role": "victim", "content": "傷ついた記憶"},
        {"role": "perpetrator", "content": "加害者の記憶"},
    ]
    weighted = asymmetric_context_weighting(sample_history, eta=0.3)
    print(f"    非対称重み付け: victim={weighted[0]['weight']}, perpetrator={weighted[1]['weight']}")
    
    print("\n" + "=" * 60)
    print("シミュレーション完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
