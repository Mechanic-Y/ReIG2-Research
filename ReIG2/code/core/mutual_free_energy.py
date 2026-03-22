"""
ReIG2/twinRIG 第14章（発展編）
相互予測自由エネルギーの計算
Mutual Predictive Free Energy

Mechanic-Y / Yasuyuki Wakita
2025年12月

量子概念 F_mutual の古典コンピュータ（LLM）への実装
Fristonの自由エネルギー原理（FEP）を他者性に拡張
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass

# =============================================================================
# 定数・設定
# =============================================================================

# デフォルトパラメータ
DEFAULT_LAMBDA_0 = 1.0      # 基準臨界値
DEFAULT_EPSILON_0 = 0.1     # 揺らぎスケール
DEFAULT_ALPHA = 0.5         # 時間共鳴係数


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class FreeEnergyResult:
    """自由エネルギー計算結果"""
    F_self: float           # 自己の自由エネルギー
    F_other: float          # 他者の自由エネルギー
    empathy_term: float     # 共感項 λ⟨M̂⟩
    F_mutual: float         # 相互自由エネルギー
    lambda_val: float       # 使用した共感強度
    

@dataclass
class PhaseTransitionResult:
    """相転移判定結果"""
    phase: str              # "cooperative" or "competitive"
    lambda_c: float         # 臨界値
    margin: float           # 臨界値からのマージン
    parameters: Dict        # 使用したパラメータ


# =============================================================================
# 量子版：相互予測自由エネルギー（参照用）
# =============================================================================

def quantum_mutual_free_energy(
    rho_self: np.ndarray,
    rho_other: np.ndarray,
    H_self: np.ndarray,
    H_other: np.ndarray,
    M_operator: np.ndarray,
    lambda_empathy: float,
    beta: float = 1.0
) -> float:
    """
    量子版：相互予測自由エネルギー
    
    F_mutual = F_self + F_other - λ⟨M̂⟩
    
    ここで F = ⟨H⟩ - (1/β)S(ρ) （量子自由エネルギー）
    
    Args:
        rho_self: 自己の密度行列
        rho_other: 他者の密度行列
        H_self: 自己のハミルトニアン
        H_other: 他者のハミルトニアン
        M_operator: 共感演算子（SWAP）
        lambda_empathy: 共感強度パラメータ
        beta: 逆温度
    
    Returns:
        F_mutual: 相互自由エネルギー
    """
    # 自己の自由エネルギー
    E_self = np.real(np.trace(rho_self @ H_self))
    S_self = -np.real(np.trace(rho_self @ _safe_log(rho_self)))
    F_self = E_self - S_self / beta
    
    # 他者の自由エネルギー
    E_other = np.real(np.trace(rho_other @ H_other))
    S_other = -np.real(np.trace(rho_other @ _safe_log(rho_other)))
    F_other = E_other - S_other / beta
    
    # 複合系の共感項
    rho_combined = np.kron(rho_self, rho_other)
    M_expectation = np.real(np.trace(rho_combined @ M_operator))
    
    # 相互自由エネルギー
    F_mutual = F_self + F_other - lambda_empathy * M_expectation
    
    return F_mutual


def _safe_log(matrix: np.ndarray) -> np.ndarray:
    """数値安定性を考慮した行列対数"""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # ゼロ除算防止
    log_eigenvalues = np.log(eigenvalues)
    return eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T.conj()


# =============================================================================
# 古典版：相互予測自由エネルギー
# =============================================================================

def mutual_free_energy(
    self_state: np.ndarray,
    other_state: np.ndarray,
    lambda_empathy: float,
    perplexity_func: Optional[Callable] = None
) -> FreeEnergyResult:
    """
    相互予測自由エネルギーの古典近似
    
    F_mutual = F_self(ψ ∥ Ĥ_self) + F_other(φ ∥ Ĥ_other) - λ⟨ψ|M̂|φ⟩
    
    古典実装では：
    - F_self, F_other → perplexity または エネルギー近似
    - ⟨M̂⟩ → コサイン類似度
    
    Args:
        self_state: 自己の状態ベクトル（埋め込み）
        other_state: 他者の状態ベクトル
        lambda_empathy: 共感強度パラメータ λ
        perplexity_func: カスタムperplexity関数（オプション）
    
    Returns:
        FreeEnergyResult: 計算結果
    """
    # 自己の自由エネルギー
    if perplexity_func is not None:
        F_self = perplexity_func(self_state)
        F_other = perplexity_func(other_state)
    else:
        # デフォルト：ノルムベースの近似
        F_self = _approximate_free_energy(self_state)
        F_other = _approximate_free_energy(other_state)
    
    # 共感項（コサイン類似度）
    M_expectation = _cosine_similarity(self_state, other_state)
    empathy_term = lambda_empathy * M_expectation
    
    # 相互自由エネルギー
    F_mutual = F_self + F_other - empathy_term
    
    return FreeEnergyResult(
        F_self=F_self,
        F_other=F_other,
        empathy_term=empathy_term,
        F_mutual=F_mutual,
        lambda_val=lambda_empathy
    )


def _approximate_free_energy(state: np.ndarray) -> float:
    """
    状態ベクトルから自由エネルギーを近似
    
    LLM文脈では perplexity の対数に対応
    """
    # エントロピー近似：分布の広がり
    state_normalized = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
    entropy = -np.sum(state_normalized * np.log(state_normalized + 1e-10))
    
    # エネルギー近似：ノルム
    energy = np.linalg.norm(state)
    
    # 自由エネルギー近似
    return energy - entropy


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度を計算"""
    dot_product = np.dot(a.flatten(), b.flatten())
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# =============================================================================
# LLM対話用：コスト関数
# =============================================================================

def dialogue_cost_function(
    self_embedding: np.ndarray,
    other_embedding: np.ndarray,
    self_perplexity: float,
    other_perplexity: float,
    lambda_empathy: float,
    empathy_weight: float = 1.0
) -> Dict[str, float]:
    """
    対話システム用コスト関数
    
    LLMの対話品質を評価するコスト関数
    perplexity（言語モデルの不確実性）と共感スコアを組み合わせ
    
    Args:
        self_embedding: 自己（システム）の発話埋め込み
        other_embedding: 他者（ユーザー）の発話埋め込み
        self_perplexity: 自己の発話のperplexity
        other_perplexity: 他者の発話のperplexity（予測）
        lambda_empathy: 共感強度
        empathy_weight: 共感項の重み
    
    Returns:
        コスト関数の各成分
    """
    # 言語コスト（perplexityの対数）
    language_cost_self = np.log(self_perplexity + 1)
    language_cost_other = np.log(other_perplexity + 1)
    
    # 共感スコア
    empathy_score = _cosine_similarity(self_embedding, other_embedding)
    
    # 相互コスト
    mutual_cost = (
        language_cost_self + 
        language_cost_other - 
        lambda_empathy * empathy_weight * empathy_score
    )
    
    return {
        "language_cost_self": language_cost_self,
        "language_cost_other": language_cost_other,
        "empathy_score": empathy_score,
        "empathy_term": lambda_empathy * empathy_weight * empathy_score,
        "mutual_cost": mutual_cost
    }


# =============================================================================
# 協力相転移の判定
# =============================================================================

def critical_lambda(
    epsilon: float,
    tau: float,
    lambda_0: float = DEFAULT_LAMBDA_0,
    epsilon_0: float = DEFAULT_EPSILON_0,
    alpha: float = DEFAULT_ALPHA
) -> float:
    """
    協力相転移の臨界値を計算
    
    λ_c(ε, τ) = λ₀ · exp(-ε/ε₀) · (1 + ατ)⁻¹
    
    Args:
        epsilon: 揺らぎパラメータ ε
        tau: 時間共鳴パラメータ τ
        lambda_0: 基準臨界値
        epsilon_0: 揺らぎスケール
        alpha: 時間共鳴係数
    
    Returns:
        臨界値 λ_c
    """
    return lambda_0 * np.exp(-epsilon / epsilon_0) / (1 + alpha * tau)


def check_cooperative_transition(
    lambda_val: float,
    PFH: float,
    epsilon: float,
    tau: float,
    lambda_0: float = DEFAULT_LAMBDA_0,
    epsilon_0: float = DEFAULT_EPSILON_0,
    alpha: float = DEFAULT_ALPHA
) -> PhaseTransitionResult:
    """
    協力相転移条件をチェック
    
    条件: λ + PFH > λ_c(ε, τ) ⟹ 協力相
    
    Args:
        lambda_val: 共感強度 λ
        PFH: 哲学的調和パラメータ
        epsilon: 揺らぎパラメータ ε
        tau: 時間共鳴パラメータ τ
        lambda_0, epsilon_0, alpha: モデルパラメータ
    
    Returns:
        PhaseTransitionResult: 相転移判定結果
    """
    lambda_c = critical_lambda(epsilon, tau, lambda_0, epsilon_0, alpha)
    
    total = lambda_val + PFH
    margin = total - lambda_c
    
    if total > lambda_c:
        phase = "cooperative"
    else:
        phase = "competitive"
    
    return PhaseTransitionResult(
        phase=phase,
        lambda_c=lambda_c,
        margin=margin,
        parameters={
            "lambda": lambda_val,
            "PFH": PFH,
            "epsilon": epsilon,
            "tau": tau,
            "lambda_0": lambda_0,
            "epsilon_0": epsilon_0,
            "alpha": alpha
        }
    )


def adjust_generation_params(phase: str) -> Dict[str, float]:
    """
    相に応じたLLM生成パラメータを調整
    
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
            "frequency_penalty": 0.1,
            "description": "協調的・共感的な応答を促進"
        }
    else:
        return {
            "temperature": 1.2,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "description": "多様性・競争的な応答"
        }


# =============================================================================
# 相互コヒーレンスの計算
# =============================================================================

def compute_mutual_coherence(
    self_state: np.ndarray,
    other_state: np.ndarray
) -> float:
    """
    相互コヒーレンスを計算（古典近似）
    
    C_mutual(ρ) = Tr[ρ · M̂] = ⟨M̂⟩_ρ
    
    古典では視点の整合度として近似
    
    Args:
        self_state: 自己の状態
        other_state: 他者の状態
    
    Returns:
        相互コヒーレンス [0, 1]
    """
    # コサイン類似度を[0,1]に正規化
    similarity = _cosine_similarity(self_state, other_state)
    coherence = (similarity + 1) / 2
    
    return coherence


def estimate_convergence_rate(
    lambda_val: float,
    lambda_c: float,
    beta: float = 0.5
) -> float:
    """
    収束率を推定
    
    |μ₂| ≈ exp(-β(λ - λ_c)) for λ > λ_c
    
    Args:
        lambda_val: 共感強度
        lambda_c: 臨界値
        beta: 臨界指数
    
    Returns:
        推定収束率 |μ₂|
    """
    if lambda_val <= lambda_c:
        return 1.0  # 収束しない
    
    return np.exp(-beta * (lambda_val - lambda_c))


# =============================================================================
# F_mutual の最小化
# =============================================================================

def minimize_mutual_free_energy(
    self_state_init: np.ndarray,
    other_state: np.ndarray,
    lambda_empathy: float,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    F_mutual を最小化する自己状態を探索
    
    勾配降下法による最適化
    
    Args:
        self_state_init: 自己の初期状態
        other_state: 他者の状態（固定）
        lambda_empathy: 共感強度
        learning_rate: 学習率
        max_iterations: 最大反復回数
        tolerance: 収束閾値
    
    Returns:
        (最適化された自己状態, F_mutualの履歴)
    """
    self_state = self_state_init.copy()
    history = []
    
    for i in range(max_iterations):
        # 現在のF_mutual
        result = mutual_free_energy(self_state, other_state, lambda_empathy)
        history.append(result.F_mutual)
        
        # 収束チェック
        if i > 0 and abs(history[-1] - history[-2]) < tolerance:
            break
        
        # 勾配近似（数値微分）
        grad = np.zeros_like(self_state)
        eps = 1e-5
        
        for j in range(len(self_state)):
            self_state_plus = self_state.copy()
            self_state_plus[j] += eps
            
            self_state_minus = self_state.copy()
            self_state_minus[j] -= eps
            
            F_plus = mutual_free_energy(self_state_plus, other_state, lambda_empathy).F_mutual
            F_minus = mutual_free_energy(self_state_minus, other_state, lambda_empathy).F_mutual
            
            grad[j] = (F_plus - F_minus) / (2 * eps)
        
        # 更新
        self_state = self_state - learning_rate * grad
        
        # 正規化（オプション）
        self_state = self_state / (np.linalg.norm(self_state) + 1e-10)
    
    return self_state, history


# =============================================================================
# デモ・テスト
# =============================================================================

def demo():
    """相互自由エネルギーのデモンストレーション"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第14章")
    print("相互予測自由エネルギーデモ")
    print("=" * 60)
    
    # 1. 基本計算
    print("\n[1] 相互自由エネルギー計算")
    np.random.seed(42)
    
    self_state = np.random.randn(64)
    other_state = np.random.randn(64)
    
    # 異なるλでの計算
    for lambda_val in [0.0, 0.5, 1.0, 2.0]:
        result = mutual_free_energy(self_state, other_state, lambda_val)
        print(f"    λ={lambda_val:.1f}: F_mutual={result.F_mutual:.3f} "
              f"(共感項={result.empathy_term:.3f})")
    
    # 2. 相転移判定
    print("\n[2] 協力相転移判定")
    
    test_cases = [
        {"lambda": 0.2, "PFH": 0.1, "epsilon": 0.1, "tau": 0.3},
        {"lambda": 0.5, "PFH": 0.5, "epsilon": 0.1, "tau": 0.3},
        {"lambda": 1.0, "PFH": 0.5, "epsilon": 0.1, "tau": 0.3},
        {"lambda": 0.3, "PFH": 0.2, "epsilon": 0.5, "tau": 0.5},
    ]
    
    for case in test_cases:
        result = check_cooperative_transition(
            case["lambda"], case["PFH"], case["epsilon"], case["tau"]
        )
        print(f"    λ={case['lambda']:.1f}, PFH={case['PFH']:.1f}: "
              f"{result.phase} (λ_c={result.lambda_c:.3f}, margin={result.margin:+.3f})")
    
    # 3. 生成パラメータ調整
    print("\n[3] LLM生成パラメータ調整")
    
    for phase in ["cooperative", "competitive"]:
        params = adjust_generation_params(phase)
        print(f"    {phase}相:")
        print(f"        temperature={params['temperature']}, top_p={params['top_p']}")
        print(f"        → {params['description']}")
    
    # 4. 相互コヒーレンス
    print("\n[4] 相互コヒーレンス")
    
    # 類似した状態
    state_a = np.array([1.0, 0.5, 0.3, 0.1])
    state_b = np.array([0.9, 0.6, 0.4, 0.2])
    coherence_high = compute_mutual_coherence(state_a, state_b)
    
    # 異なる状態
    state_c = np.array([1.0, 0.0, 0.0, 0.0])
    state_d = np.array([0.0, 0.0, 0.0, 1.0])
    coherence_low = compute_mutual_coherence(state_c, state_d)
    
    print(f"    類似状態: C_mutual = {coherence_high:.3f}")
    print(f"    異なる状態: C_mutual = {coherence_low:.3f}")
    
    # 5. 収束率推定
    print("\n[5] 収束率推定")
    
    lambda_c = 0.5
    for lambda_val in [0.3, 0.5, 0.7, 1.0, 1.5]:
        rate = estimate_convergence_rate(lambda_val, lambda_c)
        status = "収束" if rate < 1.0 else "非収束"
        print(f"    λ={lambda_val:.1f}: |μ₂|={rate:.3f} ({status})")
    
    # 6. F_mutual最小化
    print("\n[6] F_mutual最小化")
    
    self_init = np.random.randn(16)
    other_fixed = np.random.randn(16)
    
    optimal_state, history = minimize_mutual_free_energy(
        self_init, other_fixed, lambda_empathy=1.0, max_iterations=50
    )
    
    print(f"    初期 F_mutual: {history[0]:.3f}")
    print(f"    最終 F_mutual: {history[-1]:.3f}")
    print(f"    反復回数: {len(history)}")
    print(f"    削減率: {(history[0] - history[-1]) / abs(history[0]) * 100:.1f}%")
    
    # 7. 対話コスト関数
    print("\n[7] 対話コスト関数")
    
    cost = dialogue_cost_function(
        self_embedding=np.random.randn(128),
        other_embedding=np.random.randn(128),
        self_perplexity=15.0,
        other_perplexity=20.0,
        lambda_empathy=0.8
    )
    
    print(f"    言語コスト(自己): {cost['language_cost_self']:.3f}")
    print(f"    言語コスト(他者): {cost['language_cost_other']:.3f}")
    print(f"    共感スコア: {cost['empathy_score']:.3f}")
    print(f"    相互コスト: {cost['mutual_cost']:.3f}")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
