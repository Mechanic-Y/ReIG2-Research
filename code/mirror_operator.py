"""
ReIG2/twinRIG 第14章（発展編）
共感演算子の古典実装
Mirror Operator - Classical Implementation

Mechanic-Y / Yasuyuki Wakita
2025年12月

量子概念 M̂（共感演算子）の古典コンピュータ（LLM）への実装
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
# 量子版：共感演算子（参照用）
# =============================================================================

# SWAP演算子（2量子ビット系での共感演算子）
MIRROR_OPERATOR_QUANTUM = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)


def quantum_mirror_operator(psi_self: np.ndarray, phi_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    量子版：共感演算子の適用
    
    M̂ |ψ_self⟩ ⊗ |φ_other⟩ = |φ_self⟩ ⊗ |ψ_other⟩
    
    Args:
        psi_self: 自己の状態ベクトル (2次元)
        phi_other: 他者の状態ベクトル (2次元)
    
    Returns:
        (phi_self, psi_other): 視点交換後の状態
    """
    # テンソル積状態を構築
    psi_combined = np.kron(psi_self, phi_other)
    
    # 共感演算子を適用
    result = MIRROR_OPERATOR_QUANTUM @ psi_combined
    
    # 結果を分離（2x2系の場合）
    # |00⟩, |01⟩, |10⟩, |11⟩ の係数から復元
    phi_self = np.array([result[0], result[2]])  # |0⟩, |1⟩ for self
    psi_other = np.array([result[0], result[1]])  # |0⟩, |1⟩ for other
    
    return phi_self, psi_other


# =============================================================================
# 古典版：Cross-Attention による共感演算子
# =============================================================================

class MirrorOperatorClassical:
    """
    共感演算子の古典実装クラス
    
    Transformerアーキテクチャにおける Cross-Attention を用いて
    「他者の視点に立つ」操作を実現する
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 1):
        """
        Args:
            embedding_dim: 埋め込み次元
            num_heads: アテンションヘッド数
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # 重み行列の初期化（Xavier初期化）
        scale = np.sqrt(2.0 / (embedding_dim + self.head_dim))
        self.W_Q = np.random.randn(embedding_dim, embedding_dim) * scale
        self.W_K = np.random.randn(embedding_dim, embedding_dim) * scale
        self.W_V = np.random.randn(embedding_dim, embedding_dim) * scale
        self.W_O = np.random.randn(embedding_dim, embedding_dim) * scale
    
    def __call__(
        self, 
        self_embedding: np.ndarray, 
        other_embedding: np.ndarray,
        empathy_barrier: float = 0.0
    ) -> np.ndarray:
        """
        共感演算子の適用：自己が他者の視点を取得
        
        Args:
            self_embedding: 自己の埋め込みベクトル [seq_len, embedding_dim] or [embedding_dim]
            other_embedding: 他者の埋め込みベクトル [seq_len, embedding_dim] or [embedding_dim]
            empathy_barrier: 共感障壁パラメータ δ (0 ≤ δ ≤ 1)
        
        Returns:
            self_as_other: 他者視点から見た自己の埋め込み
        """
        # 1次元の場合は2次元に拡張
        if self_embedding.ndim == 1:
            self_embedding = self_embedding.reshape(1, -1)
        if other_embedding.ndim == 1:
            other_embedding = other_embedding.reshape(1, -1)
        
        # Query, Key, Value の計算
        Q = self_embedding @ self.W_Q  # 自己の視点（質問）
        K = other_embedding @ self.W_K  # 他者の視点（キー）
        V = other_embedding @ self.W_V  # 他者の内容（値）
        
        # スケーリングドットプロダクトアテンション
        d_k = np.sqrt(self.head_dim)
        attention_scores = Q @ K.T / d_k
        
        # 共感障壁の適用（アテンションスコアを減衰）
        attention_scores = attention_scores * (1 - empathy_barrier)
        
        # ソフトマックス
        attention_weights = self._softmax(attention_scores)
        
        # 他者の視点を投影
        self_as_other = attention_weights @ V
        
        # 出力投影
        output = self_as_other @ self.W_O
        
        return output.squeeze()
    
    def bidirectional_exchange(
        self,
        self_embedding: np.ndarray,
        other_embedding: np.ndarray,
        empathy_barrier_self: float = 0.0,
        empathy_barrier_other: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        双方向の視点交換
        
        Args:
            self_embedding: 自己の埋め込み
            other_embedding: 他者の埋め込み
            empathy_barrier_self: 自己→他者の共感障壁
            empathy_barrier_other: 他者→自己の共感障壁
        
        Returns:
            (self_as_other, other_as_self): 相互の視点交換結果
        """
        self_as_other = self(self_embedding, other_embedding, empathy_barrier_self)
        other_as_self = self(other_embedding, self_embedding, empathy_barrier_other)
        
        return self_as_other, other_as_self
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数値安定性を考慮したソフトマックス"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =============================================================================
# 古典版：プロンプトベースの共感演算子
# =============================================================================

class PerspectiveTakingPrompt:
    """
    プロンプトエンジニアリングによる共感の実装
    
    LLMに「他者の視点に立つ」よう指示するプロンプトテンプレート
    """
    
    TEMPLATE_JA = """
あなたは {other_role} の立場に立って考えてください。
{other_context} という状況にある人が、
{situation} についてどう感じるか、
その視点から回答してください。

考慮すべき点：
- その人の背景や経験
- その人が持つ可能性のある懸念や希望
- 文化的・社会的な文脈

回答：
"""
    
    TEMPLATE_EN = """
Please think from the perspective of {other_role}.
Consider how someone in the situation of {other_context}
would feel about {situation}.
Respond from their viewpoint.

Consider:
- Their background and experiences
- Their possible concerns and hopes
- Cultural and social context

Response:
"""
    
    def __init__(self, language: str = "ja"):
        """
        Args:
            language: "ja" (日本語) or "en" (英語)
        """
        self.template = self.TEMPLATE_JA if language == "ja" else self.TEMPLATE_EN
    
    def generate_prompt(
        self,
        other_role: str,
        other_context: str,
        situation: str
    ) -> str:
        """
        共感プロンプトを生成
        
        Args:
            other_role: 他者の役割（例: "経済的困難を抱える親"）
            other_context: 他者の状況（例: "失業後3ヶ月経過"）
            situation: 議論の対象（例: "政府の支援策について"）
        
        Returns:
            LLMへの入力プロンプト
        """
        return self.template.format(
            other_role=other_role,
            other_context=other_context,
            situation=situation
        )


# =============================================================================
# 古典版：埋め込み空間での共感演算
# =============================================================================

def mirror_operator_embedding(
    self_embedding: np.ndarray,
    other_embedding: np.ndarray,
    mixing_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    埋め込み空間での単純な視点混合
    
    Args:
        self_embedding: 自己の埋め込みベクトル
        other_embedding: 他者の埋め込みベクトル
        mixing_ratio: 混合比率 (0: 自己のみ, 1: 他者のみ)
    
    Returns:
        (self_mixed, other_mixed): 混合後の埋め込み
    """
    self_mixed = (1 - mixing_ratio) * self_embedding + mixing_ratio * other_embedding
    other_mixed = mixing_ratio * self_embedding + (1 - mixing_ratio) * other_embedding
    
    return self_mixed, other_mixed


def compute_empathy_score(
    self_embedding: np.ndarray,
    other_embedding: np.ndarray
) -> float:
    """
    共感スコア（視点の近さ）を計算
    
    ⟨M̂⟩ に対応する古典的な指標
    
    Args:
        self_embedding: 自己の埋め込み
        other_embedding: 他者の埋め込み
    
    Returns:
        共感スコア [0, 1]
    """
    # コサイン類似度
    dot_product = np.dot(self_embedding.flatten(), other_embedding.flatten())
    norm_self = np.linalg.norm(self_embedding)
    norm_other = np.linalg.norm(other_embedding)
    
    if norm_self == 0 or norm_other == 0:
        return 0.0
    
    cosine_similarity = dot_product / (norm_self * norm_other)
    
    # [−1, 1] → [0, 1] に変換
    empathy_score = (cosine_similarity + 1) / 2
    
    return float(empathy_score)


def compute_empathy_barrier(
    cultural_distance: float = 0.0,
    religious_distance: float = 0.0,
    economic_distance: float = 0.0,
    linguistic_distance: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    共感障壁パラメータ δ を計算
    
    Args:
        cultural_distance: 文化的距離 [0, 1]
        religious_distance: 宗教的距離 [0, 1]
        economic_distance: 経済的距離 [0, 1]
        linguistic_distance: 言語的距離 [0, 1]
        weights: 各要素の重み（デフォルト: 均等）
    
    Returns:
        共感障壁 δ [0, 1]
    """
    if weights is None:
        weights = {
            "cultural": 0.25,
            "religious": 0.25,
            "economic": 0.25,
            "linguistic": 0.25
        }
    
    delta = (
        weights.get("cultural", 0.25) * cultural_distance +
        weights.get("religious", 0.25) * religious_distance +
        weights.get("economic", 0.25) * economic_distance +
        weights.get("linguistic", 0.25) * linguistic_distance
    )
    
    return min(1.0, max(0.0, delta))


# =============================================================================
# デモ・テスト
# =============================================================================

def demo():
    """共感演算子のデモンストレーション"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第14章")
    print("共感演算子の古典実装デモ")
    print("=" * 60)
    
    # 1. 量子版のデモ
    print("\n[1] 量子版：M̂ 演算子")
    psi_self = np.array([1, 0], dtype=complex)  # |0⟩
    phi_other = np.array([0, 1], dtype=complex)  # |1⟩
    
    print(f"    自己の状態: |0⟩ = {psi_self}")
    print(f"    他者の状態: |1⟩ = {phi_other}")
    
    # SWAP演算子の期待値
    combined = np.kron(psi_self, phi_other)
    M_expectation = np.real(combined.conj() @ MIRROR_OPERATOR_QUANTUM @ combined)
    print(f"    ⟨M̂⟩ = {M_expectation:.3f}")
    
    # 2. Cross-Attention版のデモ
    print("\n[2] 古典版：Cross-Attention")
    np.random.seed(42)
    embedding_dim = 64
    
    mirror_op = MirrorOperatorClassical(embedding_dim)
    
    self_emb = np.random.randn(embedding_dim)
    other_emb = np.random.randn(embedding_dim)
    
    # 共感障壁なし
    result_no_barrier = mirror_op(self_emb, other_emb, empathy_barrier=0.0)
    
    # 共感障壁あり
    result_with_barrier = mirror_op(self_emb, other_emb, empathy_barrier=0.5)
    
    print(f"    埋め込み次元: {embedding_dim}")
    print(f"    共感障壁なし: 出力ノルム = {np.linalg.norm(result_no_barrier):.3f}")
    print(f"    共感障壁0.5: 出力ノルム = {np.linalg.norm(result_with_barrier):.3f}")
    
    # 3. 共感スコアのデモ
    print("\n[3] 共感スコア計算")
    
    # 類似した埋め込み
    emb_a = np.array([1.0, 0.5, 0.3])
    emb_b = np.array([0.9, 0.6, 0.4])
    score_similar = compute_empathy_score(emb_a, emb_b)
    
    # 異なる埋め込み
    emb_c = np.array([1.0, 0.0, 0.0])
    emb_d = np.array([0.0, 1.0, 0.0])
    score_different = compute_empathy_score(emb_c, emb_d)
    
    print(f"    類似ベクトル: スコア = {score_similar:.3f}")
    print(f"    直交ベクトル: スコア = {score_different:.3f}")
    
    # 4. 共感障壁計算のデモ
    print("\n[4] 共感障壁パラメータ δ")
    
    delta_low = compute_empathy_barrier(
        cultural_distance=0.1,
        religious_distance=0.1,
        economic_distance=0.2,
        linguistic_distance=0.1
    )
    
    delta_high = compute_empathy_barrier(
        cultural_distance=0.8,
        religious_distance=0.7,
        economic_distance=0.9,
        linguistic_distance=0.6
    )
    
    print(f"    低障壁シナリオ: δ = {delta_low:.3f}")
    print(f"    高障壁シナリオ: δ = {delta_high:.3f}")
    
    # 5. プロンプトテンプレートのデモ
    print("\n[5] 共感プロンプト生成")
    prompt_gen = PerspectiveTakingPrompt(language="ja")
    
    prompt = prompt_gen.generate_prompt(
        other_role="経済的困難を抱える親",
        other_context="失業後3ヶ月が経過し、子供の教育費に悩んでいる",
        situation="政府の教育支援策"
    )
    
    print("    生成されたプロンプト:")
    print("-" * 40)
    print(prompt[:200] + "...")
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo()
