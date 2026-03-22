"""
ReIG2/twinRIG World-Building Quantum Cosmology: Mirror Operator Module
======================================================================
共感演算子 M_empathy (Mirror Operator) のLLM実装

対応セクション: §14.2
- 共感演算子の定義と特性
- Cross-Attentionベースの実装
- 量子版とクラシカル版の統一インターフェース

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Theory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
Embedding = NDArray[np.float64]


# =============================================================================
# Mirror Operator Types
# =============================================================================
class MirrorOperatorType(Enum):
    """Mirror Operator実装タイプ"""
    QUANTUM = auto()           # 量子版（Qiskit互換）
    CLASSICAL = auto()         # クラシカル版（Cross-Attention）
    HYBRID = auto()            # ハイブリッド版


class EmpathyMode(Enum):
    """共感モード"""
    COGNITIVE = auto()         # 認知的共感（理解）
    AFFECTIVE = auto()         # 情動的共感（感情共有）
    COMPASSIONATE = auto()     # 思いやり共感（支援志向）
    INTEGRATED = auto()        # 統合的共感


# =============================================================================
# Abstract Mirror Operator
# =============================================================================
class BaseMirrorOperator(ABC):
    """
    Mirror Operator (共感演算子) の抽象基底クラス
    
    §14.2: M_empathy は自己状態|ψ_self⟩と他者状態|ψ_other⟩の
    間の相互作用を記述し、共感的理解を実現する演算子
    """
    
    @abstractmethod
    def apply(self, 
             self_state: Any, 
             other_state: Any) -> Any:
        """
        共感演算子の適用
        
        M_empathy |ψ_self⟩ ⊗ |ψ_other⟩ → |ψ_empathic⟩
        """
        pass
    
    @abstractmethod
    def empathy_score(self,
                     self_state: Any,
                     other_state: Any) -> float:
        """
        共感スコアの計算
        
        E(self, other) = ⟨ψ_empathic | M_empathy | ψ_self ⊗ ψ_other⟩
        """
        pass


# =============================================================================
# Quantum Mirror Operator
# =============================================================================
class QuantumMirrorOperator(BaseMirrorOperator):
    """
    量子版 Mirror Operator
    
    §14.2: 密度行列形式での共感演算子
    
    M_empathy(ρ_self, ρ_other) = 
        α Tr_other(U_int (ρ_self ⊗ ρ_other) U_int†) + 
        β ρ_self + 
        γ Tr_self(ρ_self) ⊗ ρ_other
    
    where U_int is the interaction unitary
    """
    
    def __init__(self,
                 dim_self: int = 2,
                 dim_other: int = 2,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 coupling: float = 0.1):
        self.dim_self = dim_self
        self.dim_other = dim_other
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.coupling = coupling
        
        # 正規化
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        # 相互作用ユニタリの構築
        self._U_int = self._build_interaction_unitary()
    
    def _build_interaction_unitary(self) -> ComplexMatrix:
        """
        相互作用ユニタリ U_int の構築
        
        U_int = exp(-i λ H_int) where H_int = σ_x ⊗ σ_x + σ_y ⊗ σ_y
        """
        dim_total = self.dim_self * self.dim_other
        
        if self.dim_self == 2 and self.dim_other == 2:
            # 2量子ビット系の具体的構成
            sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            
            H_int = (np.kron(sigma_x, sigma_x) + 
                    np.kron(sigma_y, sigma_y))
            
            U_int = self._matrix_exp(-1j * self.coupling * H_int)
        else:
            # 一般次元：ランダム相互作用
            H_int = np.random.randn(dim_total, dim_total) + \
                   1j * np.random.randn(dim_total, dim_total)
            H_int = 0.5 * (H_int + H_int.conj().T)  # エルミート化
            U_int = self._matrix_exp(-1j * self.coupling * H_int)
        
        return U_int
    
    def _matrix_exp(self, A: ComplexMatrix) -> ComplexMatrix:
        """行列指数関数"""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.conj().T
    
    def apply(self,
             rho_self: ComplexMatrix,
             rho_other: ComplexMatrix) -> ComplexMatrix:
        """
        量子共感演算子の適用
        
        Args:
            rho_self: 自己系の密度行列
            rho_other: 他者系の密度行列
        
        Returns:
            共感的状態の密度行列
        """
        # テンソル積
        rho_total = np.kron(rho_self, rho_other)
        
        # 相互作用適用
        rho_interacted = self._U_int @ rho_total @ self._U_int.conj().T
        
        # 部分トレース（他者系をトレースアウト）
        rho_reduced = self._partial_trace(
            rho_interacted, 
            keep='A',
            dim_A=self.dim_self,
            dim_B=self.dim_other
        )
        
        # 混合
        rho_empathic = (
            self.alpha * rho_reduced +
            self.beta * rho_self +
            self.gamma * np.trace(rho_self) * rho_other[:self.dim_self, :self.dim_self]
        )
        
        # 正規化
        trace = np.trace(rho_empathic)
        if np.abs(trace) > 1e-10:
            rho_empathic /= trace
        
        return rho_empathic
    
    def _partial_trace(self,
                      rho: ComplexMatrix,
                      keep: str,
                      dim_A: int,
                      dim_B: int) -> ComplexMatrix:
        """部分トレース"""
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        
        if keep == 'A':
            return np.trace(rho_reshaped, axis1=1, axis2=3)
        else:
            return np.trace(rho_reshaped, axis1=0, axis2=2)
    
    def empathy_score(self,
                     rho_self: ComplexMatrix,
                     rho_other: ComplexMatrix) -> float:
        """
        量子共感スコア
        
        フィデリティベースの評価
        """
        rho_empathic = self.apply(rho_self, rho_other)
        
        # 共感状態と他者状態のフィデリティ
        # F = Tr(√(√ρ_empathic ρ_other √ρ_empathic))²
        
        # 簡略版：トレース距離ベース
        diff = rho_empathic - rho_other[:self.dim_self, :self.dim_self]
        trace_dist = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff)))
        
        return float(1.0 - trace_dist)


# =============================================================================
# Classical Mirror Operator (Cross-Attention)
# =============================================================================
class CrossAttentionMirrorOperator(BaseMirrorOperator):
    """
    クラシカル版 Mirror Operator (Cross-Attention実装)
    
    §14.2: LLMにおける共感演算子の実装
    
    M_empathy(h_self, h_other) = softmax(Q_self K_other^T / √d) V_other
    
    where Q_self = W_Q h_self, K_other = W_K h_other, V_other = W_V h_other
    """
    
    def __init__(self,
                 dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 mode: EmpathyMode = EmpathyMode.INTEGRATED):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.mode = mode
        
        # 射影行列の初期化
        self._init_projections()
    
    def _init_projections(self) -> None:
        """射影行列の初期化"""
        scale = np.sqrt(2.0 / self.dim)
        
        # Query, Key, Value 射影
        self.W_Q = scale * np.random.randn(self.dim, self.dim)
        self.W_K = scale * np.random.randn(self.dim, self.dim)
        self.W_V = scale * np.random.randn(self.dim, self.dim)
        
        # 出力射影
        self.W_O = scale * np.random.randn(self.dim, self.dim)
        
        # モード別の追加射影
        if self.mode == EmpathyMode.COGNITIVE:
            self.W_cognitive = scale * np.random.randn(self.dim, self.dim)
        elif self.mode == EmpathyMode.AFFECTIVE:
            self.W_affective = scale * np.random.randn(self.dim, self.dim)
        elif self.mode == EmpathyMode.COMPASSIONATE:
            self.W_compassion = scale * np.random.randn(self.dim, self.dim)
    
    def apply(self,
             h_self: Embedding,
             h_other: Embedding) -> Embedding:
        """
        Cross-Attention共感演算子の適用
        
        Args:
            h_self: 自己の埋め込み表現 (dim,) or (seq_len, dim)
            h_other: 他者の埋め込み表現 (dim,) or (seq_len, dim)
        
        Returns:
            共感的表現 (dim,) or (seq_len, dim)
        """
        # 次元の正規化
        h_self = np.atleast_2d(h_self)
        h_other = np.atleast_2d(h_other)
        
        # Query, Key, Value の計算
        Q = h_self @ self.W_Q  # (seq_self, dim)
        K = h_other @ self.W_K  # (seq_other, dim)
        V = h_other @ self.W_V  # (seq_other, dim)
        
        # マルチヘッドに分割
        Q_heads = self._split_heads(Q)  # (num_heads, seq_self, head_dim)
        K_heads = self._split_heads(K)  # (num_heads, seq_other, head_dim)
        V_heads = self._split_heads(V)  # (num_heads, seq_other, head_dim)
        
        # スケールドドット積アテンション
        scale = np.sqrt(self.head_dim)
        attention_scores = np.einsum('hij,hkj->hik', Q_heads, K_heads) / scale
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # ドロップアウト（推論時はスキップ）
        
        # アテンション適用
        attended = np.einsum('hik,hkj->hij', attention_weights, V_heads)
        
        # ヘッドの結合
        concat = self._merge_heads(attended)  # (seq_self, dim)
        
        # 出力射影
        output = concat @ self.W_O
        
        # モード別の追加処理
        output = self._apply_mode_specific(output, h_self, h_other)
        
        return output.squeeze()
    
    def _split_heads(self, x: RealMatrix) -> RealMatrix:
        """マルチヘッドに分割"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
    
    def _merge_heads(self, x: RealMatrix) -> RealMatrix:
        """ヘッドを結合"""
        x = x.transpose(1, 0, 2)  # (seq_len, num_heads, head_dim)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.dim)
    
    def _softmax(self, x: RealMatrix, axis: int = -1) -> RealMatrix:
        """数値安定なソフトマックス"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _apply_mode_specific(self,
                            output: RealMatrix,
                            h_self: RealMatrix,
                            h_other: RealMatrix) -> RealMatrix:
        """モード別の追加処理"""
        if self.mode == EmpathyMode.COGNITIVE:
            # 認知的共感：理解の強化
            cognitive_gate = self._sigmoid(h_self @ self.W_cognitive)
            output = output * cognitive_gate
        
        elif self.mode == EmpathyMode.AFFECTIVE:
            # 情動的共感：感情の混合
            affective_blend = 0.7 * output + 0.3 * (h_other @ self.W_affective)
            output = affective_blend
        
        elif self.mode == EmpathyMode.COMPASSIONATE:
            # 思いやり共感：支援志向の強化
            compassion_boost = h_self @ self.W_compassion
            output = output + 0.2 * self._relu(compassion_boost)
        
        elif self.mode == EmpathyMode.INTEGRATED:
            # 統合的共感：残差接続
            output = output + 0.1 * h_self
        
        return output
    
    def _sigmoid(self, x: RealMatrix) -> RealMatrix:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x: RealMatrix) -> RealMatrix:
        return np.maximum(0, x)
    
    def empathy_score(self,
                     h_self: Embedding,
                     h_other: Embedding) -> float:
        """
        共感スコアの計算
        
        共感的表現と他者表現のコサイン類似度
        """
        h_empathic = self.apply(h_self, h_other)
        
        h_empathic = h_empathic.flatten()
        h_other = np.atleast_2d(h_other).mean(axis=0)  # 平均プーリング
        
        norm_empathic = np.linalg.norm(h_empathic)
        norm_other = np.linalg.norm(h_other)
        
        if norm_empathic < 1e-10 or norm_other < 1e-10:
            return 0.0
        
        return float(np.dot(h_empathic, h_other) / (norm_empathic * norm_other))


# =============================================================================
# Empathy State
# =============================================================================
@dataclass
class EmpathyState:
    """
    共感状態
    
    共感演算子適用後の状態を保持
    """
    empathic_representation: Union[ComplexMatrix, Embedding]
    empathy_score: float
    mode: EmpathyMode
    self_contribution: float = 0.0
    other_contribution: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def balance(self) -> float:
        """自己/他者バランス"""
        total = self.self_contribution + self.other_contribution
        if total < 1e-10:
            return 0.5
        return self.other_contribution / total


# =============================================================================
# Empathy Processor
# =============================================================================
class EmpathyProcessor:
    """
    共感処理器
    
    複数の共感モードを統合し、コンテキストに応じた
    共感的応答を生成
    """
    
    def __init__(self,
                 dim: int = 128,
                 use_quantum: bool = False):
        self.dim = dim
        self.use_quantum = use_quantum
        
        # 各モードの演算子
        self.operators = {
            EmpathyMode.COGNITIVE: CrossAttentionMirrorOperator(
                dim, mode=EmpathyMode.COGNITIVE
            ),
            EmpathyMode.AFFECTIVE: CrossAttentionMirrorOperator(
                dim, mode=EmpathyMode.AFFECTIVE
            ),
            EmpathyMode.COMPASSIONATE: CrossAttentionMirrorOperator(
                dim, mode=EmpathyMode.COMPASSIONATE
            ),
            EmpathyMode.INTEGRATED: CrossAttentionMirrorOperator(
                dim, mode=EmpathyMode.INTEGRATED
            ),
        }
        
        # 量子版（オプション）
        if use_quantum:
            self.quantum_operator = QuantumMirrorOperator()
    
    def process(self,
               h_self: Embedding,
               h_other: Embedding,
               mode: Optional[EmpathyMode] = None,
               context: Optional[Dict[str, Any]] = None
               ) -> EmpathyState:
        """
        共感処理の実行
        
        Args:
            h_self: 自己表現
            h_other: 他者表現
            mode: 共感モード（Noneで自動選択）
            context: 追加コンテキスト
        
        Returns:
            EmpathyState
        """
        # モード選択
        if mode is None:
            mode = self._auto_select_mode(h_self, h_other, context)
        
        # 演算子の取得と適用
        operator = self.operators[mode]
        h_empathic = operator.apply(h_self, h_other)
        score = operator.empathy_score(h_self, h_other)
        
        # 寄与度の計算
        self_contrib = self._compute_contribution(h_empathic, h_self)
        other_contrib = self._compute_contribution(h_empathic, h_other)
        
        return EmpathyState(
            empathic_representation=h_empathic,
            empathy_score=score,
            mode=mode,
            self_contribution=self_contrib,
            other_contribution=other_contrib,
            metadata={
                'context': context,
                'operator_type': 'cross_attention'
            }
        )
    
    def _auto_select_mode(self,
                         h_self: Embedding,
                         h_other: Embedding,
                         context: Optional[Dict[str, Any]] = None
                         ) -> EmpathyMode:
        """共感モードの自動選択"""
        if context is None:
            return EmpathyMode.INTEGRATED
        
        # コンテキストに基づく選択
        if context.get('emotional_intensity', 0) > 0.7:
            return EmpathyMode.AFFECTIVE
        elif context.get('needs_support', False):
            return EmpathyMode.COMPASSIONATE
        elif context.get('requires_understanding', False):
            return EmpathyMode.COGNITIVE
        
        return EmpathyMode.INTEGRATED
    
    def _compute_contribution(self,
                             h_empathic: Embedding,
                             h_source: Embedding) -> float:
        """寄与度の計算"""
        h_empathic = np.atleast_2d(h_empathic).mean(axis=0)
        h_source = np.atleast_2d(h_source).mean(axis=0)
        
        norm_emp = np.linalg.norm(h_empathic)
        norm_src = np.linalg.norm(h_source)
        
        if norm_emp < 1e-10 or norm_src < 1e-10:
            return 0.0
        
        return float(np.abs(np.dot(h_empathic, h_source)) / (norm_emp * norm_src))
    
    def blend_modes(self,
                   h_self: Embedding,
                   h_other: Embedding,
                   weights: Optional[Dict[EmpathyMode, float]] = None
                   ) -> EmpathyState:
        """
        複数モードの混合
        """
        if weights is None:
            weights = {
                EmpathyMode.COGNITIVE: 0.3,
                EmpathyMode.AFFECTIVE: 0.3,
                EmpathyMode.COMPASSIONATE: 0.2,
                EmpathyMode.INTEGRATED: 0.2,
            }
        
        # 各モードの出力を計算
        blended = np.zeros(self.dim)
        total_score = 0.0
        total_weight = 0.0
        
        for mode, weight in weights.items():
            state = self.process(h_self, h_other, mode)
            blended += weight * np.atleast_2d(state.empathic_representation).mean(axis=0)
            total_score += weight * state.empathy_score
            total_weight += weight
        
        if total_weight > 0:
            blended /= total_weight
            total_score /= total_weight
        
        return EmpathyState(
            empathic_representation=blended,
            empathy_score=total_score,
            mode=EmpathyMode.INTEGRATED,
            metadata={'blended': True, 'weights': weights}
        )


# =============================================================================
# Mirror Operator Factory
# =============================================================================
class MirrorOperatorFactory:
    """
    Mirror Operatorのファクトリ
    """
    
    @staticmethod
    def create(operator_type: MirrorOperatorType,
              **kwargs) -> BaseMirrorOperator:
        """
        Mirror Operatorの生成
        
        Args:
            operator_type: 実装タイプ
            **kwargs: 追加パラメータ
        
        Returns:
            BaseMirrorOperator
        """
        if operator_type == MirrorOperatorType.QUANTUM:
            return QuantumMirrorOperator(**kwargs)
        elif operator_type == MirrorOperatorType.CLASSICAL:
            return CrossAttentionMirrorOperator(**kwargs)
        elif operator_type == MirrorOperatorType.HYBRID:
            # ハイブリッドはクラシカルベースで量子的特性を模倣
            return CrossAttentionMirrorOperator(**kwargs)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")


# =============================================================================
# Empathy Metrics
# =============================================================================
class EmpathyMetrics:
    """
    共感メトリクス
    
    共感の質を評価する各種指標
    """
    
    @staticmethod
    def perspective_taking_score(h_self: Embedding,
                                h_other: Embedding,
                                h_empathic: Embedding) -> float:
        """
        視点取得スコア
        
        自己視点から他者視点への移行度
        """
        h_self = np.atleast_2d(h_self).mean(axis=0)
        h_other = np.atleast_2d(h_other).mean(axis=0)
        h_empathic = np.atleast_2d(h_empathic).mean(axis=0)
        
        # 自己から共感への距離
        dist_self_emp = np.linalg.norm(h_empathic - h_self)
        # 自己から他者への距離
        dist_self_other = np.linalg.norm(h_other - h_self)
        
        if dist_self_other < 1e-10:
            return 1.0
        
        # 視点移行の程度
        return float(min(1.0, dist_self_emp / dist_self_other))
    
    @staticmethod
    def emotional_resonance(h_empathic: Embedding,
                           h_other: Embedding) -> float:
        """
        感情的共鳴度
        
        共感表現と他者表現の類似度
        """
        h_empathic = np.atleast_2d(h_empathic).mean(axis=0)
        h_other = np.atleast_2d(h_other).mean(axis=0)
        
        norm_emp = np.linalg.norm(h_empathic)
        norm_other = np.linalg.norm(h_other)
        
        if norm_emp < 1e-10 or norm_other < 1e-10:
            return 0.0
        
        return float(np.dot(h_empathic, h_other) / (norm_emp * norm_other))
    
    @staticmethod
    def self_other_balance(empathy_state: EmpathyState) -> float:
        """
        自己-他者バランス
        
        0=自己中心、0.5=バランス、1=他者中心
        """
        return empathy_state.balance


# =============================================================================
# Demo Function
# =============================================================================
def demo_mirror_operator():
    """Mirror Operatorのデモ"""
    print("=" * 60)
    print("ReIG2/twinRIG Mirror Operator (Empathy Operator) Demo")
    print("=" * 60)
    
    dim = 64
    
    # 1. Cross-Attention版のデモ
    print("\n--- Cross-Attention Mirror Operator ---")
    
    ca_operator = CrossAttentionMirrorOperator(
        dim=dim,
        num_heads=4,
        mode=EmpathyMode.INTEGRATED
    )
    
    # ダミー埋め込み
    h_self = np.random.randn(dim)
    h_other = np.random.randn(dim)
    
    h_empathic = ca_operator.apply(h_self, h_other)
    score = ca_operator.empathy_score(h_self, h_other)
    
    print(f"入力次元: {dim}")
    print(f"共感スコア: {score:.4f}")
    print(f"出力形状: {h_empathic.shape}")
    
    # 2. 量子版のデモ
    print("\n--- Quantum Mirror Operator ---")
    
    q_operator = QuantumMirrorOperator(
        dim_self=2,
        dim_other=2,
        coupling=0.2
    )
    
    # 密度行列
    rho_self = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0⟩
    rho_other = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)  # |+⟩
    
    rho_empathic = q_operator.apply(rho_self, rho_other)
    q_score = q_operator.empathy_score(rho_self, rho_other)
    
    print(f"量子共感スコア: {q_score:.4f}")
    print(f"共感状態トレース: {np.trace(rho_empathic):.4f}")
    
    # 3. EmpathyProcessorのデモ
    print("\n--- Empathy Processor (Multi-mode) ---")
    
    processor = EmpathyProcessor(dim=dim)
    
    # 各モードでの処理
    for mode in EmpathyMode:
        state = processor.process(h_self, h_other, mode=mode)
        print(f"{mode.name:15s}: score={state.empathy_score:.4f}, "
              f"balance={state.balance:.4f}")
    
    # 混合モード
    blended = processor.blend_modes(h_self, h_other)
    print(f"\nBlended mode: score={blended.empathy_score:.4f}")
    
    # 4. メトリクス
    print("\n--- Empathy Metrics ---")
    
    pt_score = EmpathyMetrics.perspective_taking_score(
        h_self, h_other, h_empathic
    )
    er_score = EmpathyMetrics.emotional_resonance(h_empathic, h_other)
    
    print(f"視点取得スコア: {pt_score:.4f}")
    print(f"感情共鳴度: {er_score:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_mirror_operator()
