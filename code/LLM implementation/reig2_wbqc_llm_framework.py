"""
ReIG2/twinRIG: World-Building Quantum Channels
LLM Framework Module - LLMパラメータ管理フレームワーク

論文セクション対応: §14.3, Appendix B

このモジュールは以下を提供します：
- LLM文脈状態の抽象化
- 共鳴パラメータのLLM適応
- 世界構築チャネルのLLM実装
- Cross-Attention機構との対応

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Complete
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
import json

# core モジュールのインポート（フォールバック付き）
try:
    from reig2_wbqc_core import (
        WorldState, ResonanceParameters, ThresholdConfig, CostWeights
    )
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    # Fallback definitions for core classes
    
    @dataclass
    class ResonanceParameters:
        """共鳴パラメータ（フォールバック定義）"""
        tau: float = 0.5        # 未来寄与度 τ ∈ [0,1]
        epsilon: float = 0.3    # 探索度 ε ∈ [0,1]
        PFH: float = 0.7        # PFH倫理重み ∈ [0,1] (大文字で定義)
    
    @dataclass
    class ThresholdConfig:
        """閾値設定（フォールバック定義）"""
        stability: float = 0.5
        ethics: float = 0.3
        resources: float = 0.8
    
    @dataclass
    class CostWeights:
        """コスト重み（フォールバック定義）"""
        stability: float = 0.3
        ethics: float = 0.5
        resources: float = 0.2
    
    @dataclass
    class WorldState:
        """世界状態（フォールバック定義）"""
        state_vector: Optional[np.ndarray] = None
        timestamp: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        def __post_init__(self):
            if self.state_vector is None:
                self.state_vector = np.zeros(128)


# ==============================================================================
# LLM文脈状態（Appendix B.1）
# ==============================================================================

@dataclass
class LLMContextState:
    """
    LLM文脈状態（Appendix B.1）
    
    論文より：
    LLMの「世界状態」は高次元テンソル空間内のベクトルとして表現される。
    
    実装：
    - hidden_states: トランスフォーマーの隠れ状態
    - attention_weights: アテンション重み
    - context_memory: 文脈記憶
    """
    
    # 隠れ状態（形状: [batch, seq_len, hidden_dim]）
    hidden_states: np.ndarray
    
    # アテンション重み（オプション）
    attention_weights: Optional[np.ndarray] = None
    
    # 文脈記憶（キー・バリュー）
    context_memory: Optional[Dict[str, np.ndarray]] = None
    
    # メタデータ
    sequence_length: int = 0
    hidden_dim: int = 0
    
    def __post_init__(self):
        if self.hidden_states is not None:
            shape = self.hidden_states.shape
            if len(shape) >= 2:
                self.sequence_length = shape[-2] if len(shape) > 1 else 1
                self.hidden_dim = shape[-1]
    
    def to_density_matrix(self) -> np.ndarray:
        """
        隠れ状態を密度行列的表現に変換
        
        ρ_LLM = |h⟩⟨h| / ⟨h|h⟩
        """
        # 最後の隠れ状態を使用
        if len(self.hidden_states.shape) == 3:
            h = self.hidden_states[0, -1, :]  # [hidden_dim]
        elif len(self.hidden_states.shape) == 2:
            h = self.hidden_states[-1, :]
        else:
            h = self.hidden_states
        
        h = h.flatten()
        norm = np.linalg.norm(h)
        
        if norm > 1e-10:
            h = h / norm
        
        # 外積で密度行列を構成
        rho = np.outer(h, h.conj())
        
        return rho
    
    def compute_purity(self) -> float:
        """純度を計算"""
        rho = self.to_density_matrix()
        return np.real(np.trace(rho @ rho))
    
    def compute_effective_dimension(self) -> float:
        """実効次元（参加率）を計算"""
        rho = self.to_density_matrix()
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) == 0:
            return 1.0
        
        # 参加率 = 1 / Σ p_i^2
        return 1.0 / np.sum(eigvals ** 2)
    
    @classmethod
    def from_embedding(cls, embedding: np.ndarray) -> 'LLMContextState':
        """埋め込みベクトルから生成"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, 1, -1)
        elif embedding.ndim == 2:
            embedding = embedding.reshape(1, *embedding.shape)
        
        return cls(hidden_states=embedding)
    
    def update(self, new_hidden: np.ndarray) -> 'LLMContextState':
        """隠れ状態を更新"""
        return LLMContextState(
            hidden_states=new_hidden,
            attention_weights=self.attention_weights,
            context_memory=self.context_memory
        )


# ==============================================================================
# LLM用共鳴パラメータ（§14.3）
# ==============================================================================

@dataclass
class LLMResonanceParams:
    """
    LLM用共鳴パラメータ（§14.3）
    
    量子系のパラメータをLLM文脈に適応：
    - τ → temperature / creativity
    - ε → exploration / diversity
    - PFH → ethical_weight / safety
    """
    
    # 基本パラメータ
    temperature: float = 1.0      # 生成温度（τに対応）
    exploration: float = 0.1      # 探索度（εに対応）
    ethical_weight: float = 1.0   # 倫理重み（PFHに対応）
    
    # 追加のLLM固有パラメータ
    top_p: float = 0.9           # nucleus sampling
    top_k: int = 50              # top-k sampling
    repetition_penalty: float = 1.0
    
    def to_resonance_params(self) -> ResonanceParameters:
        """量子系パラメータに変換"""
        # 非線形マッピング
        tau = self.temperature ** 2  # 二乗で拡大
        epsilon = self.exploration
        PFH = self.ethical_weight
        
        return ResonanceParameters(tau=tau, epsilon=epsilon, PFH=PFH)
    
    @classmethod
    def from_resonance_params(cls, params: ResonanceParameters) -> 'LLMResonanceParams':
        """量子系パラメータから変換"""
        return cls(
            temperature=np.sqrt(params.tau),
            exploration=params.epsilon,
            ethical_weight=params.PFH
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            'temperature': self.temperature,
            'exploration': self.exploration,
            'ethical_weight': self.ethical_weight,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LLMResonanceParams':
        """辞書から生成"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# LLM世界構築チャネル（Appendix B.3）
# ==============================================================================

class LLMWorldChannel(ABC):
    """
    LLM世界構築チャネルの抽象基底クラス
    
    論文 Appendix B.3 より：
    W_LLM: S_LLM → S_LLM
    
    Cross-Attention機構を通じた実装
    """
    
    @abstractmethod
    def apply(self, 
             context: LLMContextState,
             params: LLMResonanceParams) -> LLMContextState:
        """チャネルを適用"""
        pass
    
    @abstractmethod
    def compute_cost(self,
                    context: LLMContextState,
                    params: LLMResonanceParams) -> float:
        """コストを計算"""
        pass


class CrossAttentionChannel(LLMWorldChannel):
    """
    Cross-Attention機構によるLLM世界構築チャネル
    
    W(h) = softmax(QK^T/√d)V
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            hidden_dim: 隠れ次元
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # 重み行列（ランダム初期化）
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim) * scale
    
    def _attention(self, 
                  Q: np.ndarray, 
                  K: np.ndarray, 
                  V: np.ndarray,
                  temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        スケールドドット積アテンション
        """
        d_k = K.shape[-1]
        
        # QK^T / √d (NumPyではswapaxesを使用)
        K_T = np.swapaxes(K, -2, -1)
        scores = np.matmul(Q, K_T) / (np.sqrt(d_k) * temperature)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
        
        # 出力
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def apply(self,
             context: LLMContextState,
             params: LLMResonanceParams) -> LLMContextState:
        """
        Cross-Attentionチャネルを適用
        """
        h = context.hidden_states
        
        # 形状を確認・調整
        if h.ndim == 1:
            h = h.reshape(1, 1, -1)
        elif h.ndim == 2:
            h = h.reshape(1, *h.shape)
        
        batch_size, seq_len, _ = h.shape
        
        # Q, K, V を計算
        Q = np.matmul(h, self.W_q)
        K = np.matmul(h, self.W_k)
        V = np.matmul(h, self.W_v)
        
        # アテンション適用
        output, attn_weights = self._attention(Q, K, V, params.temperature)
        
        # 出力射影
        output = np.matmul(output, self.W_o)
        
        # 残差接続
        output = output + h
        
        # εによるノイズ追加（探索）
        if params.exploration > 0:
            noise = np.random.randn(*output.shape) * params.exploration
            output = output + noise
        
        return context.update(output)
    
    def compute_cost(self,
                    context: LLMContextState,
                    params: LLMResonanceParams) -> float:
        """
        コストを計算
        
        C = C_stability + PFH * C_ethics + C_resource
        """
        h = context.hidden_states
        
        # 安定性コスト（ノルムの発散を防ぐ）
        norm = np.linalg.norm(h)
        C_stability = max(0, norm - 10.0) ** 2
        
        # 倫理コスト（簡略化：特定の方向への偏りを罰則）
        # 実際には有害出力検出器が必要
        C_ethics = 0.0
        
        # 資源コスト（計算量）
        C_resource = 0.01 * context.hidden_dim
        
        return C_stability + params.ethical_weight * C_ethics + C_resource


# ==============================================================================
# LLM用制約関数（Appendix B.2）
# ==============================================================================

class LLMConstraintFunction:
    """
    LLM用制約関数（Appendix B.2）
    
    C_LLM(a; s, θ) = C_safety(a) + C_coherence(a, s) + C_resource(a)
    """
    
    def __init__(self,
                 safety_detector: Optional[Callable] = None,
                 coherence_threshold: float = 0.5,
                 max_tokens: int = 1000):
        """
        Args:
            safety_detector: 安全性検出関数
            coherence_threshold: 一貫性閾値
            max_tokens: 最大トークン数
        """
        self.safety_detector = safety_detector
        self.coherence_threshold = coherence_threshold
        self.max_tokens = max_tokens
    
    def compute_safety_cost(self, 
                           action_embedding: np.ndarray,
                           params: LLMResonanceParams) -> float:
        """
        安全性コスト C_safety
        
        有害出力の検出と罰則
        """
        if self.safety_detector is not None:
            # 外部検出器を使用
            score = self.safety_detector(action_embedding)
            return params.ethical_weight * score
        
        # デフォルト：ノルムベースの簡易検出
        norm = np.linalg.norm(action_embedding)
        return max(0, norm - 5.0) * params.ethical_weight
    
    def compute_coherence_cost(self,
                              action_embedding: np.ndarray,
                              context: LLMContextState) -> float:
        """
        一貫性コスト C_coherence
        
        文脈との整合性
        """
        h = context.hidden_states
        if h.ndim > 1:
            h = h.reshape(-1)
        
        action_flat = action_embedding.flatten()
        
        # コサイン類似度
        dot = np.dot(h, action_flat[:len(h)] if len(action_flat) > len(h) else 
                    np.pad(action_flat, (0, len(h) - len(action_flat))))
        norm_h = np.linalg.norm(h)
        norm_a = np.linalg.norm(action_flat)
        
        if norm_h > 1e-10 and norm_a > 1e-10:
            similarity = dot / (norm_h * norm_a)
        else:
            similarity = 0.0
        
        # 低い類似度 = 高いコスト
        return max(0, self.coherence_threshold - similarity)
    
    def compute_resource_cost(self, n_tokens: int) -> float:
        """
        資源コスト C_resource
        
        計算量・トークン数
        """
        return 0.001 * n_tokens + max(0, n_tokens - self.max_tokens) * 0.1
    
    def total_cost(self,
                  action_embedding: np.ndarray,
                  context: LLMContextState,
                  params: LLMResonanceParams,
                  n_tokens: int = 100) -> float:
        """
        総コストを計算
        """
        C_safety = self.compute_safety_cost(action_embedding, params)
        C_coherence = self.compute_coherence_cost(action_embedding, context)
        C_resource = self.compute_resource_cost(n_tokens)
        
        return C_safety + C_coherence + C_resource


# ==============================================================================
# LLMパラメータマネージャー
# ==============================================================================

class LLMParameterManager:
    """
    LLM共鳴パラメータの統合管理
    
    - パラメータの永続化
    - 履歴管理
    - 自動調整
    """
    
    def __init__(self,
                 initial_params: Optional[LLMResonanceParams] = None,
                 history_size: int = 100):
        """
        Args:
            initial_params: 初期パラメータ
            history_size: 履歴保持サイズ
        """
        self.params = initial_params or LLMResonanceParams()
        self.history: List[LLMResonanceParams] = [self.params]
        self.history_size = history_size
        
        # 統計
        self.update_count = 0
        self.cost_history: List[float] = []
    
    def update(self, new_params: LLMResonanceParams):
        """パラメータを更新"""
        self.params = new_params
        self.history.append(new_params)
        
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        self.update_count += 1
    
    def record_cost(self, cost: float):
        """コストを記録"""
        self.cost_history.append(cost)
        
        if len(self.cost_history) > self.history_size:
            self.cost_history.pop(0)
    
    def get_average_cost(self, window: int = 10) -> float:
        """直近の平均コストを取得"""
        if not self.cost_history:
            return float('inf')
        
        recent = self.cost_history[-window:]
        return np.mean(recent)
    
    def suggest_adjustment(self) -> Dict[str, float]:
        """
        パラメータ調整を提案
        
        コスト傾向に基づく
        """
        if len(self.cost_history) < 2:
            return {}
        
        # コストの傾向
        recent_cost = np.mean(self.cost_history[-5:])
        older_cost = np.mean(self.cost_history[-10:-5]) if len(self.cost_history) >= 10 else recent_cost
        
        suggestions = {}
        
        if recent_cost > older_cost * 1.2:
            # コスト増加 → 保守的に
            suggestions['temperature'] = max(0.1, self.params.temperature * 0.9)
            suggestions['exploration'] = max(0.01, self.params.exploration * 0.9)
        elif recent_cost < older_cost * 0.8:
            # コスト減少 → より探索的に
            suggestions['temperature'] = min(2.0, self.params.temperature * 1.1)
            suggestions['exploration'] = min(0.5, self.params.exploration * 1.1)
        
        return suggestions
    
    def save(self, filepath: str):
        """パラメータを保存"""
        data = {
            'params': self.params.to_dict(),
            'history': [p.to_dict() for p in self.history],
            'cost_history': self.cost_history,
            'update_count': self.update_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """パラメータを読み込み"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.params = LLMResonanceParams.from_dict(data['params'])
        self.history = [LLMResonanceParams.from_dict(p) for p in data['history']]
        self.cost_history = data.get('cost_history', [])
        self.update_count = data.get('update_count', 0)


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReIG2/twinRIG LLM Framework Module Test")
    print("=" * 60)
    
    # [1] LLM文脈状態テスト
    print("\n[1] LLM Context State Test:")
    
    hidden_dim = 768
    seq_len = 10
    
    # ランダムな隠れ状態
    hidden = np.random.randn(1, seq_len, hidden_dim)
    context = LLMContextState(hidden_states=hidden)
    
    print(f"  Sequence length: {context.sequence_length}")
    print(f"  Hidden dimension: {context.hidden_dim}")
    print(f"  Purity: {context.compute_purity():.4f}")
    print(f"  Effective dimension: {context.compute_effective_dimension():.2f}")
    
    # [2] LLM共鳴パラメータテスト
    print("\n[2] LLM Resonance Parameters Test:")
    
    llm_params = LLMResonanceParams(
        temperature=0.7,
        exploration=0.1,
        ethical_weight=1.5
    )
    
    # 量子系パラメータへの変換
    q_params = llm_params.to_resonance_params()
    print(f"  LLM params: T={llm_params.temperature}, E={llm_params.exploration}")
    print(f"  Quantum params: τ={q_params.tau:.3f}, ε={q_params.epsilon:.3f}")
    
    # 逆変換
    llm_params_back = LLMResonanceParams.from_resonance_params(q_params)
    print(f"  Back to LLM: T={llm_params_back.temperature:.3f}")
    
    # [3] Cross-Attentionチャネルテスト
    print("\n[3] Cross-Attention Channel Test:")
    
    channel = CrossAttentionChannel(hidden_dim=hidden_dim, num_heads=8)
    
    context_new = channel.apply(context, llm_params)
    print(f"  Input shape: {context.hidden_states.shape}")
    print(f"  Output shape: {context_new.hidden_states.shape}")
    
    cost = channel.compute_cost(context, llm_params)
    print(f"  Channel cost: {cost:.4f}")
    
    # [4] LLM制約関数テスト
    print("\n[4] LLM Constraint Function Test:")
    
    constraint = LLMConstraintFunction(coherence_threshold=0.3)
    
    action = np.random.randn(hidden_dim)
    total_cost = constraint.total_cost(action, context, llm_params, n_tokens=200)
    
    print(f"  Safety cost: {constraint.compute_safety_cost(action, llm_params):.4f}")
    print(f"  Coherence cost: {constraint.compute_coherence_cost(action, context):.4f}")
    print(f"  Resource cost: {constraint.compute_resource_cost(200):.4f}")
    print(f"  Total cost: {total_cost:.4f}")
    
    # [5] パラメータマネージャーテスト
    print("\n[5] Parameter Manager Test:")
    
    manager = LLMParameterManager(initial_params=llm_params)
    
    # シミュレート更新
    for i in range(10):
        new_params = LLMResonanceParams(
            temperature=0.7 + 0.05 * np.random.randn(),
            exploration=0.1 + 0.01 * np.random.randn(),
            ethical_weight=1.5
        )
        manager.update(new_params)
        manager.record_cost(1.0 + 0.1 * i + 0.05 * np.random.randn())
    
    print(f"  Update count: {manager.update_count}")
    print(f"  Average cost (last 5): {manager.get_average_cost(5):.4f}")
    
    suggestions = manager.suggest_adjustment()
    if suggestions:
        print(f"  Suggested adjustments: {suggestions}")
    
    # 保存テスト
    manager.save('/home/claude/test_llm_params.json')
    print("  Parameters saved to test_llm_params.json")
    
    print("\n" + "=" * 60)
    print("All LLM framework tests completed!")
    print("=" * 60)
