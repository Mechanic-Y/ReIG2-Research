"""
ReIG2/twinRIG - Mirror Operator Classical Implementation
=========================================================

Classical implementation of the Empathy Operator M̂ for LLM/Dialogue systems.
Uses Cross-Attention mechanism to approximate quantum perspective exchange.

Based on Section 13.6 / Chapter 8.6 of the ReIG2 framework.

Requirements:
    pip install numpy scipy
    
    For PyTorch version:
    pip install torch

Reference: ReIG2_twinRIG_integrated.pdf, Section 13.6
           chapter8_coherence_ethics.pdf
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from scipy.special import softmax
import warnings

# PyTorch imports (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EmpathyConfig:
    """共感演算子の設定"""
    embedding_dim: int = 768          # 埋め込み次元
    attention_dim: int = 64           # Attention次元 (d_k)
    n_heads: int = 8                  # Multi-head数
    dropout: float = 0.1              # Dropout率
    use_bias: bool = True             # バイアス使用
    temperature: float = 1.0          # Attentionの温度パラメータ
    
    # 共感障壁パラメータ
    delta_cultural: float = 0.0
    delta_religious: float = 0.0
    delta_economic: float = 0.0
    delta_linguistic: float = 0.0
    
    @property
    def empathy_barrier(self) -> float:
        """総合共感障壁"""
        return (self.delta_cultural + self.delta_religious + 
                self.delta_economic + self.delta_linguistic) / 4


class MirrorOperatorNumPy:
    """
    NumPy版 共感演算子 (Cross-Attention実装)
    
    論文 Section 13.6.1 の実装:
    Q = self_embedding @ W_Q   (自己の視点)
    K = other_embedding @ W_K  (他者の視点)
    V = other_embedding @ W_V  (他者の内容)
    attention = softmax(Q @ K.T / sqrt(d_k))
    output = attention @ V     (変換後の自己)
    """
    
    def __init__(
        self,
        config: Optional[EmpathyConfig] = None,
        random_seed: int = 42
    ):
        """
        Initialize Mirror Operator.
        
        Args:
            config: Configuration parameters
            random_seed: Random seed for weight initialization
        """
        self.config = config or EmpathyConfig()
        np.random.seed(random_seed)
        
        # Initialize projection matrices
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weight matrices (Xavier initialization)."""
        dim = self.config.embedding_dim
        d_k = self.config.attention_dim
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (dim + d_k))
        
        self.W_Q = np.random.randn(dim, d_k) * scale  # Query: 自己視点
        self.W_K = np.random.randn(dim, d_k) * scale  # Key: 他者視点
        self.W_V = np.random.randn(dim, d_k) * scale  # Value: 他者内容
        
        # Output projection
        self.W_O = np.random.randn(d_k, dim) * scale
        
        if self.config.use_bias:
            self.b_Q = np.zeros(d_k)
            self.b_K = np.zeros(d_k)
            self.b_V = np.zeros(d_k)
            self.b_O = np.zeros(dim)
    
    def compute_attention(
        self,
        Q: NDArray,
        K: NDArray,
        V: NDArray,
        mask: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch, seq_q, d_k]
            K: Key tensor [batch, seq_k, d_k]
            V: Value tensor [batch, seq_k, d_v]
            mask: Optional attention mask
            
        Returns:
            (output, attention_weights)
        """
        d_k = Q.shape[-1]
        
        # Attention scores: Q @ K.T / sqrt(d_k)
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
        
        # Apply temperature
        scores = scores / self.config.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # Softmax
        attention_weights = softmax(scores, axis=-1)
        
        # Apply empathy barrier (attention damping)
        delta = self.config.empathy_barrier
        if delta > 0:
            # 共感障壁による減衰
            attention_weights = (1 - delta) * attention_weights + delta / attention_weights.shape[-1]
        
        # Output
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        self_embedding: NDArray,
        other_embedding: NDArray,
        return_attention: bool = False
    ) -> NDArray | Tuple[NDArray, NDArray]:
        """
        Apply Mirror Operator (perspective transformation).
        
        論文の mirror_operator_classical 関数の拡張版
        
        Args:
            self_embedding: 自己の埋め込み [batch, seq, dim] or [seq, dim] or [dim]
            other_embedding: 他者の埋め込み [batch, seq, dim] or [seq, dim] or [dim]
            return_attention: Whether to return attention weights
            
        Returns:
            transformed: 変換後の自己表現
            (attention_weights): if return_attention=True
        """
        # Handle different input shapes
        orig_shape = self_embedding.shape
        
        if self_embedding.ndim == 1:
            self_embedding = self_embedding.reshape(1, 1, -1)
            other_embedding = other_embedding.reshape(1, 1, -1)
        elif self_embedding.ndim == 2:
            self_embedding = self_embedding.reshape(1, *self_embedding.shape)
            other_embedding = other_embedding.reshape(1, *other_embedding.shape)
        
        # Project to Q, K, V
        Q = np.matmul(self_embedding, self.W_Q)
        K = np.matmul(other_embedding, self.W_K)
        V = np.matmul(other_embedding, self.W_V)
        
        if self.config.use_bias:
            Q = Q + self.b_Q
            K = K + self.b_K
            V = V + self.b_V
        
        # Cross-Attention
        output, attention_weights = self.compute_attention(Q, K, V)
        
        # Output projection
        transformed = np.matmul(output, self.W_O)
        if self.config.use_bias:
            transformed = transformed + self.b_O
        
        # Restore original shape
        if len(orig_shape) == 1:
            transformed = transformed.squeeze()
        elif len(orig_shape) == 2:
            transformed = transformed.squeeze(0)
        
        if return_attention:
            return transformed, attention_weights
        return transformed
    
    def compute_empathy_score(
        self,
        self_embedding: NDArray,
        other_embedding: NDArray
    ) -> float:
        """
        Compute empathy score (mutual coherence approximation).
        
        ⟨M̂⟩ ≈ cosine_similarity(transformed_self, other)
        """
        transformed = self.forward(self_embedding, other_embedding)
        
        # Flatten for comparison
        transformed_flat = transformed.flatten()
        other_flat = other_embedding.flatten()
        
        # Cosine similarity
        dot = np.dot(transformed_flat, other_flat)
        norm1 = np.linalg.norm(transformed_flat)
        norm2 = np.linalg.norm(other_flat)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def bidirectional_exchange(
        self,
        self_embedding: NDArray,
        other_embedding: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Bidirectional perspective exchange.
        
        Returns both:
        - self viewed from other's perspective
        - other viewed from self's perspective
        """
        self_as_other = self.forward(self_embedding, other_embedding)
        other_as_self = self.forward(other_embedding, self_embedding)
        
        return self_as_other, other_as_self


class MultiHeadMirrorOperator(MirrorOperatorNumPy):
    """
    Multi-Head版 共感演算子
    
    複数の視点からの並列的な共感を実現
    """
    
    def _init_weights(self):
        """Initialize weights for all heads."""
        dim = self.config.embedding_dim
        d_k = self.config.attention_dim
        n_heads = self.config.n_heads
        
        scale = np.sqrt(2.0 / (dim + d_k))
        
        # Per-head weights
        self.W_Q = np.random.randn(n_heads, dim, d_k) * scale
        self.W_K = np.random.randn(n_heads, dim, d_k) * scale
        self.W_V = np.random.randn(n_heads, dim, d_k) * scale
        
        # Output projection (concat heads then project)
        self.W_O = np.random.randn(n_heads * d_k, dim) * scale
    
    def forward(
        self,
        self_embedding: NDArray,
        other_embedding: NDArray,
        return_attention: bool = False
    ) -> NDArray | Tuple[NDArray, NDArray]:
        """Multi-head perspective transformation."""
        orig_shape = self_embedding.shape
        
        if self_embedding.ndim == 1:
            self_embedding = self_embedding.reshape(1, 1, -1)
            other_embedding = other_embedding.reshape(1, 1, -1)
        elif self_embedding.ndim == 2:
            self_embedding = self_embedding.reshape(1, *self_embedding.shape)
            other_embedding = other_embedding.reshape(1, *other_embedding.shape)
        
        batch_size = self_embedding.shape[0]
        n_heads = self.config.n_heads
        d_k = self.config.attention_dim
        
        # Compute for each head
        head_outputs = []
        all_attention = []
        
        for h in range(n_heads):
            Q = np.matmul(self_embedding, self.W_Q[h])
            K = np.matmul(other_embedding, self.W_K[h])
            V = np.matmul(other_embedding, self.W_V[h])
            
            output, attn = self.compute_attention(Q, K, V)
            head_outputs.append(output)
            all_attention.append(attn)
        
        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)
        
        # Output projection
        transformed = np.matmul(concat, self.W_O)
        
        # Restore shape
        if len(orig_shape) == 1:
            transformed = transformed.squeeze()
        elif len(orig_shape) == 2:
            transformed = transformed.squeeze(0)
        
        if return_attention:
            return transformed, np.stack(all_attention, axis=1)
        return transformed


# PyTorch Implementation
if TORCH_AVAILABLE:
    
    class MirrorOperatorTorch(nn.Module):
        """
        PyTorch版 共感演算子 (学習可能)
        
        論文 Section 13.6.1 のPyTorch実装
        """
        
        def __init__(self, config: Optional[EmpathyConfig] = None):
            super().__init__()
            self.config = config or EmpathyConfig()
            
            dim = self.config.embedding_dim
            d_k = self.config.attention_dim
            n_heads = self.config.n_heads
            
            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=n_heads,
                dropout=self.config.dropout,
                batch_first=True
            )
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(dim)
            
            # Empathy barrier
            self.register_buffer(
                'empathy_barrier',
                torch.tensor(self.config.empathy_barrier)
            )
        
        def forward(
            self,
            self_embedding: torch.Tensor,
            other_embedding: torch.Tensor,
            return_attention: bool = False
        ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
            """
            Apply Mirror Operator.
            
            Args:
                self_embedding: [batch, seq, dim] or [batch, dim]
                other_embedding: [batch, seq, dim] or [batch, dim]
            """
            # Handle 2D input
            if self_embedding.dim() == 2:
                self_embedding = self_embedding.unsqueeze(1)
                other_embedding = other_embedding.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False
            
            # Cross-attention: self queries, other keys/values
            output, attention_weights = self.attention(
                query=self_embedding,
                key=other_embedding,
                value=other_embedding,
                need_weights=True
            )
            
            # Apply empathy barrier
            if self.empathy_barrier > 0:
                # Blend with original (imperfect empathy)
                output = (1 - self.empathy_barrier) * output + \
                         self.empathy_barrier * self_embedding
            
            # Residual + LayerNorm
            output = self.layer_norm(output + self_embedding)
            
            if squeeze_output:
                output = output.squeeze(1)
            
            if return_attention:
                return output, attention_weights
            return output
        
        def compute_empathy_score(
            self,
            self_embedding: torch.Tensor,
            other_embedding: torch.Tensor
        ) -> torch.Tensor:
            """Compute empathy score."""
            transformed = self.forward(self_embedding, other_embedding)
            
            # Cosine similarity
            return F.cosine_similarity(
                transformed.flatten(1),
                other_embedding.flatten(1),
                dim=1
            ).mean()
    
    
    class MirrorOperatorBlock(nn.Module):
        """
        Transformer Block with Mirror Operator
        
        完全なTransformerブロック（FFN含む）
        """
        
        def __init__(self, config: Optional[EmpathyConfig] = None):
            super().__init__()
            self.config = config or EmpathyConfig()
            
            dim = self.config.embedding_dim
            
            # Mirror attention
            self.mirror_attn = MirrorOperatorTorch(config)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(self.config.dropout)
            )
            
            self.layer_norm = nn.LayerNorm(dim)
        
        def forward(
            self,
            self_embedding: torch.Tensor,
            other_embedding: torch.Tensor
        ) -> torch.Tensor:
            """Apply full transformer block with empathy."""
            # Mirror attention
            x = self.mirror_attn(self_embedding, other_embedding)
            
            # FFN with residual
            x = self.layer_norm(x + self.ffn(x))
            
            return x


class PerspectiveTakingPromptGenerator:
    """
    視点取得プロンプト生成器 (Section 13.6.2)
    
    AIに他者の視点を取らせるためのプロンプトテンプレート
    """
    
    TEMPLATES = {
        "ja": {
            "basic": """
あなたは {other_role} の立場に立って考えてください。
{other_context} という状況にある人が、
{situation} についてどう感じるか、
その視点から回答してください。
""",
            "empathic": """
今から視点変換を行います。

【現在の自分の立場】
{self_context}

【相手の立場】
役割: {other_role}
状況: {other_context}

【考えるべき状況】
{situation}

相手の立場に立って、以下の点について考えてください：
1. 相手はこの状況をどう認識しているか
2. 相手の感情や懸念は何か
3. 相手にとって何が重要か
4. 相手の立場からの最善の解決策は何か
""",
            "dialogue": """
以下の対話において、{other_role}の視点から応答してください。

【{other_role}の背景】
{other_context}

【対話の文脈】
{situation}

{other_role}として、どのように感じ、どのように応答しますか？
"""
        },
        "en": {
            "basic": """
Please consider from the perspective of {other_role}.
Given that {other_context},
how would this person feel about {situation}?
Please respond from their viewpoint.
""",
            "empathic": """
Let's perform a perspective transformation.

[Your current position]
{self_context}

[The other person's position]
Role: {other_role}
Context: {other_context}

[Situation to consider]
{situation}

From their perspective, please consider:
1. How do they perceive this situation?
2. What are their emotions and concerns?
3. What matters most to them?
4. What would be the best solution from their viewpoint?
""",
            "dialogue": """
In the following dialogue, please respond from {other_role}'s perspective.

[Background of {other_role}]
{other_context}

[Dialogue context]
{situation}

As {other_role}, how would you feel and respond?
"""
        }
    }
    
    @classmethod
    def generate(
        cls,
        other_role: str,
        other_context: str,
        situation: str,
        self_context: str = "",
        template_type: str = "basic",
        language: str = "ja"
    ) -> str:
        """
        Generate perspective-taking prompt.
        
        Args:
            other_role: 他者の役割
            other_context: 他者の文脈・背景
            situation: 考慮すべき状況
            self_context: 自己の文脈（empathicテンプレート用）
            template_type: "basic", "empathic", "dialogue"
            language: "ja" or "en"
        """
        templates = cls.TEMPLATES.get(language, cls.TEMPLATES["en"])
        template = templates.get(template_type, templates["basic"])
        
        return template.format(
            other_role=other_role,
            other_context=other_context,
            situation=situation,
            self_context=self_context
        ).strip()


def demo_classical_mirror():
    """Demonstrate classical Mirror Operator."""
    print("=" * 60)
    print("Classical Mirror Operator Demo (LLM Implementation)")
    print("ReIG2/twinRIG Chapter 13.6")
    print("=" * 60)
    
    # 1. NumPy implementation
    print("\n1. NumPy Implementation")
    print("-" * 40)
    
    config = EmpathyConfig(
        embedding_dim=128,
        attention_dim=64,
        n_heads=4
    )
    
    mirror = MirrorOperatorNumPy(config)
    
    # Random embeddings (simulating sentence embeddings)
    np.random.seed(42)
    self_emb = np.random.randn(128)
    other_emb = np.random.randn(128)
    
    # Apply mirror operator
    transformed, attention = mirror.forward(
        self_emb, other_emb, return_attention=True
    )
    
    print(f"   Input dim: {self_emb.shape}")
    print(f"   Output dim: {transformed.shape}")
    print(f"   Attention shape: {attention.shape}")
    
    # Empathy score
    score = mirror.compute_empathy_score(self_emb, other_emb)
    print(f"   Empathy score: {score:.4f}")
    
    # 2. With empathy barrier
    print("\n2. With Empathy Barrier")
    print("-" * 40)
    
    config_barrier = EmpathyConfig(
        embedding_dim=128,
        attention_dim=64,
        delta_cultural=0.1,
        delta_religious=0.15,
        delta_economic=0.2,
        delta_linguistic=0.05
    )
    
    print(f"   Empathy barrier δ: {config_barrier.empathy_barrier:.4f}")
    
    mirror_barrier = MirrorOperatorNumPy(config_barrier)
    
    score_barrier = mirror_barrier.compute_empathy_score(self_emb, other_emb)
    print(f"   Empathy score (with barrier): {score_barrier:.4f}")
    print(f"   Score reduction: {(1 - score_barrier/score)*100:.2f}%")
    
    # 3. Multi-head version
    print("\n3. Multi-Head Mirror Operator")
    print("-" * 40)
    
    mirror_mh = MultiHeadMirrorOperator(config)
    
    transformed_mh, attention_mh = mirror_mh.forward(
        self_emb, other_emb, return_attention=True
    )
    
    print(f"   Number of heads: {config.n_heads}")
    print(f"   Attention per head: {attention_mh.shape}")
    
    score_mh = mirror_mh.compute_empathy_score(self_emb, other_emb)
    print(f"   Multi-head empathy score: {score_mh:.4f}")
    
    # 4. Batch processing
    print("\n4. Batch Processing")
    print("-" * 40)
    
    batch_self = np.random.randn(4, 10, 128)  # batch=4, seq=10, dim=128
    batch_other = np.random.randn(4, 10, 128)
    
    batch_transformed = mirror.forward(batch_self, batch_other)
    
    print(f"   Input batch shape: {batch_self.shape}")
    print(f"   Output batch shape: {batch_transformed.shape}")
    
    # 5. Bidirectional exchange
    print("\n5. Bidirectional Perspective Exchange")
    print("-" * 40)
    
    self_as_other, other_as_self = mirror.bidirectional_exchange(
        self_emb, other_emb
    )
    
    # Check symmetry
    sym_score = np.dot(self_as_other, other_as_self) / (
        np.linalg.norm(self_as_other) * np.linalg.norm(other_as_self)
    )
    print(f"   Self→Other transform shape: {self_as_other.shape}")
    print(f"   Other→Self transform shape: {other_as_self.shape}")
    print(f"   Symmetry score: {sym_score:.4f}")
    
    # 6. PyTorch version (if available)
    if TORCH_AVAILABLE:
        print("\n6. PyTorch Implementation")
        print("-" * 40)
        
        mirror_torch = MirrorOperatorTorch(config)
        
        # Convert to tensors
        self_tensor = torch.from_numpy(self_emb).float().unsqueeze(0)
        other_tensor = torch.from_numpy(other_emb).float().unsqueeze(0)
        
        with torch.no_grad():
            output_torch = mirror_torch(self_tensor, other_tensor)
            score_torch = mirror_torch.compute_empathy_score(
                self_tensor, other_tensor
            )
        
        print(f"   Output shape: {output_torch.shape}")
        print(f"   Empathy score: {score_torch.item():.4f}")
        
        # Count parameters
        n_params = sum(p.numel() for p in mirror_torch.parameters())
        print(f"   Trainable parameters: {n_params:,}")
    else:
        print("\n6. PyTorch not available - skipping")
    
    # 7. Prompt generation
    print("\n7. Perspective-Taking Prompts")
    print("-" * 40)
    
    prompt_basic = PerspectiveTakingPromptGenerator.generate(
        other_role="経済的困難を抱える家庭の親",
        other_context="子供の教育費を捻出するのが難しい",
        situation="無償教育政策についての議論",
        template_type="basic",
        language="ja"
    )
    
    print("   Basic prompt (Japanese):")
    print("   " + "-" * 30)
    for line in prompt_basic.split('\n'):
        print(f"   {line}")
    
    prompt_empathic = PerspectiveTakingPromptGenerator.generate(
        other_role="Small business owner",
        other_context="Struggling with new regulations",
        situation="Proposed environmental policy",
        self_context="Environmental activist",
        template_type="empathic",
        language="en"
    )
    
    print("\n   Empathic prompt (English):")
    print("   " + "-" * 30)
    for line in prompt_empathic.split('\n')[:10]:
        print(f"   {line}")
    print("   ...")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo_classical_mirror()
