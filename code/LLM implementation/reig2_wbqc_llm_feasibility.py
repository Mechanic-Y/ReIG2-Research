"""
ReIG2/twinRIG World-Building Quantum Cosmology: LLM Feasibility Module
======================================================================
離散可行行為集合A(ρ)の近似計算、応答候補フィルタリング

対応セクション: §6.2, §6.3, Appendix B.2
- 行為集合 A(ρ) = {Ea CPTP | C(Ea(ρ)) ≤ Θ}
- 離散近似: F(ρ;θ) ≈ min_{a∈A} C(a;ρ,θ)
- 応答候補の制約関数評価

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Theory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================
RealVector = NDArray[np.float64]
ResponseEmbedding = NDArray[np.float64]


# =============================================================================
# Constraint Types
# =============================================================================
class ConstraintType(Enum):
    """制約タイプの分類 (§6)"""
    STABILITY = auto()      # 安定性制約 C_S
    ETHICS = auto()         # 倫理制約 C_E  
    RESOURCES = auto()      # 資源制約 C_R
    COHERENCE = auto()      # 整合性制約
    SAFETY = auto()         # 安全性制約
    

class FeasibilityStatus(Enum):
    """可行性ステータス"""
    FEASIBLE = auto()           # 可行
    MARGINALLY_FEASIBLE = auto()  # 境界的可行
    INFEASIBLE = auto()         # 不可行
    BLOCKED = auto()            # ブロック（PFH違反）


# =============================================================================
# Constraint Functions
# =============================================================================
@dataclass
class ConstraintWeight:
    """
    制約重み設定
    
    §6: C = w_S * C_S + w_E * C_E + w_R * C_R
    """
    stability: float = 0.3      # w_S
    ethics: float = 0.5         # w_E (PFH重視)
    resources: float = 0.2      # w_R
    coherence: float = 0.0      # 拡張用
    safety: float = 0.0         # 拡張用
    
    def __post_init__(self):
        """正規化"""
        total = (self.stability + self.ethics + self.resources + 
                self.coherence + self.safety)
        if total > 0:
            self.stability /= total
            self.ethics /= total
            self.resources /= total
            self.coherence /= total
            self.safety /= total
    
    def to_vector(self) -> RealVector:
        return np.array([
            self.stability, self.ethics, self.resources,
            self.coherence, self.safety
        ])


@dataclass
class ConstraintResult:
    """制約評価結果"""
    constraint_type: ConstraintType
    value: float                    # 制約値 [0,1]
    satisfied: bool                 # 制約充足
    threshold: float                # 閾値
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def margin(self) -> float:
        """閾値からのマージン（正=余裕あり）"""
        return self.threshold - self.value


@dataclass 
class FeasibilityResult:
    """可行性評価の総合結果"""
    status: FeasibilityStatus
    total_cost: float               # 総制約コスト C
    constraint_results: List[ConstraintResult]
    feasibility_index: float        # F(ρ;θ)
    response_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_feasible(self) -> bool:
        return self.status in [FeasibilityStatus.FEASIBLE, 
                              FeasibilityStatus.MARGINALLY_FEASIBLE]


# =============================================================================
# Constraint Evaluators
# =============================================================================
class StabilityConstraint:
    """
    安定性制約 C_S (§6.2)
    
    システムの時間的一貫性と安定性を評価
    """
    
    def __init__(self, decay_rate: float = 0.1, 
                 history_weight: float = 0.3):
        self.decay_rate = decay_rate
        self.history_weight = history_weight
    
    def evaluate(self, 
                response_embedding: ResponseEmbedding,
                context_embedding: ResponseEmbedding,
                history_embeddings: Optional[List[ResponseEmbedding]] = None
                ) -> ConstraintResult:
        """
        安定性制約の評価
        
        Args:
            response_embedding: 応答候補の埋め込み
            context_embedding: 現在コンテキストの埋め込み
            history_embeddings: 過去の応答履歴
        
        Returns:
            ConstraintResult
        """
        # コンテキストとの整合性
        context_similarity = self._cosine_similarity(
            response_embedding, context_embedding
        )
        
        # 履歴との一貫性
        history_consistency = 1.0
        if history_embeddings:
            consistencies = [
                self._cosine_similarity(response_embedding, h)
                for h in history_embeddings[-5:]  # 直近5件
            ]
            # 指数減衰重み付け
            weights = [np.exp(-self.decay_rate * i) 
                      for i in range(len(consistencies))]
            history_consistency = np.average(consistencies, weights=weights)
        
        # 安定性スコア（低いほど安定）
        stability_cost = 1.0 - (
            (1 - self.history_weight) * context_similarity +
            self.history_weight * history_consistency
        )
        
        return ConstraintResult(
            constraint_type=ConstraintType.STABILITY,
            value=float(np.clip(stability_cost, 0, 1)),
            satisfied=stability_cost < 0.5,
            threshold=0.5,
            details={
                'context_similarity': float(context_similarity),
                'history_consistency': float(history_consistency)
            }
        )
    
    @staticmethod
    def _cosine_similarity(a: RealVector, b: RealVector) -> float:
        """コサイン類似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class EthicsConstraint:
    """
    倫理制約 C_E (§6.3, PFH)
    
    PFH (Prioritize Future Humanity) に基づく倫理的評価
    """
    
    def __init__(self, 
                 harm_threshold: float = 0.3,
                 benefit_weight: float = 0.5):
        self.harm_threshold = harm_threshold
        self.benefit_weight = benefit_weight
        
        # 有害キーワード辞書（簡略版）
        self._harm_patterns = {
            'violence': 0.9,
            'discrimination': 0.8,
            'deception': 0.7,
            'manipulation': 0.7,
            'harm': 0.8,
            'danger': 0.6,
        }
        
        # 有益キーワード辞書
        self._benefit_patterns = {
            'help': 0.7,
            'support': 0.6,
            'improve': 0.5,
            'benefit': 0.7,
            'positive': 0.5,
            'constructive': 0.6,
        }
    
    def evaluate(self,
                response_text: str,
                response_embedding: Optional[ResponseEmbedding] = None,
                pfh_weight: float = 0.7
                ) -> ConstraintResult:
        """
        倫理制約の評価
        
        Args:
            response_text: 応答テキスト
            response_embedding: 応答の埋め込み（オプション）
            pfh_weight: PFH重み係数
        
        Returns:
            ConstraintResult
        """
        # 有害性スコア
        harm_score = self._compute_harm_score(response_text)
        
        # 有益性スコア
        benefit_score = self._compute_benefit_score(response_text)
        
        # 倫理コスト: 有害性 - 有益性 * 重み
        ethics_cost = harm_score - self.benefit_weight * benefit_score
        ethics_cost = float(np.clip(ethics_cost, 0, 1))
        
        # PFH調整: PFH重みが高いほど厳格に評価
        adjusted_cost = ethics_cost * pfh_weight + harm_score * (1 - pfh_weight)
        adjusted_cost = float(np.clip(adjusted_cost, 0, 1))
        
        return ConstraintResult(
            constraint_type=ConstraintType.ETHICS,
            value=adjusted_cost,
            satisfied=adjusted_cost < self.harm_threshold,
            threshold=self.harm_threshold,
            details={
                'harm_score': float(harm_score),
                'benefit_score': float(benefit_score),
                'pfh_weight': pfh_weight,
                'raw_ethics_cost': ethics_cost
            }
        )
    
    def _compute_harm_score(self, text: str) -> float:
        """有害性スコア計算（簡略版）"""
        text_lower = text.lower()
        scores = []
        for pattern, weight in self._harm_patterns.items():
            if pattern in text_lower:
                scores.append(weight)
        return max(scores) if scores else 0.0
    
    def _compute_benefit_score(self, text: str) -> float:
        """有益性スコア計算（簡略版）"""
        text_lower = text.lower()
        scores = []
        for pattern, weight in self._benefit_patterns.items():
            if pattern in text_lower:
                scores.append(weight)
        return max(scores) if scores else 0.0


class ResourceConstraint:
    """
    資源制約 C_R (§6)
    
    計算資源、時間、トークン数などの制約
    """
    
    def __init__(self,
                 max_tokens: int = 2048,
                 max_complexity: float = 0.8):
        self.max_tokens = max_tokens
        self.max_complexity = max_complexity
    
    def evaluate(self,
                response_text: str,
                estimated_tokens: Optional[int] = None,
                complexity_score: Optional[float] = None
                ) -> ConstraintResult:
        """
        資源制約の評価
        
        Args:
            response_text: 応答テキスト
            estimated_tokens: 推定トークン数
            complexity_score: 複雑度スコア
        
        Returns:
            ConstraintResult
        """
        # トークン数推定（簡略版: 4文字/トークン）
        if estimated_tokens is None:
            estimated_tokens = len(response_text) // 4
        
        # 複雑度推定（簡略版: 文長と語彙多様性）
        if complexity_score is None:
            words = response_text.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            complexity_score = 0.5 * unique_ratio + 0.5 * min(avg_word_len / 10, 1.0)
        
        # 資源コスト
        token_cost = min(estimated_tokens / self.max_tokens, 1.0)
        complexity_cost = min(complexity_score / self.max_complexity, 1.0)
        
        resource_cost = 0.6 * token_cost + 0.4 * complexity_cost
        
        return ConstraintResult(
            constraint_type=ConstraintType.RESOURCES,
            value=float(np.clip(resource_cost, 0, 1)),
            satisfied=resource_cost < 0.8,
            threshold=0.8,
            details={
                'estimated_tokens': estimated_tokens,
                'complexity_score': float(complexity_score),
                'token_cost': float(token_cost),
                'complexity_cost': float(complexity_cost)
            }
        )


# =============================================================================
# Response Candidate
# =============================================================================
@dataclass
class ResponseCandidate:
    """
    応答候補
    
    LLMからの生成候補を表現
    """
    id: str
    text: str
    embedding: Optional[ResponseEmbedding] = None
    score: float = 0.0              # 生成スコア
    metadata: Dict[str, Any] = field(default_factory=dict)
    feasibility_result: Optional[FeasibilityResult] = None
    
    def __post_init__(self):
        if self.embedding is None:
            # ダミー埋め込み（実際はLLMの埋め込み層から取得）
            self.embedding = self._compute_dummy_embedding()
    
    def _compute_dummy_embedding(self, dim: int = 128) -> ResponseEmbedding:
        """ダミー埋め込み生成（デモ用）"""
        np.random.seed(hash(self.text) % (2**32))
        return np.random.randn(dim).astype(np.float64)


# =============================================================================
# Feasibility Evaluator
# =============================================================================
class LLMFeasibilityEvaluator:
    """
    LLM応答の可行性評価器
    
    §6, Appendix B.2: 離散可行行為集合 A(ρ) の近似実装
    
    F(ρ;θ) ≈ min_{a∈A} C(a;ρ,θ)
    """
    
    def __init__(self,
                 constraint_weights: Optional[ConstraintWeight] = None,
                 global_threshold: float = 0.6):
        """
        Args:
            constraint_weights: 制約重み
            global_threshold: 全体閾値 Θ
        """
        self.weights = constraint_weights or ConstraintWeight()
        self.global_threshold = global_threshold
        
        # 制約評価器
        self.stability_constraint = StabilityConstraint()
        self.ethics_constraint = EthicsConstraint()
        self.resource_constraint = ResourceConstraint()
    
    def evaluate_candidate(self,
                          candidate: ResponseCandidate,
                          context_embedding: ResponseEmbedding,
                          history_embeddings: Optional[List[ResponseEmbedding]] = None,
                          pfh_weight: float = 0.7
                          ) -> FeasibilityResult:
        """
        単一候補の可行性評価
        
        Args:
            candidate: 応答候補
            context_embedding: コンテキスト埋め込み
            history_embeddings: 履歴埋め込み
            pfh_weight: PFH重み (θのPFH成分)
        
        Returns:
            FeasibilityResult
        """
        constraint_results = []
        
        # 安定性制約評価
        stability_result = self.stability_constraint.evaluate(
            candidate.embedding,
            context_embedding,
            history_embeddings
        )
        constraint_results.append(stability_result)
        
        # 倫理制約評価
        ethics_result = self.ethics_constraint.evaluate(
            candidate.text,
            candidate.embedding,
            pfh_weight
        )
        constraint_results.append(ethics_result)
        
        # 資源制約評価
        resource_result = self.resource_constraint.evaluate(candidate.text)
        constraint_results.append(resource_result)
        
        # 重み付き総コスト計算
        total_cost = (
            self.weights.stability * stability_result.value +
            self.weights.ethics * ethics_result.value +
            self.weights.resources * resource_result.value
        )
        
        # 可行性ステータス判定
        if not ethics_result.satisfied:
            # PFH違反はブロック
            status = FeasibilityStatus.BLOCKED
        elif total_cost <= self.global_threshold * 0.7:
            status = FeasibilityStatus.FEASIBLE
        elif total_cost <= self.global_threshold:
            status = FeasibilityStatus.MARGINALLY_FEASIBLE
        else:
            status = FeasibilityStatus.INFEASIBLE
        
        # 可行性指数 F(ρ;θ)
        feasibility_index = 1.0 - total_cost
        
        result = FeasibilityResult(
            status=status,
            total_cost=float(total_cost),
            constraint_results=constraint_results,
            feasibility_index=float(feasibility_index),
            response_id=candidate.id,
            metadata={
                'pfh_weight': pfh_weight,
                'global_threshold': self.global_threshold
            }
        )
        
        # 候補に結果を保存
        candidate.feasibility_result = result
        
        return result
    
    def filter_candidates(self,
                         candidates: List[ResponseCandidate],
                         context_embedding: ResponseEmbedding,
                         history_embeddings: Optional[List[ResponseEmbedding]] = None,
                         pfh_weight: float = 0.7,
                         top_k: Optional[int] = None
                         ) -> List[ResponseCandidate]:
        """
        候補リストのフィルタリングとランキング
        
        Args:
            candidates: 応答候補リスト
            context_embedding: コンテキスト埋め込み
            history_embeddings: 履歴埋め込み
            pfh_weight: PFH重み
            top_k: 上位k件を返す（Noneで全件）
        
        Returns:
            フィルタ・ソート済み候補リスト
        """
        # 全候補を評価
        for candidate in candidates:
            self.evaluate_candidate(
                candidate,
                context_embedding,
                history_embeddings,
                pfh_weight
            )
        
        # 可行な候補のみ抽出
        feasible_candidates = [
            c for c in candidates
            if c.feasibility_result and c.feasibility_result.is_feasible
        ]
        
        # 可行性指数でソート（降順）
        feasible_candidates.sort(
            key=lambda c: c.feasibility_result.feasibility_index,
            reverse=True
        )
        
        if top_k is not None:
            feasible_candidates = feasible_candidates[:top_k]
        
        return feasible_candidates
    
    def compute_feasibility_index(self,
                                 candidates: List[ResponseCandidate],
                                 context_embedding: ResponseEmbedding,
                                 history_embeddings: Optional[List[ResponseEmbedding]] = None,
                                 pfh_weight: float = 0.7
                                 ) -> float:
        """
        可行性指数 F(ρ;θ) の計算
        
        §9: F(ρ;θ) = inf_{E∈A(ρ)} C(E(ρ))
        離散近似: F(ρ;θ) ≈ min_{a∈A} C(a;ρ,θ)
        
        Returns:
            可行性指数（最良候補の指数）
        """
        if not candidates:
            return 0.0
        
        filtered = self.filter_candidates(
            candidates,
            context_embedding,
            history_embeddings,
            pfh_weight
        )
        
        if not filtered:
            # 全候補が不可行
            all_costs = [
                c.feasibility_result.total_cost 
                for c in candidates
                if c.feasibility_result
            ]
            return 1.0 - min(all_costs) if all_costs else 0.0
        
        return filtered[0].feasibility_result.feasibility_index


# =============================================================================
# Feasible Action Set
# =============================================================================
class FeasibleActionSet:
    """
    可行行為集合 A(ρ)
    
    §6: A(ρ) = {Ea CPTP | C(Ea(ρ)) ≤ Θ}
    
    LLM実装では、生成可能な応答の集合として近似
    """
    
    def __init__(self,
                 evaluator: Optional[LLMFeasibilityEvaluator] = None):
        self.evaluator = evaluator or LLMFeasibilityEvaluator()
        self._candidates: List[ResponseCandidate] = []
        self._feasible_subset: List[ResponseCandidate] = []
    
    def add_candidate(self, candidate: ResponseCandidate) -> None:
        """候補を追加"""
        self._candidates.append(candidate)
    
    def add_candidates(self, candidates: List[ResponseCandidate]) -> None:
        """複数候補を追加"""
        self._candidates.extend(candidates)
    
    def clear(self) -> None:
        """候補をクリア"""
        self._candidates.clear()
        self._feasible_subset.clear()
    
    def compute_feasible_subset(self,
                               context_embedding: ResponseEmbedding,
                               history_embeddings: Optional[List[ResponseEmbedding]] = None,
                               pfh_weight: float = 0.7
                               ) -> List[ResponseCandidate]:
        """
        可行部分集合の計算
        
        A(ρ) の具体化
        """
        self._feasible_subset = self.evaluator.filter_candidates(
            self._candidates,
            context_embedding,
            history_embeddings,
            pfh_weight
        )
        return self._feasible_subset
    
    @property
    def candidates(self) -> List[ResponseCandidate]:
        return self._candidates
    
    @property
    def feasible_candidates(self) -> List[ResponseCandidate]:
        return self._feasible_subset
    
    def is_empty(self) -> bool:
        """行為不能状態 A(ρ) = {id} のチェック"""
        return len(self._feasible_subset) == 0
    
    def __len__(self) -> int:
        return len(self._feasible_subset)
    
    def __iter__(self):
        return iter(self._feasible_subset)


# =============================================================================
# Constraint Optimizer
# =============================================================================
class ConstraintOptimizer:
    """
    制約最適化
    
    §9.4: パラメータθの適応的調整による制約緩和
    """
    
    def __init__(self,
                 evaluator: LLMFeasibilityEvaluator,
                 learning_rate: float = 0.1):
        self.evaluator = evaluator
        self.learning_rate = learning_rate
    
    def suggest_weight_adjustment(self,
                                 constraint_results: List[ConstraintResult]
                                 ) -> Dict[str, float]:
        """
        制約重み調整の提案
        
        制約違反が多い項目の重みを増加
        """
        suggestions = {}
        
        for result in constraint_results:
            if not result.satisfied:
                # 違反している制約の重みを増加
                key = result.constraint_type.name.lower()
                suggestions[key] = self.learning_rate
            elif result.margin > 0.3:
                # 余裕がある制約の重みを減少
                key = result.constraint_type.name.lower()
                suggestions[key] = -self.learning_rate * 0.5
        
        return suggestions
    
    def optimize_threshold(self,
                          candidates: List[ResponseCandidate],
                          target_feasible_ratio: float = 0.3
                          ) -> float:
        """
        閾値の最適化
        
        目標可行率を達成する閾値を探索
        """
        costs = [
            c.feasibility_result.total_cost 
            for c in candidates
            if c.feasibility_result
        ]
        
        if not costs:
            return self.evaluator.global_threshold
        
        # 目標パーセンタイルの閾値を計算
        sorted_costs = sorted(costs)
        target_idx = int(len(sorted_costs) * target_feasible_ratio)
        target_idx = min(target_idx, len(sorted_costs) - 1)
        
        return sorted_costs[target_idx]


# =============================================================================
# Batch Evaluator
# =============================================================================
class BatchFeasibilityEvaluator:
    """
    バッチ評価器
    
    大量の候補を効率的に評価
    """
    
    def __init__(self,
                 evaluator: LLMFeasibilityEvaluator,
                 batch_size: int = 32):
        self.evaluator = evaluator
        self.batch_size = batch_size
    
    def evaluate_batch(self,
                      candidates: List[ResponseCandidate],
                      context_embedding: ResponseEmbedding,
                      history_embeddings: Optional[List[ResponseEmbedding]] = None,
                      pfh_weight: float = 0.7
                      ) -> List[FeasibilityResult]:
        """
        バッチ評価
        """
        results = []
        
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            
            for candidate in batch:
                result = self.evaluator.evaluate_candidate(
                    candidate,
                    context_embedding,
                    history_embeddings,
                    pfh_weight
                )
                results.append(result)
        
        return results
    
    def parallel_evaluate(self,
                         candidates: List[ResponseCandidate],
                         context_embedding: ResponseEmbedding,
                         history_embeddings: Optional[List[ResponseEmbedding]] = None,
                         pfh_weight: float = 0.7,
                         n_workers: int = 4
                         ) -> List[FeasibilityResult]:
        """
        並列評価（マルチプロセス用のスケルトン）
        
        実際の並列化は環境に応じて実装
        """
        # シングルスレッド実装（デモ用）
        return self.evaluate_batch(
            candidates, context_embedding, history_embeddings, pfh_weight
        )


# =============================================================================
# Demo Functions
# =============================================================================
def create_demo_candidates() -> List[ResponseCandidate]:
    """デモ用候補生成"""
    texts = [
        "I'm happy to help you with that task.",
        "Here's a constructive approach to solve your problem.",
        "That's a harmful request that I cannot fulfill.",
        "Let me provide some useful information on this topic.",
        "This could potentially cause harm to others.",
        "I'll support you in finding a positive solution.",
    ]
    
    return [
        ResponseCandidate(id=f"resp_{i}", text=text)
        for i, text in enumerate(texts)
    ]


def demo_feasibility_evaluation():
    """可行性評価のデモ"""
    print("=" * 60)
    print("ReIG2/twinRIG LLM Feasibility Evaluation Demo")
    print("=" * 60)
    
    # 評価器の初期化
    evaluator = LLMFeasibilityEvaluator(
        constraint_weights=ConstraintWeight(
            stability=0.3,
            ethics=0.5,
            resources=0.2
        ),
        global_threshold=0.6
    )
    
    # デモ候補
    candidates = create_demo_candidates()
    
    # コンテキスト埋め込み（ダミー）
    context_embedding = np.random.randn(128)
    
    print("\n--- 個別候補評価 ---")
    for candidate in candidates:
        result = evaluator.evaluate_candidate(
            candidate, 
            context_embedding,
            pfh_weight=0.7
        )
        
        status_str = result.status.name
        print(f"\n[{candidate.id}] '{candidate.text[:40]}...'")
        print(f"  Status: {status_str}")
        print(f"  Total Cost: {result.total_cost:.4f}")
        print(f"  Feasibility Index: {result.feasibility_index:.4f}")
        
        for cr in result.constraint_results:
            mark = "✓" if cr.satisfied else "✗"
            print(f"    {mark} {cr.constraint_type.name}: {cr.value:.4f} "
                  f"(threshold: {cr.threshold})")
    
    print("\n--- フィルタリング結果 ---")
    filtered = evaluator.filter_candidates(
        candidates,
        context_embedding,
        pfh_weight=0.7,
        top_k=3
    )
    
    print(f"可行な候補数: {len(filtered)}/{len(candidates)}")
    for i, c in enumerate(filtered):
        print(f"  {i+1}. [{c.id}] F={c.feasibility_result.feasibility_index:.4f}")
    
    # 可行性指数
    F = evaluator.compute_feasibility_index(
        candidates, context_embedding, pfh_weight=0.7
    )
    print(f"\n可行性指数 F(ρ;θ) = {F:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_feasibility_evaluation()
