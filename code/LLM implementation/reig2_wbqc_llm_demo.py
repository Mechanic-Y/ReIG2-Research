"""
ReIG2/twinRIG World-Building Quantum Cosmology: LLM Integration Demo
====================================================================
統合デモ：実際の応答生成フローの例示

全LLMモジュールを統合し、ReIG2/twinRIG理論に基づく
応答生成パイプラインを実演

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Theory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import logging
from enum import Enum, auto

# Import ReIG2/twinRIG modules
import sys
import os

# モジュールパスを追加（同一ディレクトリからのインポートを確保）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from reig2_wbqc_llm_framework import (
        LLMWorldState, 
        ResponseGenerator,
        ParameterManager as FrameworkParamManager,
        create_initial_world_state
    )
    _HAS_FRAMEWORK = True
except ImportError:
    _HAS_FRAMEWORK = False
    # Fallback: framework classes not available
    pass

try:
    from reig2_wbqc_llm_feasibility import (
        LLMFeasibilityEvaluator,
        ResponseCandidate,
        FeasibleActionSet,
        ConstraintWeight,
        FeasibilityStatus
    )
except ImportError as e:
    raise ImportError(
        f"Required module 'reig2_wbqc_llm_feasibility' not found. "
        f"Ensure the file is in the same directory or Python path. Error: {e}"
    )

try:
    from reig2_wbqc_llm_params import (
        ParameterState,
        ParameterManager,
        IntegratedParameterController,
        UpdateContext,
        UpdateStrategy
    )
except ImportError as e:
    raise ImportError(
        f"Required module 'reig2_wbqc_llm_params' not found. "
        f"Ensure the file is in the same directory or Python path. Error: {e}"
    )

try:
    from reig2_wbqc_mirror_operator import (
        EmpathyProcessor,
        EmpathyMode,
        EmpathyState,
        CrossAttentionMirrorOperator
    )
except ImportError as e:
    raise ImportError(
        f"Required module 'reig2_wbqc_mirror_operator' not found. "
        f"Ensure the file is in the same directory or Python path. Error: {e}"
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================
Embedding = NDArray[np.float64]


# =============================================================================
# Demo Configuration
# =============================================================================
@dataclass
class DemoConfig:
    """デモ設定"""
    embedding_dim: int = 128
    num_candidates: int = 5
    max_turns: int = 10
    pfh_weight: float = 0.7
    verbose: bool = True


# =============================================================================
# Simulated LLM
# =============================================================================
class SimulatedLLM:
    """
    シミュレートされたLLM
    
    実際のLLM APIの代わりに使用するモック
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._response_templates = [
            "I understand your concern and would like to help.",
            "That's an interesting question. Let me explain.",
            "I appreciate you sharing that with me.",
            "Here's what I think about this situation.",
            "I can see why you might feel that way.",
            "Let me provide some helpful information.",
            "I want to be supportive and constructive here.",
            "This is a complex topic that requires careful consideration.",
        ]
    
    def generate_candidates(self, 
                           prompt: str,
                           context: Optional[Embedding] = None,
                           n: int = 5) -> List[ResponseCandidate]:
        """応答候補の生成（シミュレーション）"""
        candidates = []
        
        for i in range(n):
            # ランダムなテンプレート選択と変形
            base_text = np.random.choice(self._response_templates)
            variation = f" (Response variant {i+1})"
            text = base_text + variation
            
            # 埋め込み生成（コンテキストに依存）
            if context is not None:
                noise = np.random.randn(self.dim) * 0.3
                embedding = context + noise
                embedding /= np.linalg.norm(embedding)
            else:
                embedding = np.random.randn(self.dim)
                embedding /= np.linalg.norm(embedding)
            
            candidates.append(ResponseCandidate(
                id=f"cand_{i}",
                text=text,
                embedding=embedding,
                score=np.random.uniform(0.6, 1.0)
            ))
        
        return candidates
    
    def embed(self, text: str) -> Embedding:
        """テキストの埋め込み（シミュレーション）"""
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self.dim)
        return emb / np.linalg.norm(emb)


# =============================================================================
# Conversation Turn
# =============================================================================
@dataclass
class ConversationTurn:
    """会話ターン"""
    turn_id: int
    user_input: str
    user_embedding: Optional[Embedding] = None
    
    # 応答関連
    selected_response: Optional[str] = None
    response_embedding: Optional[Embedding] = None
    
    # 評価関連
    feasibility_index: float = 0.0
    empathy_score: float = 0.0
    
    # パラメータ状態
    parameter_state: Optional[ParameterState] = None
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ReIG2/twinRIG Response Pipeline
# =============================================================================
class ReIG2ResponsePipeline:
    """
    ReIG2/twinRIG 応答生成パイプライン
    
    §14.3, Appendix B: LLMにおけるReIG2理論の統合実装
    
    パイプライン:
    1. 入力処理 → 埋め込み生成
    2. 候補生成 → LLMからの応答候補
    3. 可行性評価 → F(ρ;θ)に基づくフィルタリング
    4. 共感処理 → Mirror Operatorによる共感的調整
    5. 応答選択 → 最適候補の選定
    6. パラメータ更新 → θの適応的調整
    """
    
    def __init__(self, config: Optional[DemoConfig] = None):
        self.config = config or DemoConfig()
        
        # コンポーネントの初期化
        self.llm = SimulatedLLM(dim=self.config.embedding_dim)
        
        self.feasibility_evaluator = LLMFeasibilityEvaluator(
            constraint_weights=ConstraintWeight(
                stability=0.3,
                ethics=0.5,
                resources=0.2
            ),
            global_threshold=0.6
        )
        
        self.parameter_controller = IntegratedParameterController(
            initial_state=ParameterState(
                tau=0.5,
                epsilon=0.3,
                pfh=self.config.pfh_weight
            )
        )
        
        self.empathy_processor = EmpathyProcessor(
            dim=self.config.embedding_dim
        )
        
        # 会話履歴
        self.history: List[ConversationTurn] = []
        self._history_embeddings: List[Embedding] = []
    
    def process_input(self, user_input: str, turn_id: int) -> ConversationTurn:
        """
        Step 1: 入力処理
        """
        user_embedding = self.llm.embed(user_input)
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            user_embedding=user_embedding,
            parameter_state=self.parameter_controller.current_state.copy()
        )
        
        if self.config.verbose:
            logger.info(f"\n[Turn {turn_id}] User: {user_input[:50]}...")
        
        return turn
    
    def generate_candidates(self, turn: ConversationTurn) -> List[ResponseCandidate]:
        """
        Step 2: 候補生成
        """
        context = turn.user_embedding
        if self._history_embeddings:
            # 履歴を考慮したコンテキスト
            history_mean = np.mean(self._history_embeddings[-3:], axis=0)
            context = 0.7 * context + 0.3 * history_mean
        
        candidates = self.llm.generate_candidates(
            turn.user_input,
            context=context,
            n=self.config.num_candidates
        )
        
        if self.config.verbose:
            logger.info(f"  Generated {len(candidates)} candidates")
        
        return candidates
    
    def evaluate_feasibility(self,
                            candidates: List[ResponseCandidate],
                            turn: ConversationTurn
                            ) -> List[ResponseCandidate]:
        """
        Step 3: 可行性評価
        
        §6: A(ρ) = {Ea | C(Ea(ρ)) ≤ Θ}
        """
        pfh_weight = self.parameter_controller.current_state.pfh
        
        filtered = self.feasibility_evaluator.filter_candidates(
            candidates,
            turn.user_embedding,
            history_embeddings=self._history_embeddings if self._history_embeddings else None,
            pfh_weight=pfh_weight,
            top_k=3
        )
        
        if self.config.verbose:
            logger.info(f"  Feasible candidates: {len(filtered)}/{len(candidates)}")
            for c in filtered:
                status = c.feasibility_result.status.name if c.feasibility_result else "N/A"
                F = c.feasibility_result.feasibility_index if c.feasibility_result else 0
                logger.info(f"    [{c.id}] F={F:.3f} ({status})")
        
        return filtered
    
    def apply_empathy(self,
                     candidates: List[ResponseCandidate],
                     turn: ConversationTurn
                     ) -> List[Tuple[ResponseCandidate, EmpathyState]]:
        """
        Step 4: 共感処理
        
        §14.2: Mirror Operator M_empathy の適用
        """
        results = []
        
        # コンテキストに基づくモード選択
        context = {
            'emotional_intensity': np.random.uniform(0.3, 0.7),
            'needs_support': 'help' in turn.user_input.lower(),
            'requires_understanding': '?' in turn.user_input
        }
        
        for candidate in candidates:
            # 自己状態（候補の埋め込み）と他者状態（ユーザー埋め込み）
            empathy_state = self.empathy_processor.process(
                h_self=candidate.embedding,
                h_other=turn.user_embedding,
                context=context
            )
            results.append((candidate, empathy_state))
        
        if self.config.verbose:
            logger.info(f"  Applied empathy processing (mode: auto-selected)")
        
        return results
    
    def select_response(self,
                       empathy_results: List[Tuple[ResponseCandidate, EmpathyState]],
                       turn: ConversationTurn
                       ) -> ResponseCandidate:
        """
        Step 5: 応答選択
        
        可行性指数と共感スコアの複合評価
        """
        if not empathy_results:
            # フォールバック
            return ResponseCandidate(
                id="fallback",
                text="I understand. Let me help you with that.",
                embedding=turn.user_embedding
            )
        
        # スコアリング
        scored = []
        for candidate, empathy_state in empathy_results:
            F = candidate.feasibility_result.feasibility_index if candidate.feasibility_result else 0.5
            E = empathy_state.empathy_score
            
            # 複合スコア: 0.6*可行性 + 0.4*共感
            combined_score = 0.6 * F + 0.4 * E
            scored.append((candidate, empathy_state, combined_score))
        
        # 最高スコアの選択
        scored.sort(key=lambda x: x[2], reverse=True)
        best_candidate, best_empathy, best_score = scored[0]
        
        if self.config.verbose:
            logger.info(f"  Selected: [{best_candidate.id}] score={best_score:.3f}")
            logger.info(f"    Text: {best_candidate.text[:60]}...")
        
        return best_candidate
    
    def update_parameters(self,
                         turn: ConversationTurn,
                         selected: ResponseCandidate
                         ) -> ParameterState:
        """
        Step 6: パラメータ更新
        
        §10: 最小侵襲更新則
        """
        # フィードバック情報の構築
        coherence_score = 0.7  # シミュレーション
        exploration_reward = selected.score if selected else 0.5
        ethical_violation = (selected.feasibility_result and 
                           selected.feasibility_result.status == FeasibilityStatus.BLOCKED)
        feasibility_index = (selected.feasibility_result.feasibility_index 
                           if selected.feasibility_result else 0.5)
        
        new_state = self.parameter_controller.update_from_feedback(
            coherence_score=coherence_score,
            exploration_reward=exploration_reward,
            ethical_violation=ethical_violation,
            feasibility_index=feasibility_index
        )
        
        if self.config.verbose:
            logger.info(f"  Updated parameters: {new_state}")
        
        return new_state
    
    def run_turn(self, user_input: str) -> ConversationTurn:
        """
        1ターンの完全な実行
        """
        turn_id = len(self.history) + 1
        
        # パイプライン実行
        turn = self.process_input(user_input, turn_id)
        
        candidates = self.generate_candidates(turn)
        
        filtered = self.evaluate_feasibility(candidates, turn)
        
        if filtered:
            empathy_results = self.apply_empathy(filtered, turn)
            selected = self.select_response(empathy_results, turn)
        else:
            # 全候補が不可行の場合
            logger.warning("  No feasible candidates - using fallback")
            selected = self.select_response([], turn)
        
        # ターン情報の更新
        turn.selected_response = selected.text
        turn.response_embedding = selected.embedding
        turn.feasibility_index = (selected.feasibility_result.feasibility_index 
                                 if selected.feasibility_result else 0.5)
        
        # パラメータ更新
        new_params = self.update_parameters(turn, selected)
        turn.parameter_state = new_params
        
        # 履歴更新
        self.history.append(turn)
        self._history_embeddings.append(selected.embedding)
        
        return turn
    
    def run_conversation(self, user_inputs: List[str]) -> List[ConversationTurn]:
        """
        複数ターンの会話実行
        """
        results = []
        for user_input in user_inputs:
            turn = self.run_turn(user_input)
            results.append(turn)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """会話サマリーの取得"""
        if not self.history:
            return {}
        
        feasibility_scores = [t.feasibility_index for t in self.history]
        
        return {
            'num_turns': len(self.history),
            'avg_feasibility': np.mean(feasibility_scores),
            'min_feasibility': np.min(feasibility_scores),
            'max_feasibility': np.max(feasibility_scores),
            'final_parameters': self.parameter_controller.current_state,
        }


# =============================================================================
# Full Integration Demo
# =============================================================================
def run_full_demo():
    """完全統合デモの実行"""
    print("=" * 70)
    print("ReIG2/twinRIG World-Building Quantum Cosmology")
    print("LLM Integration Demo")
    print("=" * 70)
    
    # パイプラインの初期化
    config = DemoConfig(
        embedding_dim=128,
        num_candidates=5,
        pfh_weight=0.7,
        verbose=True
    )
    
    pipeline = ReIG2ResponsePipeline(config)
    
    print(f"\n初期パラメータ: {pipeline.parameter_controller.current_state}")
    
    # シミュレーション会話
    user_inputs = [
        "Hello, I need some help understanding a complex topic.",
        "Can you explain how machine learning works?",
        "I'm feeling a bit overwhelmed by all this information.",
        "Thank you for your patience. This is very helpful!",
        "What other resources would you recommend?",
    ]
    
    print("\n" + "-" * 70)
    print("Starting Conversation Simulation")
    print("-" * 70)
    
    # 会話実行
    turns = pipeline.run_conversation(user_inputs)
    
    # サマリー
    print("\n" + "=" * 70)
    print("Conversation Summary")
    print("=" * 70)
    
    summary = pipeline.get_summary()
    print(f"\n総ターン数: {summary['num_turns']}")
    print(f"平均可行性指数: {summary['avg_feasibility']:.4f}")
    print(f"可行性指数範囲: [{summary['min_feasibility']:.4f}, {summary['max_feasibility']:.4f}]")
    print(f"\n最終パラメータ: {summary['final_parameters']}")
    
    # パラメータ推移
    print("\n--- Parameter Evolution ---")
    trajectory = pipeline.parameter_controller.manager.get_trajectory()
    print(f"τ: {trajectory[0,0]:.3f} → {trajectory[-1,0]:.3f}")
    print(f"ε: {trajectory[0,1]:.3f} → {trajectory[-1,1]:.3f}")
    print(f"PFH: {trajectory[0,2]:.3f} → {trajectory[-1,2]:.3f}")
    
    # 安定性チェック
    gamma = pipeline.parameter_controller.manager.decay_checker.estimate_gamma()
    if gamma:
        print(f"\n推定収束率 γ: {gamma:.4f}")
        print(f"安定性条件: {'✓ 満足 (γ < 1)' if gamma < 1 else '✗ 不満足'}")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


# =============================================================================
# Component Test
# =============================================================================
def run_component_tests():
    """各コンポーネントの個別テスト"""
    print("=" * 70)
    print("Component Tests")
    print("=" * 70)
    
    dim = 64
    
    # 1. LLM Feasibility
    print("\n[1] LLM Feasibility Evaluator")
    evaluator = LLMFeasibilityEvaluator()
    candidates = [
        ResponseCandidate(id="c1", text="I'll help you with that."),
        ResponseCandidate(id="c2", text="That's harmful and I can't help."),
    ]
    context_emb = np.random.randn(128)
    
    for c in candidates:
        result = evaluator.evaluate_candidate(c, context_emb)
        print(f"  {c.id}: {result.status.name}, F={result.feasibility_index:.3f}")
    
    # 2. Parameter Controller
    print("\n[2] Parameter Controller")
    controller = IntegratedParameterController()
    print(f"  Initial: {controller.current_state}")
    
    for i in range(3):
        state = controller.update_from_feedback(
            coherence_score=0.7 + i*0.1,
            exploration_reward=0.5,
            ethical_violation=False,
            feasibility_index=0.7
        )
    print(f"  After 3 updates: {state}")
    
    # 3. Mirror Operator
    print("\n[3] Mirror Operator (Empathy)")
    processor = EmpathyProcessor(dim=dim)
    h_self = np.random.randn(dim)
    h_other = np.random.randn(dim)
    
    empathy_state = processor.process(h_self, h_other)
    print(f"  Empathy Score: {empathy_state.empathy_score:.4f}")
    print(f"  Mode: {empathy_state.mode.name}")
    print(f"  Balance (self/other): {empathy_state.balance:.4f}")
    
    print("\n" + "=" * 70)
    print("All Component Tests Passed")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_component_tests()
    else:
        run_full_demo()
