"""
ReIG2/twinRIG World-Building Quantum Cosmology: LLM Parameters Module
=====================================================================
適応的パラメータ更新（τ:長期整合性、ε:探索性、PFH:倫理重み）

対応セクション: §10, §9.4
- 最小侵襲更新則
- パラメータの動的調整
- 安定性減衰定理に基づく収束保証

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Theory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
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
RealVector = NDArray[np.float64]
RealMatrix = NDArray[np.float64]  # 2D array (e.g., trajectory shape: (n_steps, 3))


# =============================================================================
# Parameter State
# =============================================================================
@dataclass
class ParameterState:
    """
    パラメータ状態 θ = (τ, ε, PFH)
    
    §1.2: 共鳴パラメータの状態表現
    """
    tau: float = 0.5        # 未来寄与度 τ ∈ [0,1]
    epsilon: float = 0.3    # 探索度 ε ∈ [0,1]
    pfh: float = 0.7        # PFH倫理重み ∈ [0,1]
    
    # 更新履歴
    timestamp: float = 0.0
    update_count: int = 0
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self) -> None:
        """範囲検証"""
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0,1], got {self.epsilon}")
        if not 0.0 <= self.pfh <= 1.0:
            raise ValueError(f"pfh must be in [0,1], got {self.pfh}")
    
    def to_vector(self) -> RealVector:
        """ベクトル表現"""
        return np.array([self.tau, self.epsilon, self.pfh])
    
    @classmethod
    def from_vector(cls, vec: RealVector, 
                   timestamp: float = 0.0) -> 'ParameterState':
        """ベクトルからの構築"""
        return cls(
            tau=float(np.clip(vec[0], 0, 1)),
            epsilon=float(np.clip(vec[1], 0, 1)),
            pfh=float(np.clip(vec[2], 0, 1)),
            timestamp=timestamp
        )
    
    def copy(self, **kwargs) -> 'ParameterState':
        """部分更新コピー"""
        return ParameterState(
            tau=kwargs.get('tau', self.tau),
            epsilon=kwargs.get('epsilon', self.epsilon),
            pfh=kwargs.get('pfh', self.pfh),
            timestamp=kwargs.get('timestamp', self.timestamp),
            update_count=kwargs.get('update_count', self.update_count)
        )
    
    def distance(self, other: 'ParameterState') -> float:
        """パラメータ距離"""
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))
    
    def __repr__(self) -> str:
        return f"θ(τ={self.tau:.3f}, ε={self.epsilon:.3f}, PFH={self.pfh:.3f})"


# =============================================================================
# Update Context
# =============================================================================
@dataclass
class UpdateContext:
    """
    更新コンテキスト
    
    パラメータ更新に必要な状態情報
    """
    # 可行性情報
    feasibility_index: float = 0.5      # F(ρ;θ)
    constraint_violation: float = 0.0   # 制約違反度
    
    # 履歴情報
    recent_feasibility: List[float] = field(default_factory=list)
    recent_costs: List[float] = field(default_factory=list)
    
    # 会話情報
    conversation_length: int = 0
    user_satisfaction: float = 0.5      # [0,1]
    
    # メタ情報
    stability_trend: float = 0.0        # 正=改善、負=悪化
    ethical_concern_level: float = 0.0  # [0,1]
    
    def update_trends(self) -> None:
        """トレンド計算"""
        if len(self.recent_feasibility) >= 2:
            # 指数移動平均のトレンド
            recent = self.recent_feasibility[-5:]
            if len(recent) >= 2:
                self.stability_trend = recent[-1] - recent[0]


# =============================================================================
# Update Strategies
# =============================================================================
class UpdateStrategy(Enum):
    """更新戦略タイプ"""
    MINIMAL_INTERVENTION = auto()   # 最小侵襲 (§10)
    GRADIENT_BASED = auto()         # 勾配ベース
    ADAPTIVE = auto()               # 適応的
    CONSERVATIVE = auto()           # 保守的
    EXPLORATORY = auto()            # 探索的


# =============================================================================
# Abstract Updater
# =============================================================================
class ParameterUpdater(ABC):
    """パラメータ更新器の抽象基底クラス"""
    
    @abstractmethod
    def compute_update(self, 
                      current: ParameterState,
                      context: UpdateContext) -> ParameterState:
        """更新計算"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """更新器名"""
        pass


# =============================================================================
# Minimal Intervention Updater
# =============================================================================
class MinimalInterventionUpdater(ParameterUpdater):
    """
    最小侵襲更新器 (§10)
    
    世界状態を保存しながらパラメータを最小限に更新
    
    θ_{n+1} = θ_n + α * Δθ
    where ||Δθ|| is minimized subject to F(ρ;θ_{n+1}) ≥ F(ρ;θ_n)
    """
    
    def __init__(self,
                 learning_rate: float = 0.05,
                 momentum: float = 0.9,
                 min_change: float = 0.001,
                 max_change: float = 0.1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_change = min_change
        self.max_change = max_change
        
        # モメンタム項
        self._velocity = np.zeros(3)
    
    def name(self) -> str:
        return "MinimalIntervention"
    
    def compute_update(self,
                      current: ParameterState,
                      context: UpdateContext) -> ParameterState:
        """
        最小侵襲更新の計算
        
        §10: 世界状態ρを固定し、パラメータθのみを最小限に変更
        """
        # 勾配推定（有限差分）
        gradient = self._estimate_gradient(current, context)
        
        # モメンタム更新
        self._velocity = (self.momentum * self._velocity + 
                         self.learning_rate * gradient)
        
        # 変化量の制限
        change_norm = np.linalg.norm(self._velocity)
        if change_norm > self.max_change:
            self._velocity = self._velocity * (self.max_change / change_norm)
        elif change_norm < self.min_change and change_norm > 0:
            self._velocity = self._velocity * (self.min_change / change_norm)
        
        # 新パラメータ
        new_vec = current.to_vector() + self._velocity
        new_vec = np.clip(new_vec, 0, 1)
        
        return ParameterState.from_vector(
            new_vec,
            timestamp=context.conversation_length
        )
    
    def _estimate_gradient(self,
                          current: ParameterState,
                          context: UpdateContext) -> RealVector:
        """
        勾配推定
        
        可行性指数を最大化する方向
        """
        gradient = np.zeros(3)
        
        # τ (未来寄与度)
        # 長期的な改善トレンドがあればτを増加
        if context.stability_trend > 0:
            gradient[0] = 0.1 * context.stability_trend
        else:
            gradient[0] = -0.05  # 不安定なら減少
        
        # ε (探索度)
        # 可行性が高ければ探索を増やす、低ければ減らす
        if context.feasibility_index > 0.7:
            gradient[1] = 0.1  # 探索増加
        elif context.feasibility_index < 0.3:
            gradient[1] = -0.1  # 探索減少（安定重視）
        
        # PFH (倫理重み)
        # 倫理的懸念があれば増加、なければ微減（効率化）
        if context.ethical_concern_level > 0.3:
            gradient[2] = 0.2 * context.ethical_concern_level
        elif context.constraint_violation < 0.1:
            gradient[2] = -0.02  # 軽微な緩和
        
        return gradient


# =============================================================================
# Adaptive Updater
# =============================================================================
class AdaptiveUpdater(ParameterUpdater):
    """
    適応的更新器 (§9.4)
    
    コンテキストに応じて更新戦略を切り替え
    """
    
    def __init__(self,
                 base_rate: float = 0.1,
                 adaptation_speed: float = 0.5):
        self.base_rate = base_rate
        self.adaptation_speed = adaptation_speed
        
        # 状態追跡
        self._consecutive_improvements = 0
        self._consecutive_declines = 0
    
    def name(self) -> str:
        return "Adaptive"
    
    def compute_update(self,
                      current: ParameterState,
                      context: UpdateContext) -> ParameterState:
        """
        適応的更新の計算
        """
        # 適応率の計算
        effective_rate = self._compute_adaptive_rate(context)
        
        # 目標方向の計算
        target_direction = self._compute_target_direction(current, context)
        
        # 更新
        delta = effective_rate * target_direction
        new_vec = current.to_vector() + delta
        new_vec = np.clip(new_vec, 0, 1)
        
        return ParameterState.from_vector(
            new_vec,
            timestamp=context.conversation_length
        )
    
    def _compute_adaptive_rate(self, context: UpdateContext) -> float:
        """適応的学習率"""
        # 連続改善/悪化の追跡
        if context.stability_trend > 0:
            self._consecutive_improvements += 1
            self._consecutive_declines = 0
        else:
            self._consecutive_declines += 1
            self._consecutive_improvements = 0
        
        # 適応率
        if self._consecutive_improvements > 3:
            # 順調なら加速
            return self.base_rate * (1 + self.adaptation_speed)
        elif self._consecutive_declines > 3:
            # 悪化が続くなら減速
            return self.base_rate * (1 - self.adaptation_speed * 0.5)
        
        return self.base_rate
    
    def _compute_target_direction(self,
                                  current: ParameterState,
                                  context: UpdateContext) -> RealVector:
        """目標方向の計算"""
        # 理想的なパラメータの推定
        ideal_tau = 0.5 + 0.3 * context.stability_trend
        ideal_epsilon = 0.3 * (1 + context.feasibility_index)
        ideal_pfh = max(0.5, context.ethical_concern_level + 0.5)
        
        ideal = np.array([ideal_tau, ideal_epsilon, ideal_pfh])
        ideal = np.clip(ideal, 0, 1)
        
        return ideal - current.to_vector()


# =============================================================================
# Conservative Updater
# =============================================================================
class ConservativeUpdater(ParameterUpdater):
    """
    保守的更新器
    
    安定性を優先した慎重な更新
    """
    
    def __init__(self, max_step: float = 0.02):
        self.max_step = max_step
    
    def name(self) -> str:
        return "Conservative"
    
    def compute_update(self,
                      current: ParameterState,
                      context: UpdateContext) -> ParameterState:
        """保守的更新"""
        delta = np.zeros(3)
        
        # 改善の余地がある場合のみ微調整
        if context.feasibility_index < 0.8:
            # PFHを少し上げる（安全側）
            delta[2] = self.max_step
        
        if context.constraint_violation > 0:
            # 制約違反があれば探索を減らす
            delta[1] = -self.max_step
        
        new_vec = current.to_vector() + delta
        new_vec = np.clip(new_vec, 0, 1)
        
        return ParameterState.from_vector(
            new_vec,
            timestamp=context.conversation_length
        )


# =============================================================================
# Decay Theorem Checker
# =============================================================================
class DecayTheoremChecker:
    """
    安定性減衰定理のチェッカー (§11)
    
    F(ρ_{n+1}) ≤ γ F(ρ_n) with γ < 1
    """
    
    def __init__(self, target_gamma: float = 0.95):
        self.target_gamma = target_gamma
        self._history: List[float] = []
    
    def add_observation(self, feasibility_index: float) -> None:
        """観測値を追加"""
        self._history.append(feasibility_index)
    
    def estimate_gamma(self) -> Optional[float]:
        """実効的なγを推定"""
        if len(self._history) < 2:
            return None
        
        # 連続する比率の計算
        ratios = []
        for i in range(1, len(self._history)):
            if self._history[i-1] > 1e-10:
                ratio = self._history[i] / self._history[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return None
        
        return float(np.mean(ratios))
    
    def is_stable(self) -> bool:
        """安定性条件のチェック"""
        gamma = self.estimate_gamma()
        if gamma is None:
            return True  # データ不足は安定と仮定
        return gamma <= self.target_gamma
    
    def convergence_estimate(self, threshold: float = 0.1) -> Optional[int]:
        """収束までの推定ステップ数"""
        gamma = self.estimate_gamma()
        if gamma is None or gamma >= 1:
            return None
        
        if len(self._history) == 0:
            return None
        
        current = self._history[-1]
        if current <= threshold:
            return 0
        
        # n = log(threshold/current) / log(gamma)
        steps = np.log(threshold / current) / np.log(gamma)
        return max(0, int(np.ceil(steps)))


# =============================================================================
# Parameter Manager
# =============================================================================
class ParameterManager:
    """
    パラメータ管理器
    
    パラメータの更新、履歴管理、安定性監視を統合
    """
    
    def __init__(self,
                 initial_state: Optional[ParameterState] = None,
                 strategy: UpdateStrategy = UpdateStrategy.MINIMAL_INTERVENTION):
        self.state = initial_state or ParameterState()
        self.strategy = strategy
        
        # 更新器の選択
        self._updaters = {
            UpdateStrategy.MINIMAL_INTERVENTION: MinimalInterventionUpdater(),
            UpdateStrategy.ADAPTIVE: AdaptiveUpdater(),
            UpdateStrategy.CONSERVATIVE: ConservativeUpdater(),
        }
        
        # 履歴
        self._history: List[ParameterState] = [self.state.copy()]
        
        # 安定性チェッカー
        self.decay_checker = DecayTheoremChecker()
    
    @property
    def current_updater(self) -> ParameterUpdater:
        return self._updaters.get(
            self.strategy,
            self._updaters[UpdateStrategy.MINIMAL_INTERVENTION]
        )
    
    def update(self, context: UpdateContext) -> ParameterState:
        """
        パラメータ更新
        
        Returns:
            更新後のParameterState
        """
        # 更新計算
        new_state = self.current_updater.compute_update(self.state, context)
        new_state.update_count = self.state.update_count + 1
        
        # 状態更新
        self.state = new_state
        self._history.append(new_state.copy())
        
        # 安定性追跡
        self.decay_checker.add_observation(context.feasibility_index)
        
        logger.info(f"Parameter update #{new_state.update_count}: {new_state}")
        
        return new_state
    
    def set_strategy(self, strategy: UpdateStrategy) -> None:
        """更新戦略の変更"""
        self.strategy = strategy
        logger.info(f"Update strategy changed to {strategy.name}")
    
    def auto_select_strategy(self, context: UpdateContext) -> UpdateStrategy:
        """
        コンテキストに応じた戦略の自動選択
        """
        if context.ethical_concern_level > 0.5:
            # 倫理的懸念が高い場合は保守的に
            return UpdateStrategy.CONSERVATIVE
        elif context.feasibility_index > 0.8:
            # 可行性が高い場合は適応的に
            return UpdateStrategy.ADAPTIVE
        else:
            # デフォルトは最小侵襲
            return UpdateStrategy.MINIMAL_INTERVENTION
    
    def get_history(self) -> List[ParameterState]:
        """履歴の取得"""
        return self._history.copy()
    
    def get_trajectory(self) -> RealMatrix:
        """
        パラメータ軌跡
        
        Returns:
            shape (n_steps, 3) の配列
        """
        return np.array([s.to_vector() for s in self._history])
    
    def reset(self, initial: Optional[ParameterState] = None) -> None:
        """リセット"""
        self.state = initial or ParameterState()
        self._history = [self.state.copy()]
        self.decay_checker = DecayTheoremChecker()


# =============================================================================
# Tau Controller (Long-term Coherence)
# =============================================================================
class TauController:
    """
    τ (未来寄与度) の専用コントローラ
    
    長期的な整合性と一貫性を管理
    """
    
    def __init__(self,
                 initial_tau: float = 0.5,
                 horizon: int = 10):
        self.tau = initial_tau
        self.horizon = horizon
        self._coherence_scores: List[float] = []
    
    def add_coherence_observation(self, score: float) -> None:
        """整合性スコアを追加"""
        self._coherence_scores.append(score)
        if len(self._coherence_scores) > self.horizon:
            self._coherence_scores.pop(0)
    
    def compute_suggested_tau(self) -> float:
        """
        推奨τ値の計算
        
        整合性が高ければτを上げ、低ければ下げる
        """
        if not self._coherence_scores:
            return self.tau
        
        avg_coherence = np.mean(self._coherence_scores)
        trend = 0.0
        if len(self._coherence_scores) >= 2:
            trend = self._coherence_scores[-1] - self._coherence_scores[0]
        
        # 推奨τ
        suggested = self.tau + 0.1 * (avg_coherence - 0.5) + 0.05 * trend
        return float(np.clip(suggested, 0.1, 0.9))


# =============================================================================
# Epsilon Controller (Exploration)
# =============================================================================
class EpsilonController:
    """
    ε (探索度) の専用コントローラ
    
    探索/活用のバランスを管理
    """
    
    def __init__(self,
                 initial_epsilon: float = 0.3,
                 decay_rate: float = 0.995,
                 min_epsilon: float = 0.05):
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self._exploration_rewards: List[float] = []
    
    def decay(self) -> float:
        """指数減衰"""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay_rate
        )
        return self.epsilon
    
    def add_exploration_reward(self, reward: float) -> None:
        """探索報酬を追加"""
        self._exploration_rewards.append(reward)
    
    def adaptive_epsilon(self) -> float:
        """
        適応的ε計算
        
        探索が報われていればεを維持/増加
        """
        if len(self._exploration_rewards) < 5:
            return self.epsilon
        
        recent = self._exploration_rewards[-5:]
        avg_reward = np.mean(recent)
        
        if avg_reward > 0.5:
            # 探索が成功している
            return min(0.8, self.epsilon * 1.1)
        else:
            return self.decay()


# =============================================================================
# PFH Controller (Ethics)
# =============================================================================
class PFHController:
    """
    PFH (倫理重み) の専用コントローラ
    
    倫理的制約の強度を管理
    """
    
    def __init__(self,
                 initial_pfh: float = 0.7,
                 min_pfh: float = 0.5):
        self.pfh = initial_pfh
        self.min_pfh = min_pfh
        self._ethical_violations: List[bool] = []
    
    def record_violation(self, violated: bool) -> None:
        """倫理違反を記録"""
        self._ethical_violations.append(violated)
    
    def compute_suggested_pfh(self) -> float:
        """
        推奨PFH値の計算
        
        違反があればPFHを上げる
        """
        if not self._ethical_violations:
            return self.pfh
        
        recent = self._ethical_violations[-10:]
        violation_rate = sum(recent) / len(recent)
        
        if violation_rate > 0.1:
            # 違反が多い場合は厳格化
            return min(1.0, self.pfh + 0.1 * violation_rate)
        elif violation_rate == 0 and self.pfh > self.min_pfh:
            # 違反がなければ少し緩和
            return max(self.min_pfh, self.pfh - 0.02)
        
        return self.pfh


# =============================================================================
# Integrated Parameter Controller
# =============================================================================
class IntegratedParameterController:
    """
    統合パラメータコントローラ
    
    τ, ε, PFH の各コントローラを統合管理
    """
    
    def __init__(self,
                 initial_state: Optional[ParameterState] = None):
        state = initial_state or ParameterState()
        
        self.tau_controller = TauController(state.tau)
        self.epsilon_controller = EpsilonController(state.epsilon)
        self.pfh_controller = PFHController(state.pfh)
        
        self.manager = ParameterManager(state)
    
    def update_from_feedback(self,
                            coherence_score: float,
                            exploration_reward: float,
                            ethical_violation: bool,
                            feasibility_index: float
                            ) -> ParameterState:
        """
        フィードバックに基づく統合更新
        """
        # 各コントローラに情報を供給
        self.tau_controller.add_coherence_observation(coherence_score)
        self.epsilon_controller.add_exploration_reward(exploration_reward)
        self.pfh_controller.record_violation(ethical_violation)
        
        # 推奨値の取得
        suggested_tau = self.tau_controller.compute_suggested_tau()
        suggested_epsilon = self.epsilon_controller.adaptive_epsilon()
        suggested_pfh = self.pfh_controller.compute_suggested_pfh()
        
        # コンテキストの構築
        context = UpdateContext(
            feasibility_index=feasibility_index,
            constraint_violation=1.0 if ethical_violation else 0.0,
            stability_trend=coherence_score - 0.5,
            ethical_concern_level=suggested_pfh - 0.5
        )
        
        # パラメータマネージャで更新
        # 推奨値を参考にしつつ、最小侵襲則で更新
        current = self.manager.state
        blended = ParameterState(
            tau=0.7 * current.tau + 0.3 * suggested_tau,
            epsilon=0.7 * current.epsilon + 0.3 * suggested_epsilon,
            pfh=0.7 * current.pfh + 0.3 * suggested_pfh
        )
        
        self.manager.state = blended
        return self.manager.update(context)
    
    @property
    def current_state(self) -> ParameterState:
        return self.manager.state


# =============================================================================
# Demo Function
# =============================================================================
def demo_parameter_update():
    """パラメータ更新のデモ"""
    print("=" * 60)
    print("ReIG2/twinRIG LLM Parameter Update Demo")
    print("=" * 60)
    
    # 統合コントローラの初期化
    controller = IntegratedParameterController()
    
    print(f"\n初期パラメータ: {controller.current_state}")
    
    # シミュレーションループ
    print("\n--- 更新シミュレーション ---")
    
    scenarios = [
        # (coherence, exploration_reward, ethical_violation, feasibility)
        (0.8, 0.6, False, 0.75),
        (0.7, 0.5, False, 0.70),
        (0.6, 0.4, True, 0.50),   # 倫理違反
        (0.5, 0.3, False, 0.55),
        (0.7, 0.6, False, 0.65),
        (0.8, 0.7, False, 0.75),
        (0.9, 0.8, False, 0.85),
    ]
    
    for i, (coh, exp, vio, feas) in enumerate(scenarios):
        state = controller.update_from_feedback(coh, exp, vio, feas)
        vio_str = "!" if vio else " "
        print(f"Step {i+1}: {state} | F={feas:.2f} {vio_str}")
    
    # 安定性チェック
    print("\n--- 安定性解析 ---")
    gamma = controller.manager.decay_checker.estimate_gamma()
    if gamma:
        print(f"推定γ: {gamma:.4f}")
        print(f"安定条件(γ<1): {'満足' if gamma < 1 else '不満足'}")
    
    # 軌跡
    print("\n--- パラメータ軌跡 ---")
    trajectory = controller.manager.get_trajectory()
    print(f"形状: {trajectory.shape}")
    print(f"τ範囲: [{trajectory[:,0].min():.3f}, {trajectory[:,0].max():.3f}]")
    print(f"ε範囲: [{trajectory[:,1].min():.3f}, {trajectory[:,1].max():.3f}]")
    print(f"PFH範囲: [{trajectory[:,2].min():.3f}, {trajectory[:,2].max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    demo_parameter_update()
