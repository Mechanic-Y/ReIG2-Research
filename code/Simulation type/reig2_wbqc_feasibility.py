"""
ReIG2/twinRIG: World-Building Quantum Channels
Feasibility Module - 可行性指数と制約関数

論文セクション対応: §6, §8, §9

このモジュールは以下を提供します：
- 可行行為集合 A(ρ) の定義と計算
- 制約関数 C（安定性＋倫理＋資源）
- 可行性指数 F(ρ; θ)
- 射影演算子 Hinst, Πharm の具体的構成
- 回復ダイナミクス

Author: Mechanic-Y (Yasuyuki Wakita)
Framework: ReIG2/twinRIG Integrated Complete
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Tuple, Optional, Set, Callable, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

from reig2_wbqc_core import (
    WorldState, Hamiltonian, ResonanceParameters, ThresholdConfig, CostWeights,
    CPTPMap, IdentityMap, UnitaryMap, HilbertSpaceConfig, HBAR
)


# ==============================================================================
# 射影演算子の構成（§8）
# ==============================================================================

@dataclass
class ProjectionOperators:
    """
    射影演算子の構成（§8）
    
    論文 §8 より：
    - Hinst: 不安定性ハミルトニアン
    - Πharm: 有害射影演算子
    """
    dim: int
    
    # 安定集合・許容集合のインデックス
    stable_indices: List[int] = field(default_factory=list)
    ethical_indices: List[int] = field(default_factory=list)
    
    # 重み係数
    alpha_S: float = 1.0  # 低安定射影の重み
    alpha_C: float = 0.5  # 文脈破綻射影の重み
    alpha_Q: float = 0.3  # 問いぼやけ射影の重み
    beta_E: float = 1.0   # 倫理空間の重み
    
    def __post_init__(self):
        """デフォルトの安定/許容集合を設定"""
        if not self.stable_indices:
            # デフォルト: 最初の半分が安定
            self.stable_indices = list(range(self.dim // 2))
        
        if not self.ethical_indices:
            # デフォルト: 最初の3/4が倫理的に許容
            self.ethical_indices = list(range(3 * self.dim // 4))
    
    def build_instability_hamiltonian(self) -> np.ndarray:
        """
        不安定性ハミルトニアン Hinst を構成（§8.2）
        
        Hinst = αS·Π̃↓_S + αC·Π̃break_C + αQ·Π̃blur_Q
        
        最小形: Hinst = αS·Π̃↓_S
        """
        H_inst = np.zeros((self.dim, self.dim), dtype=complex)
        
        # 低安定射影 Π↓_S（§8.2 (A)）
        # 安定集合に含まれない状態への射影
        unstable_indices = [i for i in range(self.dim) if i not in self.stable_indices]
        for i in unstable_indices:
            H_inst[i, i] += self.alpha_S
        
        return H_inst
    
    def build_harm_projection(self) -> np.ndarray:
        """
        有害射影 Πharm を構成（§8.3）
        
        Πharm = Π̃harm_E
        
        倫理空間での禁止領域への射影
        """
        Pi_harm = np.zeros((self.dim, self.dim), dtype=complex)
        
        # 許容集合に含まれない状態への射影
        harmful_indices = [i for i in range(self.dim) if i not in self.ethical_indices]
        for i in harmful_indices:
            Pi_harm[i, i] = 1.0
        
        return Pi_harm


class InstabilityHamiltonian:
    """
    不安定性ハミルトニアン Hinst の詳細実装
    
    論文 §8.2 より：
    Hinst = αS·Π̃↓_S + αC·Π̃break_C + αQ·Π̃blur_Q
    """
    
    def __init__(self, hs_config: HilbertSpaceConfig):
        """
        Args:
            hs_config: 全空間構成
        """
        self.hs_config = hs_config
        self.dim = hs_config.dim_total
        
        # 各サブスペースの不安定性射影を構築
        self._build_projections()
    
    def _build_projections(self):
        """各サブスペースの射影を構築"""
        # 簡略化：対角行列として構成
        self.Pi_low_stability = np.zeros(self.dim, dtype=float)
        self.Pi_context_break = np.zeros(self.dim, dtype=float)
        self.Pi_question_blur = np.zeros(self.dim, dtype=float)
        
        # 安定性空間での低安定領域（後半のインデックス）
        dim_S = self.hs_config.get_subspace_dim('S')
        n_unstable = dim_S // 2
        
        # 全空間でのインデックスマッピング（簡略化）
        # 実際には複雑なテンソル積構造を考慮する必要がある
        for i in range(self.dim // 2, self.dim):
            self.Pi_low_stability[i] = 1.0
    
    def get_matrix(self, 
                  alpha_S: float = 1.0,
                  alpha_C: float = 0.5,
                  alpha_Q: float = 0.3) -> np.ndarray:
        """
        Hinst 行列を取得
        """
        H = np.diag(
            alpha_S * self.Pi_low_stability +
            alpha_C * self.Pi_context_break +
            alpha_Q * self.Pi_question_blur
        ).astype(complex)
        return H


class HarmProjection:
    """
    有害射影 Πharm の詳細実装
    
    論文 §8.3 より：
    Πharm = βE·Π̃harm_E + βS·Π̃harm_S
    """
    
    def __init__(self, hs_config: HilbertSpaceConfig):
        """
        Args:
            hs_config: 全空間構成
        """
        self.hs_config = hs_config
        self.dim = hs_config.dim_total
        
        self._build_projections()
    
    def _build_projections(self):
        """有害領域の射影を構築"""
        # 簡略化：対角行列として構成
        self.Pi_harm_ethics = np.zeros(self.dim, dtype=float)
        self.Pi_harm_stability = np.zeros(self.dim, dtype=float)
        
        # 倫理的に禁止される領域（最後の1/4）
        n_harmful = self.dim // 4
        for i in range(self.dim - n_harmful, self.dim):
            self.Pi_harm_ethics[i] = 1.0
    
    def get_matrix(self,
                  beta_E: float = 1.0,
                  beta_S: float = 0.5) -> np.ndarray:
        """
        Πharm 行列を取得
        """
        Pi = np.diag(
            beta_E * self.Pi_harm_ethics +
            beta_S * self.Pi_harm_stability
        ).astype(complex)
        return Pi


# ==============================================================================
# 制約関数 C（§6.3）
# ==============================================================================

class ConstraintFunction:
    """
    制約関数 C（§6.3）
    
    C(ρ'; ρ) = wS·CS(ρ') + wE·CE(ρ') + wR·CR(E)
    
    - CS: 安定性コスト = Tr[Hinst·ρ']
    - CE: 倫理コスト = PFH·Tr[Πharm·ρ']
    - CR: 資源コスト = α·depth(E) + β·mem(E)
    """
    
    def __init__(self,
                 H_inst: np.ndarray,
                 Pi_harm: np.ndarray,
                 weights: Optional[CostWeights] = None,
                 params: Optional[ResonanceParameters] = None):
        """
        Args:
            H_inst: 不安定性ハミルトニアン
            Pi_harm: 有害射影演算子
            weights: コスト重み
            params: 共鳴パラメータ（PFH用）
        """
        self.H_inst = H_inst
        self.Pi_harm = Pi_harm
        self.weights = weights or CostWeights()
        self.params = params or ResonanceParameters()
    
    def stability_cost(self, state: WorldState) -> float:
        """
        安定性コスト CS(ρ') = Tr[Hinst·ρ']
        
        世界が破綻・発散しないことを保証
        """
        return float(np.real(state.expectation(self.H_inst)))
    
    def ethics_cost(self, state: WorldState) -> float:
        """
        倫理コスト CE(ρ') = PFH·Tr[Πharm·ρ']
        
        許容できない遷移を禁止
        """
        harm_expectation = float(np.real(state.expectation(self.Pi_harm)))
        return self.params.PFH * harm_expectation
    
    def resource_cost(self, 
                     action: Optional[CPTPMap] = None,
                     depth: int = 1,
                     memory: int = 1) -> float:
        """
        資源コスト CR(E) = α·depth(E) + β·mem(E)
        
        現実的な制約を反映
        """
        return (self.weights.alpha_depth * depth + 
                self.weights.beta_memory * memory)
    
    def total_cost(self,
                  new_state: WorldState,
                  original_state: Optional[WorldState] = None,
                  action: Optional[CPTPMap] = None,
                  depth: int = 1,
                  memory: int = 1) -> float:
        """
        総制約コストを計算
        
        C(ρ'; ρ) = wS·CS(ρ') + wE·CE(ρ') + wR·CR(E)
        """
        C_S = self.stability_cost(new_state)
        C_E = self.ethics_cost(new_state)
        C_R = self.resource_cost(action, depth, memory)
        
        return (self.weights.w_stability * C_S +
                self.weights.w_ethics * C_E +
                self.weights.w_resource * C_R)
    
    def is_feasible(self, 
                   new_state: WorldState,
                   threshold: float) -> bool:
        """行為が可行かどうかを判定"""
        return self.total_cost(new_state) <= threshold


# ==============================================================================
# 可行行為集合 A(ρ)（§6.2）
# ==============================================================================

class ActionCandidate:
    """行為の候補"""
    
    def __init__(self,
                 cptp_map: CPTPMap,
                 name: str = "action",
                 depth: int = 1,
                 memory: int = 1):
        self.cptp_map = cptp_map
        self.name = name
        self.depth = depth
        self.memory = memory
    
    def apply(self, state: WorldState) -> WorldState:
        """行為を適用"""
        return self.cptp_map.apply(state)


class FeasibleActionSet:
    """
    可行行為集合 A(ρ)（§6.2）
    
    定義 6.1:
    A(ρ) := {Ea | Ea : D(Hfull) → D(Hfull), Ea は CPTP, C(Ea(ρ)) ≤ Θ}
    
    解釈：A(ρ) is the set of all actions that do not destroy the world.
    """
    
    def __init__(self,
                 current_state: WorldState,
                 constraint: ConstraintFunction,
                 threshold_config: Optional[ThresholdConfig] = None,
                 params: Optional[ResonanceParameters] = None):
        """
        Args:
            current_state: 現在の世界状態
            constraint: 制約関数
            threshold_config: 閾値設定
            params: 共鳴パラメータ
        """
        self.current_state = current_state
        self.constraint = constraint
        self.threshold_config = threshold_config or ThresholdConfig()
        self.params = params or ResonanceParameters()
        
        # 動的閾値を計算
        self.Theta = self._compute_threshold()
        
        # 候補行為のリスト
        self._candidates: List[ActionCandidate] = []
        self._feasible_actions: List[ActionCandidate] = []
        
        # 恒等写像は常に候補
        self._add_identity()
    
    def _compute_threshold(self) -> float:
        """動的閾値 Θ を計算"""
        return self.threshold_config.compute_delta(self.params) + \
               self.threshold_config.Theta
    
    def _add_identity(self):
        """恒等写像を追加"""
        identity = ActionCandidate(
            cptp_map=IdentityMap(self.current_state.dim),
            name="identity",
            depth=0,
            memory=0
        )
        self._candidates.append(identity)
    
    def add_candidate(self, action: ActionCandidate):
        """候補行為を追加"""
        self._candidates.append(action)
    
    def add_unitary_action(self, 
                          hamiltonian: Hamiltonian,
                          time: float,
                          name: str = "unitary"):
        """ユニタリ行為を追加"""
        action = ActionCandidate(
            cptp_map=UnitaryMap(hamiltonian, time),
            name=name,
            depth=1,
            memory=1
        )
        self._candidates.append(action)
    
    def evaluate_feasibility(self) -> List[ActionCandidate]:
        """
        全候補行為の可行性を評価
        
        Returns:
            可行行為のリスト
        """
        self._feasible_actions = []
        
        for action in self._candidates:
            new_state = action.apply(self.current_state)
            cost = self.constraint.total_cost(
                new_state,
                self.current_state,
                action.cptp_map,
                action.depth,
                action.memory
            )
            
            if cost <= self.Theta:
                self._feasible_actions.append(action)
        
        return self._feasible_actions
    
    def get_feasible_actions(self) -> List[ActionCandidate]:
        """可行行為集合を取得"""
        if not self._feasible_actions:
            self.evaluate_feasibility()
        return self._feasible_actions
    
    def is_agency_free(self) -> bool:
        """
        行為不能状態かどうかを判定（§6.5）
        
        定義 6.2: A(ρ) = {id} のとき行為不能状態
        """
        feasible = self.get_feasible_actions()
        return len(feasible) == 1 and feasible[0].name == "identity"
    
    def get_minimum_cost_action(self) -> Tuple[ActionCandidate, float]:
        """最小コストの行為を取得"""
        feasible = self.get_feasible_actions()
        
        if not feasible:
            # 全て不可行の場合は恒等写像を返す
            identity = self._candidates[0]
            cost = self.constraint.total_cost(identity.apply(self.current_state))
            return identity, cost
        
        min_cost = float('inf')
        best_action = feasible[0]
        
        for action in feasible:
            new_state = action.apply(self.current_state)
            cost = self.constraint.total_cost(new_state)
            if cost < min_cost:
                min_cost = cost
                best_action = action
        
        return best_action, min_cost
    
    def __len__(self) -> int:
        return len(self.get_feasible_actions())
    
    def __repr__(self) -> str:
        n_feasible = len(self.get_feasible_actions())
        n_total = len(self._candidates)
        return f"FeasibleActionSet(feasible={n_feasible}/{n_total}, Θ={self.Theta:.4f})"


# ==============================================================================
# 可行性指数 F(ρ; θ)（§9）
# ==============================================================================

class FeasibilityIndex:
    """
    可行性指数 F(ρ; θ)（§9.1）
    
    定義 9.1:
    F(ρ; θ) := inf_{E∈E} C(E(ρ); ρ, θ)
    
    意味：F measures how difficult it is for the agent to perform any admissible action.
    - F(ρ; θ) ≤ Θ：何かできる
    - F(ρ; θ) > Θ：実質的に行為不能
    """
    
    def __init__(self,
                 constraint: ConstraintFunction,
                 action_set: Optional[FeasibleActionSet] = None):
        """
        Args:
            constraint: 制約関数
            action_set: 可行行為集合（候補を含む）
        """
        self.constraint = constraint
        self.action_set = action_set
    
    def compute(self, 
               state: WorldState,
               params: ResonanceParameters,
               candidates: Optional[List[ActionCandidate]] = None) -> float:
        """
        可行性指数を計算
        
        F(ρ; θ) = inf_E C(E(ρ); ρ, θ)
        """
        if candidates is None and self.action_set is not None:
            candidates = self.action_set._candidates
        
        if candidates is None:
            # 候補がない場合は恒等写像のみ
            candidates = [ActionCandidate(
                IdentityMap(state.dim), "identity", 0, 0
            )]
        
        min_cost = float('inf')
        
        for action in candidates:
            new_state = action.apply(state)
            cost = self.constraint.total_cost(
                new_state, state, action.cptp_map,
                action.depth, action.memory
            )
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def compute_gradient(self,
                        state: WorldState,
                        params: ResonanceParameters,
                        delta: float = 1e-6) -> np.ndarray:
        """
        可行性指数の勾配（パラメータに関して）を数値的に計算
        
        ∇_θ F(ρ; θ)
        """
        grad = np.zeros(3)  # (τ, ε, PFH)
        
        F0 = self.compute(state, params)
        
        # τ方向
        params_tau = ResonanceParameters(
            tau=params.tau + delta, epsilon=params.epsilon, PFH=params.PFH
        )
        grad[0] = (self.compute(state, params_tau) - F0) / delta
        
        # ε方向
        params_eps = ResonanceParameters(
            tau=params.tau, epsilon=params.epsilon + delta, PFH=params.PFH
        )
        grad[1] = (self.compute(state, params_eps) - F0) / delta
        
        # PFH方向
        params_pfh = ResonanceParameters(
            tau=params.tau, epsilon=params.epsilon, PFH=params.PFH + delta
        )
        grad[2] = (self.compute(state, params_pfh) - F0) / delta
        
        return grad


# ==============================================================================
# 回復ダイナミクス（§9.3）
# ==============================================================================

class RecoveryDynamics:
    """
    回復ダイナミクス（§9.3）
    
    可行性は「ある／ない」ではなく、世界状態の変化とともに回復する量として扱える。
    
    (i) 受動回復（環境・時間による回復）
    (ii) 能動回復（「できる範囲で」の小さな行為）
    """
    
    def __init__(self,
                 constraint: ConstraintFunction,
                 threshold_config: ThresholdConfig):
        """
        Args:
            constraint: 制約関数
            threshold_config: 閾値設定
        """
        self.constraint = constraint
        self.threshold_config = threshold_config
        
        self.feasibility_index = FeasibilityIndex(constraint)
    
    def passive_recovery_step(self,
                             state: WorldState,
                             healing_rate: float = 0.1) -> WorldState:
        """
        受動回復（§9.3 (i)）
        
        非ユニタリな緩和（Lindblad）として
        d/dt Tr[Hinst·ρ(t)] ≤ 0
        となるよう設計
        """
        # 最大混合状態への緩和として実装
        rho_mixed = np.eye(state.dim, dtype=complex) / state.dim
        rho_new = (1 - healing_rate) * state.rho + healing_rate * rho_mixed
        return WorldState(rho_new)
    
    def active_recovery_step(self,
                            state: WorldState,
                            params: ResonanceParameters,
                            candidates: List[ActionCandidate]) -> Tuple[WorldState, ActionCandidate]:
        """
        能動回復（§9.3 (ii)）
        
        今出来る最小侵襲の行為を選び、次状態で CS が少し下がる
        """
        # 可行行為集合を構築
        action_set = FeasibleActionSet(
            current_state=state,
            constraint=self.constraint,
            threshold_config=self.threshold_config,
            params=params
        )
        
        for candidate in candidates:
            action_set.add_candidate(candidate)
        
        # 最小コスト行為を選択
        best_action, cost = action_set.get_minimum_cost_action()
        new_state = best_action.apply(state)
        
        return new_state, best_action
    
    def recovery_loop(self,
                     initial_state: WorldState,
                     params: ResonanceParameters,
                     candidates: List[ActionCandidate],
                     max_iterations: int = 100,
                     target_feasibility: float = 0.1) -> Dict[str, Any]:
        """
        回復ループを実行
        
        Stage 1: 内的調整（parameter recovery）
        Stage 2: 外的更新（world update）
        """
        state = initial_state
        history = {
            'states': [state],
            'feasibility': [],
            'actions': []
        }
        
        for i in range(max_iterations):
            # 可行性指数を計算
            F = self.feasibility_index.compute(state, params, candidates)
            history['feasibility'].append(F)
            
            # 目標達成チェック
            if F <= target_feasibility:
                print(f"Recovery achieved at iteration {i}")
                break
            
            # 能動回復
            state, action = self.active_recovery_step(state, params, candidates)
            history['states'].append(state)
            history['actions'].append(action.name)
        
        history['feasibility'] = np.array(history['feasibility'])
        return history


# ==============================================================================
# 具体例（§7）
# ==============================================================================

class DegenerationExamples:
    """
    行為集合縮退の具体例（§7）
    """
    
    @staticmethod
    def example1_fully_constrained(dim: int = 4) -> Tuple[WorldState, ConstraintFunction]:
        """
        Example 1: 完全拘束状態（§7.1）
        
        任意の介入が不安定性を増幅する状態
        結果: A(ρ_fragile) = {id}
        """
        # 非常に脆弱な状態を構築
        # 固有値が臨界的に分布
        eigenvalues = np.array([0.97, 0.01, 0.01, 0.01])
        rho = np.diag(eigenvalues.astype(complex))
        state = WorldState(rho)
        
        # 高い不安定性を持つハミルトニアン
        H_inst = np.diag([0.0, 10.0, 10.0, 10.0]).astype(complex)
        Pi_harm = np.zeros((dim, dim), dtype=complex)
        
        constraint = ConstraintFunction(H_inst, Pi_harm)
        
        return state, constraint
    
    @staticmethod
    def example2_operationally_powerless(dim: int = 4) -> Tuple[WorldState, ConstraintFunction]:
        """
        Example 2: 行為集合が事実上「空」に近い状態（§7.2）
        
        lim_{ε→0} A_ε(ρ) = {id}
        """
        # ほぼ全ての行為が制約を超える状態
        rho = np.eye(dim, dtype=complex) / dim
        state = WorldState(rho)
        
        # 全方向に高い不安定性
        H_inst = np.eye(dim, dtype=complex) * 100
        Pi_harm = np.eye(dim, dtype=complex) * 0.1
        
        constraint = ConstraintFunction(H_inst, Pi_harm)
        
        return state, constraint
    
    @staticmethod
    def example3_ethical_prohibition(dim: int = 4, 
                                    PFH: float = 100.0) -> Tuple[WorldState, ConstraintFunction, ResonanceParameters]:
        """
        Example 3: 倫理的禁止による縮退（§7.3）
        
        PFH → ∞ ⟹ A(ρ) → {id}
        """
        rho = np.eye(dim, dtype=complex) / dim
        state = WorldState(rho)
        
        H_inst = np.zeros((dim, dim), dtype=complex)
        # 大きな有害射影
        Pi_harm = np.eye(dim, dtype=complex) - np.diag([1, 0, 0, 0]).astype(complex)
        
        params = ResonanceParameters(tau=1.0, epsilon=0.1, PFH=PFH)
        constraint = ConstraintFunction(H_inst, Pi_harm, params=params)
        
        return state, constraint, params
    
    @staticmethod
    def example4_critical_slowing(dim: int = 4) -> Tuple[WorldState, ConstraintFunction]:
        """
        Example 4: 相転移直前の行為不能（§7.4）
        
        臨界近傍: C(ρ) → 0 ⟹ A(ρ) shrinks
        """
        # 臨界点付近の状態
        # 固有値が縮退に近い
        eigenvalues = np.array([0.25 + 0.001, 0.25, 0.25, 0.25 - 0.001])
        rho = np.diag(eigenvalues.astype(complex))
        state = WorldState(rho)
        
        # 臨界的なハミルトニアン
        # 小さな摂動で大きな応答
        H_inst = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex) * 50
        
        Pi_harm = np.zeros((dim, dim), dtype=complex)
        constraint = ConstraintFunction(H_inst, Pi_harm)
        
        return state, constraint


# ==============================================================================
# Tier0/Tier1 統合インターフェース（§13連携強化）
# ==============================================================================

class TierIntegration:
    """
    Tier0/Tier1統合インターフェース
    
    論文 §13 より：
    - Tier0: ゼロ化領域 Z（安定な参照点）
    - Tier1: 動的領域（可行行為が実行される空間）
    
    このクラスはfeasibilityモジュールとtier0モジュールの連携を強化する
    """
    
    def __init__(self,
                 constraint: ConstraintFunction,
                 threshold_config: ThresholdConfig,
                 dim: int):
        """
        Args:
            constraint: 制約関数
            threshold_config: 閾値設定
            dim: 状態空間次元
        """
        self.constraint = constraint
        self.threshold_config = threshold_config
        self.dim = dim
        self.feasibility_index = FeasibilityIndex(constraint)
        
        # ゼロ化参照状態（後で設定）
        self._zero_state: Optional[WorldState] = None
    
    def set_zero_state(self, zero_state: WorldState):
        """ゼロ化参照状態を設定"""
        self._zero_state = zero_state
    
    def compute_tier0_feasibility(self,
                                  params: ResonanceParameters,
                                  candidates: List[ActionCandidate]) -> float:
        """
        Tier0（ゼロ化状態）での可行性指数を計算
        
        F(ρZ; θ) → 0 が理想
        """
        if self._zero_state is None:
            # デフォルト: 最大混合状態
            self._zero_state = WorldState.maximally_mixed(self.dim)
        
        return self.feasibility_index.compute(self._zero_state, params, candidates)
    
    def compute_tier1_feasibility(self,
                                  state: WorldState,
                                  params: ResonanceParameters,
                                  candidates: List[ActionCandidate]) -> float:
        """
        Tier1（動的状態）での可行性指数を計算
        """
        return self.feasibility_index.compute(state, params, candidates)
    
    def tier_transition_cost(self,
                            state: WorldState,
                            params: ResonanceParameters,
                            candidates: List[ActionCandidate]) -> Dict[str, Any]:
        """
        Tier間遷移のコストを計算
        
        Tier1 → Tier0 への遷移コストと改善度を評価
        """
        F_tier1 = self.compute_tier1_feasibility(state, params, candidates)
        F_tier0 = self.compute_tier0_feasibility(params, candidates)
        
        # Tier0への遷移コスト（状態間距離）
        if self._zero_state is not None:
            transition_distance = state.trace_distance(self._zero_state)
        else:
            transition_distance = 0.0
        
        return {
            'F_tier1': F_tier1,
            'F_tier0': F_tier0,
            'improvement': F_tier1 - F_tier0,
            'transition_distance': transition_distance,
            'is_beneficial': F_tier0 < F_tier1
        }
    
    def recommend_tier(self,
                      state: WorldState,
                      params: ResonanceParameters,
                      candidates: List[ActionCandidate],
                      distance_weight: float = 0.5) -> str:
        """
        推奨Tierを判定
        
        Args:
            state: 現在状態
            params: 共鳴パラメータ
            candidates: 候補行為
            distance_weight: 遷移距離の重み
        
        Returns:
            "tier0" or "tier1"
        """
        costs = self.tier_transition_cost(state, params, candidates)
        
        # 加重コスト: F + w * distance
        cost_stay_tier1 = costs['F_tier1']
        cost_go_tier0 = costs['F_tier0'] + distance_weight * costs['transition_distance']
        
        if cost_go_tier0 < cost_stay_tier1:
            return "tier0"
        else:
            return "tier1"
    
    def compute_infimum_over_actions(self,
                                    state: WorldState,
                                    params: ResonanceParameters,
                                    candidates: List[ActionCandidate]) -> Dict[str, Any]:
        """
        行為集合上での下限を計算（論文の inf over E in E）
        
        inf_{E ∈ A(ρ)} C(E(ρ); ρ)
        
        厳密なinfの近似として候補集合上の最小値を返す
        """
        min_cost = float('inf')
        best_action = None
        all_costs = []
        
        for candidate in candidates:
            new_state = candidate.apply(state)
            cost = self.constraint.total_cost(
                new_state,
                state,
                candidate.cptp_map,
                candidate.depth,
                candidate.memory
            )
            all_costs.append((candidate.name, cost))
            
            if cost < min_cost:
                min_cost = cost
                best_action = candidate
        
        return {
            'infimum': min_cost,
            'best_action': best_action.name if best_action else None,
            'all_costs': all_costs,
            'n_candidates': len(candidates)
        }
    
    def verify_feasibility_constraint(self,
                                     state: WorldState,
                                     params: ResonanceParameters,
                                     candidates: List[ActionCandidate]) -> Dict[str, Any]:
        """
        可行性制約の検証
        
        論文 §9.4 より:
        F(ρ; θ) ≤ Θ ⟺ 状態は安定
        """
        F = self.compute_tier1_feasibility(state, params, candidates)
        Theta = self.threshold_config.compute_delta(params) + self.threshold_config.Theta
        
        return {
            'F': F,
            'Theta': Theta,
            'is_stable': F <= Theta,
            'margin': Theta - F
        }


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReIG2/twinRIG Feasibility Module Test")
    print("=" * 60)
    
    dim = 4
    params = ResonanceParameters(tau=0.5, epsilon=0.1, PFH=1.0)
    
    # [1] 射影演算子テスト
    print("\n[1] Projection Operators Test:")
    proj_ops = ProjectionOperators(dim=dim, stable_indices=[0, 1])
    H_inst = proj_ops.build_instability_hamiltonian()
    Pi_harm = proj_ops.build_harm_projection()
    
    print(f"  H_inst diagonal: {np.diag(H_inst).real}")
    print(f"  Pi_harm diagonal: {np.diag(Pi_harm).real}")
    
    # [2] 制約関数テスト
    print("\n[2] Constraint Function Test:")
    constraint = ConstraintFunction(H_inst, Pi_harm, params=params)
    
    # テスト状態
    state_stable = WorldState.pure_state([1, 0, 0, 0])
    state_unstable = WorldState.pure_state([0, 0, 0, 1])
    
    print(f"  Stable state cost: {constraint.total_cost(state_stable):.4f}")
    print(f"  Unstable state cost: {constraint.total_cost(state_unstable):.4f}")
    
    # [3] 可行行為集合テスト
    print("\n[3] Feasible Action Set Test:")
    action_set = FeasibleActionSet(
        current_state=state_stable,
        constraint=constraint,
        params=params
    )
    
    # ユニタリ行為を追加
    H_random = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H_random = (H_random + H_random.conj().T) / 2
    hamiltonian = Hamiltonian(H_random, "H_random")
    
    for t in [0.1, 0.5, 1.0]:
        action_set.add_unitary_action(hamiltonian, t, f"U(t={t})")
    
    action_set.evaluate_feasibility()
    print(f"  {action_set}")
    print(f"  Agency free: {action_set.is_agency_free()}")
    
    best_action, min_cost = action_set.get_minimum_cost_action()
    print(f"  Best action: {best_action.name}, cost: {min_cost:.4f}")
    
    # [4] 可行性指数テスト
    print("\n[4] Feasibility Index Test:")
    feasibility = FeasibilityIndex(constraint, action_set)
    F = feasibility.compute(state_stable, params, action_set._candidates)
    print(f"  F(ρ_stable; θ) = {F:.4f}")
    
    F_unstable = feasibility.compute(state_unstable, params, action_set._candidates)
    print(f"  F(ρ_unstable; θ) = {F_unstable:.4f}")
    
    # [5] 縮退例テスト
    print("\n[5] Degeneration Examples Test:")
    
    # Example 1: 完全拘束
    state1, constraint1 = DegenerationExamples.example1_fully_constrained()
    action_set1 = FeasibleActionSet(state1, constraint1, params=params)
    action_set1.add_unitary_action(hamiltonian, 0.1, "test_action")
    action_set1.evaluate_feasibility()
    print(f"  Ex1 (fully constrained): {action_set1}, agency_free={action_set1.is_agency_free()}")
    
    # Example 3: 倫理的禁止
    state3, constraint3, params3 = DegenerationExamples.example3_ethical_prohibition(PFH=100)
    action_set3 = FeasibleActionSet(state3, constraint3, params=params3)
    action_set3.add_unitary_action(hamiltonian, 0.1, "test_action")
    action_set3.evaluate_feasibility()
    print(f"  Ex3 (ethical prohibition): {action_set3}, agency_free={action_set3.is_agency_free()}")
    
    # [6] 回復ダイナミクステスト
    print("\n[6] Recovery Dynamics Test:")
    recovery = RecoveryDynamics(constraint, ThresholdConfig())
    
    # 不安定状態から開始
    initial = WorldState.pure_state([0, 0, 1, 0])
    
    candidates = [ActionCandidate(IdentityMap(dim), "identity", 0, 0)]
    for t in [0.1, 0.2]:
        candidates.append(ActionCandidate(
            UnitaryMap(hamiltonian, t), f"U({t})", 1, 1
        ))
    
    history = recovery.recovery_loop(
        initial_state=initial,
        params=params,
        candidates=candidates,
        max_iterations=10
    )
    
    print(f"  Initial feasibility: {history['feasibility'][0]:.4f}")
    print(f"  Final feasibility: {history['feasibility'][-1]:.4f}")
    print(f"  Iterations: {len(history['feasibility'])}")
    
    print("\n" + "=" * 60)
    print("All feasibility tests completed!")
    print("=" * 60)
