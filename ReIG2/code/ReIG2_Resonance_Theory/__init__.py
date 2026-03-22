"""
ReIG2 Resonance Operator Theory
================================
共鳴事象としての更新：非線形作用素半群と情報幾何による統合理論

Modules:
    state           -- 状態空間定義 (Ψ, 集合系)
    operators       -- 8基本作用素 (Ĉ, L̂, Ê, Â, τ̂, M̂, Û, ρ̂ε)
    semigroup       -- 共鳴事象作用素半群 𝔑 の合成
    alignment       -- 3軸整合 (周波数・テンソル・位相)
    info_geometry   -- 情報幾何層 (Fisher計量・曲率・測地線)
    ai_agent        -- AI実装層 (目的関数・安全制約・二段階更新)
    simulation      -- シミュレーション・可視化
"""

__version__ = "1.0.0"
__author__ = "Yasuyuki Wakita"

from .state import SubjectState, Ensemble
from .operators import (
    ContactOperator,
    CooperationLayerOperator,
    EnvironmentShareOperator,
    AlignmentOperator,
    ThresholdGate,
    EmpathyOperator,
    UpdateOperator,
    RelaxationOperator,
)
from .semigroup import ResonanceSemigroup
from .alignment import ThreeAxisAlignment
from .info_geometry import InformationGeometry
from .ai_agent import ResonanceAIAgent
