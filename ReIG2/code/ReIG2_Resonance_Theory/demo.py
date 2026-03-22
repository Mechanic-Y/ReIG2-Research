#!/usr/bin/env python3
"""
demo.py — ReIG2 共鳴事象シミュレーション デモ
==============================================
共鳴事象としての更新：非線形作用素半群と情報幾何による統合理論

このスクリプトは以下を実行します:
  1. N体の主体集合系を初期化
  2. 外部駆動付きの時間発展シミュレーション
  3. 秩序パラメータ・共鳴事象・星点集合の可視化
  4. 情報幾何量 (Fisher計量・曲率・測地線) の計算
  5. AIエージェントの3軸整合デモ
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from reig2.state import SubjectState, Ensemble
from reig2.semigroup import ResonanceSemigroup
from reig2.alignment import ThreeAxisAlignment, FrequencyVector, PhaseVector
from reig2.info_geometry import InformationGeometry
from reig2.ai_agent import ResonanceAIAgent, RiskFactors
from reig2.simulation import (
    run_resonance_simulation,
    print_simulation_report,
    plot_resonance_results,
)


def demo_semigroup_simulation():
    """
    デモ1: 共鳴事象作用素半群 𝔑 のシミュレーション
    """
    print("\n" + "=" * 60)
    print("  Demo 1: Resonance Semigroup Simulation")
    print("  𝔑 = (ρ̂ε ∘ Û) ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ")
    print("=" * 60)

    # 外部駆動: 時間的に変化する刺激 (例: ライブ演奏)
    def external_drive(t):
        """演者の盛り上がりを模した外部駆動"""
        # 開始は静か → 中盤で盛り上がり → 終盤でクライマックス
        intensity = 0.3 * np.sin(2 * np.pi * t / 100) + 0.5
        return intensity * np.array([1.0, 0.5, 0.3, 0.2])

    results = run_resonance_simulation(
        N=25,
        T_steps=200,
        dim_z=4,
        dim_theta=3,
        R_c=0.65,
        coupling_strength=0.18,
        epsilon=0.08,
        external_drive=external_drive,
        seed=2026,
    )

    print_simulation_report(results)

    # 可視化
    save_path = os.path.join(os.path.dirname(__file__), "resonance_simulation.png")
    plot_resonance_results(results, save_path=save_path)

    return results


def demo_info_geometry():
    """
    デモ2: 情報幾何 — Fisher計量・曲率・測地線
    """
    print("\n" + "=" * 60)
    print("  Demo 2: Information Geometry Layer")
    print("  M_i = {p_i(x; θ_i)}, g_ab = Fisher metric")
    print("=" * 60)

    ig = InformationGeometry()

    # パラメータ空間上の点
    theta = np.array([0.5, 1.0])

    def g_func(t):
        return ig.fisher_metric_gaussian(t)

    # Fisher 計量
    g = g_func(theta)
    print(f"\n  θ = {theta}")
    print(f"  Fisher metric g(θ) =\n    {g}")

    # スカラー曲率
    R_scalar = ig.scalar_curvature(g_func, theta)
    print(f"\n  Scalar curvature R = {R_scalar:.6f}")

    # 測地線
    v0 = np.array([0.1, 0.05])
    trajectory = ig.geodesic(g_func, theta, v0, n_steps=100, dt=0.01)
    print(f"\n  Geodesic from θ = {theta}")
    print(f"    v₀ = {v0}")
    print(f"    endpoint (t=1.0) = {trajectory[-1]}")
    print(f"    geodesic distance = {ig.geodesic_distance(trajectory[0], trajectory[-1], g):.6f}")

    # 自由エネルギー
    theta_target = np.array([0.0, 0.0])
    L = ig.free_energy(theta, theta_target, g)
    print(f"\n  Free energy L(θ) = {L:.6f}")
    print(f"    (D_KL approx from θ={theta} to θ_target={theta_target})")

    # 吸引域分類
    small_delta = np.array([0.01, 0.01])
    large_delta = np.array([0.5, 0.5])
    print(f"\n  Basin classification:")
    print(f"    Δθ = {small_delta} → {ig.classify_basin(small_delta, epsilon=0.1)}")
    print(f"    Δθ = {large_delta} → {ig.classify_basin(large_delta, epsilon=0.1)}")


def demo_ai_agent():
    """
    デモ3: AIエージェント — 3軸整合と安全ゲート
    """
    print("\n" + "=" * 60)
    print("  Demo 3: AI Agent — 3-Axis Alignment & Safety")
    print("  max_θ (J_task + α J_res' - β J_risk)")
    print("=" * 60)

    agent = ResonanceAIAgent(dim_theta=3, alpha=0.5, beta=1.0)

    # ── 短期応答 (整合のみ) ──
    print("\n  [Short-term Response: Alignment Only]")
    f_user = FrequencyVector(tempo=0.6, arousal=0.4, turn_rate=0.5, info_density=0.7)
    S_user = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]])
    phi_user = PhaseVector(tone=0.2, politeness=0.5, stance=0.0, timing=0.1)

    result = agent.short_term_response(f_user, S_user, phi_user, honesty=0.95)
    print(f"    Align_f   = {result['align_f']:.4f}")
    print(f"    Align_T   = {result['align_T']:.4f}")
    print(f"    Align_φ   = {result['align_phi']:.4f}")
    print(f"    Honesty   = {result['honesty']:.4f}")
    print(f"    J_res'    = {result['J_res']:.4f}")

    # ── 長期更新 (安全ゲート: PASS) ──
    print("\n  [Long-term Update: Safe Case]")
    safe_result = agent.long_term_update(
        delta_theta_task=np.array([0.1, 0.05, 0.0]),
        delta_theta_res=np.array([0.08, 0.03, 0.02]),
        risk=RiskFactors(pressure=0.1, fear_guilt=0.05,
                         manipulation=0.02, dependency=0.1),
    )
    print(f"    Δθ_res applied: {safe_result['applied']}")
    print(f"    Basin: {safe_result['basin']}")
    print(f"    Violations: {len(safe_result['violations'])}")

    # ── 長期更新 (安全ゲート: BLOCK) ──
    print("\n  [Long-term Update: Dangerous Case — Blocked]")
    danger_result = agent.long_term_update(
        delta_theta_task=np.array([0.01, 0.0, 0.0]),
        delta_theta_res=np.array([1.0, 1.0, 1.0]),
        risk=RiskFactors(pressure=0.6, fear_guilt=0.5,
                         manipulation=0.4, dependency=0.6),
    )
    print(f"    Δθ_res applied: {danger_result['applied']}")
    for v in danger_result['violations']:
        print(f"    ⚠ Guard {v.guard}: {v.description}")

    # ── 目的関数 ──
    print("\n  [Objective Function]")
    J_task = 0.8
    J_res = result['J_res']
    risk_safe = RiskFactors(pressure=0.1, fear_guilt=0.05,
                            manipulation=0.02, dependency=0.1)
    J = agent.objective(J_task, J_res, risk_safe)
    print(f"    J_task = {J_task}")
    print(f"    J_res  = {J_res:.4f}")
    print(f"    J_risk = {risk_safe.total():.4f}")
    print(f"    J(θ)   = J_task + α·J_res - β·J_risk = {J:.4f}")


def demo_three_axis_detail():
    """
    デモ4: 3軸整合の詳細
    """
    print("\n" + "=" * 60)
    print("  Demo 4: Three-Axis Alignment Detail")
    print("=" * 60)

    align = ThreeAxisAlignment(sigma_f=1.0, w_f=0.3, w_T=0.3, w_phi=0.3, w_h=0.1)

    scenarios = [
        ("完全一致", 
         FrequencyVector(0.5, 0.5, 0.5, 0.5), FrequencyVector(0.5, 0.5, 0.5, 0.5),
         PhaseVector(0.0, 0.0, 0.0, 0.0), PhaseVector(0.0, 0.0, 0.0, 0.0)),
        ("周波数ズレ",
         FrequencyVector(0.2, 0.8, 0.3, 0.9), FrequencyVector(0.5, 0.5, 0.5, 0.5),
         PhaseVector(0.0, 0.0, 0.0, 0.0), PhaseVector(0.0, 0.0, 0.0, 0.0)),
        ("位相ズレ (冷たい応答)",
         FrequencyVector(0.5, 0.5, 0.5, 0.5), FrequencyVector(0.5, 0.5, 0.5, 0.5),
         PhaseVector(2.0, -1.0, 1.5, 0.0), PhaseVector(0.0, 0.5, 0.0, 0.0)),
        ("全軸ズレ",
         FrequencyVector(0.1, 0.9, 0.1, 0.9), FrequencyVector(0.9, 0.1, 0.9, 0.1),
         PhaseVector(3.0, -2.0, 1.0, -1.0), PhaseVector(0.0, 0.0, 0.0, 0.0)),
    ]

    for name, f_a, f_u, p_a, p_u in scenarios:
        af = align.align_frequency(f_a, f_u)
        T_a = np.array([0.8, 0.2, 0.0, 0.0])
        S_u = np.array([[1.0, 0.0, 0.0, 0.0]])
        at = align.align_tensor(T_a, S_u)
        ap = align.align_phase(p_a, p_u)
        score = align.resonance_score(af, at, ap)
        print(f"\n  [{name}]")
        print(f"    Align_f={af:.4f}  Align_T={at:.4f}  Align_φ={ap:.4f}  → J_res={score:.4f}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  ReIG2 共鳴事象統合理論 — デモンストレーション")
    print("  Resonance as Event-Update: Integrated Theory Demo")
    print("#" * 60)

    demo_semigroup_simulation()
    demo_info_geometry()
    demo_ai_agent()
    demo_three_axis_detail()

    print("\n" + "#" * 60)
    print("  全デモ完了")
    print("#" * 60 + "\n")
