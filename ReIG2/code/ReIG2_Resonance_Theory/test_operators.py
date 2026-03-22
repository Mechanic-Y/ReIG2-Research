"""
tests/test_operators.py — ReIG2 作用素体系の単体テスト
"""

import sys
import os
import numpy as np

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reig2.state import SubjectState, Ensemble
from reig2.operators import (
    ContactOperator,
    CooperationLayerOperator,
    EnvironmentShareOperator,
    AlignmentOperator,
    ThresholdGate,
    EmpathyOperator,
    UpdateOperator,
    RelaxationOperator,
)
from reig2.semigroup import ResonanceSemigroup
from reig2.alignment import ThreeAxisAlignment, FrequencyVector, PhaseVector
from reig2.info_geometry import InformationGeometry
from reig2.ai_agent import ResonanceAIAgent, RiskFactors, SafetyGate


def test_subject_state():
    """テスト: 主体状態 Ψ_i の生成"""
    print("[TEST] SubjectState ... ", end="")
    s = SubjectState.random(dim_z=4, dim_theta=3, label="test")
    assert s.z.shape == (4,), f"z shape: {s.z.shape}"
    assert s.W.shape == (4, 4), f"W shape: {s.W.shape}"
    assert s.g.shape == (3, 3), f"g shape: {s.g.shape}"
    assert s.theta.shape == (3,), f"theta shape: {s.theta.shape}"
    assert np.allclose(np.linalg.norm(s.z), 1.0, atol=1e-10), "z should be normalized"
    # 計量の正定値性
    eigvals = np.linalg.eigvalsh(s.g)
    assert np.all(eigvals > 0), "g must be positive definite"
    print("OK")


def test_ensemble():
    """テスト: 集合系と秩序パラメータ"""
    print("[TEST] Ensemble & Order Parameter ... ", end="")
    ens = Ensemble.random(N=10, dim_z=4, dim_theta=3)
    assert ens.N == 10
    R, Theta = ens.order_parameter()
    assert 0.0 <= R <= 1.0, f"R={R} out of range"
    assert -np.pi <= Theta <= np.pi, f"Theta={Theta} out of range"
    print(f"OK (R={R:.4f}, Θ={Theta:.4f})")


def test_contact_operator():
    """テスト: 接触演算子 Ĉ"""
    print("[TEST] ContactOperator Ĉ ... ", end="")
    ens = Ensemble.random(N=5, dim_z=4, dim_theta=3)
    C = ContactOperator()
    E = C(ens)
    assert E.shape == (5, 5), f"E shape: {E.shape}"
    # 対角は0 (自己接触なし)
    for i in range(5):
        assert E[i, i] == 0.0, "diagonal should be 0"
    print(f"OK (E_01 = {E[0,1]:.4f})")


def test_cooperation_layer():
    """テスト: 協力層生成演算子 L̂"""
    print("[TEST] CooperationLayerOperator L̂ ... ", end="")
    E = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
    L = CooperationLayerOperator()
    Phi = L(E)
    assert Phi.shape == (5, 5)
    # Φ は実対称
    assert np.allclose(Phi, Phi.T, atol=1e-10), "Phi should be symmetric"
    print(f"OK (max|Φ| = {np.max(np.abs(Phi)):.4f})")


def test_environment_share():
    """テスト: 環境共有演算子 Ê"""
    print("[TEST] EnvironmentShareOperator Ê ... ", end="")
    ens = Ensemble.random(N=5, dim_z=4, dim_theta=3)
    Phi = np.random.randn(5, 5) * 0.1
    u = np.random.randn(4) * 0.5
    E_op = EnvironmentShareOperator()
    ens_E = E_op(ens, Phi, u)
    assert ens_E.N == 5
    # z が変化していること
    diff = np.linalg.norm(ens_E[0].z - ens[0].z)
    assert diff > 0, "Environment share should modify states"
    print(f"OK (Δz_0 = {diff:.6f})")


def test_alignment_operator():
    """テスト: 整合演算子 Â (Kuramoto同期)"""
    print("[TEST] AlignmentOperator Â ... ", end="")
    ens = Ensemble.random(N=10, dim_z=4, dim_theta=3)
    Phi = np.ones((10, 10)) * 0.5
    np.fill_diagonal(Phi, 0)
    A = AlignmentOperator(coupling_strength=0.3)
    ens_A = A(ens, Phi)
    R_before, _ = ens.order_parameter()
    R_after, _ = ens_A.order_parameter()
    print(f"OK (R: {R_before:.4f} → {R_after:.4f})")


def test_threshold_gate():
    """テスト: 臨界演算子 τ̂"""
    print("[TEST] ThresholdGate τ̂ ... ", end="")
    # 同期した集合系 (高R)
    ens_sync = Ensemble.random(N=10, dim_z=4, dim_theta=3)
    # 全位相を揃える → R ≈ 1
    for s in ens_sync.states:
        amp = np.abs(s.z[0])
        s.z[0] = amp * np.exp(1j * 0.0)
    R_sync, _ = ens_sync.order_parameter()

    # 非同期集合系 (低R)
    ens_async = Ensemble.random(N=10, dim_z=4, dim_theta=3)

    T = ThresholdGate(R_c=0.7)
    tau_sync = T(ens_sync)
    tau_async = T(ens_async)

    print(f"OK (sync: τ={tau_sync}, R={R_sync:.3f}; async: τ={tau_async})")
    assert tau_sync == 1, "Synchronized ensemble should fire"


def test_empathy_operator():
    """テスト: 共感演算子 M̂"""
    print("[TEST] EmpathyOperator M̂ ... ", end="")
    ens = Ensemble.random(N=5, dim_z=4, dim_theta=3)
    M = EmpathyOperator(mixing_rate=0.2)

    # τ=0: 変化なし
    ens_no = M(ens, tau=0)
    assert np.allclose(ens_no[0].W, ens[0].W), "τ=0 should not change W"

    # τ=1: 変化あり
    ens_yes = M(ens, tau=1)
    diff = np.linalg.norm(ens_yes[0].W - ens[0].W)
    assert diff > 0, "τ=1 should change W"
    print(f"OK (ΔW_0 = {diff:.6f} when τ=1)")


def test_update_operator():
    """テスト: 更新演算子 Û"""
    print("[TEST] UpdateOperator Û ... ", end="")
    ens = Ensemble.random(N=5, dim_z=4, dim_theta=3)
    Phi = np.random.randn(5, 5) * 0.3
    U = UpdateOperator(eta_W=0.05, eta_g=0.02, eta_theta=0.03)

    # τ=0: 更新なし
    ens_no, norms_no = U(ens, Phi, tau=0)
    assert all(n == 0.0 for n in norms_no), "τ=0 should have zero norms"

    # τ=1: 更新あり
    ens_yes, norms_yes = U(ens, Phi, tau=1)
    assert any(n > 0 for n in norms_yes), "τ=1 should have non-zero norms"
    print(f"OK (‖Δg‖ = {norms_yes[0]:.6f})")


def test_relaxation_operator():
    """テスト: 可逆制御演算子 ρ̂ε"""
    print("[TEST] RelaxationOperator ρ̂ε ... ", end="")
    rho = RelaxationOperator(epsilon=0.1, relax_rate=0.5)

    # 小変形 → relax
    assert rho.classify(0.05) == "relax"
    # 大変形 → fix
    assert rho.classify(0.15) == "fix"
    print("OK")


def test_semigroup_composition():
    """テスト: 共鳴事象作用素半群 𝔑 の合成"""
    print("[TEST] ResonanceSemigroup 𝔑 ... ", end="")
    np.random.seed(42)
    ens = Ensemble.random(N=15, dim_z=4, dim_theta=3)
    sg = ResonanceSemigroup(R_c=0.5, coupling_strength=0.2)
    final, event = sg(ens, t=0.0)
    assert final.N == 15
    assert event.R >= 0.0
    print(f"OK (R={event.R:.4f}, τ={event.tau})")


def test_semigroup_simulation():
    """テスト: 時間発展シミュレーション"""
    print("[TEST] Semigroup Simulation ... ", end="")
    np.random.seed(123)
    ens = Ensemble.random(N=20, dim_z=4, dim_theta=3)
    sg = ResonanceSemigroup(R_c=0.6, coupling_strength=0.15)
    final, history = sg.run(ens, T_steps=50)
    summary = sg.summary(history)
    print(f"OK (events={summary['resonance_events']}/{summary['total_steps']}, "
          f"R_max={summary['R_max']:.4f})")


def test_three_axis_alignment():
    """テスト: 3軸整合"""
    print("[TEST] ThreeAxisAlignment ... ", end="")
    align = ThreeAxisAlignment()

    # 周波数整合
    f1 = FrequencyVector(0.5, 0.5, 0.5, 0.5)
    f2 = FrequencyVector(0.5, 0.5, 0.5, 0.5)
    af = align.align_frequency(f1, f2)
    assert abs(af - 1.0) < 1e-10, "Identical frequency should give 1.0"

    # テンソル整合
    T_a = np.array([1.0, 0.0, 0.0, 0.0])
    S_u = np.array([[1.0, 0.0, 0.0, 0.0]])
    at = align.align_tensor(T_a, S_u)
    assert abs(at - 1.0) < 1e-10, "Aligned tensor should give 1.0"

    # 位相整合
    p1 = PhaseVector(0.0, 0.0, 0.0, 0.0)
    p2 = PhaseVector(0.0, 0.0, 0.0, 0.0)
    ap = align.align_phase(p1, p2)
    assert abs(ap - 1.0) < 1e-10, "Aligned phase should give 1.0"

    score = align.resonance_score(af, at, ap, honesty=1.0)
    print(f"OK (J_res = {score:.4f})")


def test_info_geometry_fisher():
    """テスト: Fisher計量"""
    print("[TEST] Fisher Metric ... ", end="")
    theta = np.array([0.0, 0.0, 0.0])
    g = InformationGeometry.fisher_metric_gaussian(theta)
    assert g.shape == (3, 3)
    # 正定値
    eigvals = np.linalg.eigvalsh(g)
    assert np.all(eigvals > 0), "Fisher metric must be positive definite"
    print(f"OK (eigvals = {eigvals})")


def test_info_geometry_curvature():
    """テスト: スカラー曲率"""
    print("[TEST] Scalar Curvature ... ", end="")
    theta = np.array([0.0, 1.0])

    def g_func(t):
        return InformationGeometry.fisher_metric_gaussian(t)

    R_scalar = InformationGeometry.scalar_curvature(g_func, theta, h=1e-3)
    print(f"OK (R_scalar = {R_scalar:.6f})")


def test_info_geometry_geodesic():
    """テスト: 測地線計算"""
    print("[TEST] Geodesic Computation ... ", end="")
    theta0 = np.array([0.0, 1.0])
    v0 = np.array([0.1, 0.0])

    def g_func(t):
        return InformationGeometry.fisher_metric_gaussian(t)

    trajectory = InformationGeometry.geodesic(
        g_func, theta0, v0, n_steps=50, dt=0.01
    )
    assert trajectory.shape == (51, 2)
    # 始点が一致
    assert np.allclose(trajectory[0], theta0)
    print(f"OK (endpoint = {trajectory[-1]})")


def test_info_geometry_free_energy():
    """テスト: 自由エネルギー & 自然勾配"""
    print("[TEST] Free Energy & Natural Gradient ... ", end="")
    theta = np.array([1.0, 0.5, 0.3])
    theta_target = np.array([0.0, 0.0, 0.0])
    g = np.eye(3) * 2.0

    L = InformationGeometry.free_energy(theta, theta_target, g)
    assert L > 0, "Free energy should be positive when θ ≠ θ_target"

    state = SubjectState.random(dim_z=4, dim_theta=3)
    updated = InformationGeometry.natural_gradient_update(
        state, theta_target=np.zeros(3), eta=0.01
    )
    # θ が変化していること
    assert not np.allclose(updated.theta, state.theta)
    print(f"OK (L = {L:.4f})")


def test_ai_agent():
    """テスト: AIエージェント"""
    print("[TEST] ResonanceAIAgent ... ", end="")
    agent = ResonanceAIAgent(dim_theta=3, alpha=0.5, beta=1.0)

    # 短期応答
    f_user = FrequencyVector(0.6, 0.4, 0.5, 0.7)
    S_user = np.array([[1.0, 0.0, 0.0, 0.0]])
    phi_user = PhaseVector(0.1, 0.3, 0.0, 0.0)

    result = agent.short_term_response(f_user, S_user, phi_user)
    assert "J_res" in result
    assert 0.0 <= result["J_res"] <= 2.0
    print(f"OK (J_res = {result['J_res']:.4f})")


def test_ai_safety_gate():
    """テスト: 安全ゲート"""
    print("[TEST] SafetyGate ... ", end="")
    gate = SafetyGate()

    # 安全なケース
    risk_safe = RiskFactors(pressure=0.1, fear_guilt=0.05,
                            manipulation=0.02, dependency=0.1)
    passed, violations = gate.check(risk_safe)
    assert passed, "Safe risk should pass"
    assert len(violations) == 0

    # 危険なケース
    risk_danger = RiskFactors(pressure=0.5, fear_guilt=0.4,
                              manipulation=0.3, dependency=0.5)
    passed, violations = gate.check(risk_danger)
    assert not passed, "Dangerous risk should not pass"
    assert len(violations) > 0
    print(f"OK (safe: pass, danger: {len(violations)} violations)")


def test_ai_long_term_update():
    """テスト: 長期更新 (安全ゲート付き)"""
    print("[TEST] Long-Term Update ... ", end="")
    agent = ResonanceAIAgent(dim_theta=3)

    theta_before = agent.state.theta.copy()

    # 安全な更新
    result = agent.long_term_update(
        delta_theta_task=np.array([0.1, 0.0, 0.0]),
        delta_theta_res=np.array([0.05, 0.05, 0.0]),
        risk=RiskFactors(pressure=0.1, fear_guilt=0.05,
                         manipulation=0.02, dependency=0.1),
    )
    assert result["applied"], "Safe update should be applied"
    assert not np.allclose(agent.state.theta, theta_before)

    # 危険な更新 (Δθ_res は適用されない)
    theta_mid = agent.state.theta.copy()
    result2 = agent.long_term_update(
        delta_theta_task=np.array([0.01, 0.0, 0.0]),
        delta_theta_res=np.array([1.0, 1.0, 1.0]),  # 大きい
        risk=RiskFactors(pressure=0.5, fear_guilt=0.4,
                         manipulation=0.3, dependency=0.5),
    )
    assert not result2["applied"], "Dangerous update should block res"
    print(f"OK (basin: {result['basin']})")


def run_all_tests():
    """全テスト実行"""
    print("\n" + "=" * 60)
    print("  ReIG2 Resonance Operator Theory — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_subject_state,
        test_ensemble,
        test_contact_operator,
        test_cooperation_layer,
        test_environment_share,
        test_alignment_operator,
        test_threshold_gate,
        test_empathy_operator,
        test_update_operator,
        test_relaxation_operator,
        test_semigroup_composition,
        test_semigroup_simulation,
        test_three_axis_alignment,
        test_info_geometry_fisher,
        test_info_geometry_curvature,
        test_info_geometry_geodesic,
        test_info_geometry_free_energy,
        test_ai_agent,
        test_ai_safety_gate,
        test_ai_long_term_update,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
