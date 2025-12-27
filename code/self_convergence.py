"""
ReIG2/twinRIG 第5章
自己参照的収束シミュレーション
Self-Referential Convergence Simulation

Mechanic-Y / Yasuyuki Wakita
2025年12月

定理5.1（自己参照不動点の存在と収束）の数値検証
Banachの不動点定理に基づく収束解析
"""

import numpy as np
from scipy.linalg import expm, norm
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# =============================================================================
# データクラス
# =============================================================================

@dataclass
class ConvergenceResult:
    """収束結果"""
    converged: bool
    fixed_point: np.ndarray
    iterations: int
    history: List[np.ndarray]
    errors: List[float]
    final_error: float
    contraction_constant: float


@dataclass
class SpectralAnalysis:
    """スペクトル解析結果"""
    eigenvalues: np.ndarray
    spectral_radius: float
    spectral_gap: float
    second_eigenvalue: complex


# =============================================================================
# 縮小写像の条件検証
# =============================================================================

def estimate_contraction_constant(
    operator: np.ndarray,
    n_samples: int = 100
) -> float:
    """
    縮小定数 κ を推定
    
    ||T(ψ) - T(φ)|| ≤ κ ||ψ - φ|| for all ψ, φ
    
    Args:
        operator: 演算子 T
        n_samples: サンプル数
    
    Returns:
        推定された縮小定数 κ
    """
    dim = operator.shape[0]
    max_ratio = 0.0
    
    for _ in range(n_samples):
        # ランダムな状態ペア
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)
        
        phi = np.random.randn(dim) + 1j * np.random.randn(dim)
        phi = phi / np.linalg.norm(phi)
        
        # 変換
        T_psi = operator @ psi
        T_psi = T_psi / np.linalg.norm(T_psi)
        
        T_phi = operator @ phi
        T_phi = T_phi / np.linalg.norm(T_phi)
        
        # 比率
        input_dist = np.linalg.norm(psi - phi)
        output_dist = np.linalg.norm(T_psi - T_phi)
        
        if input_dist > 1e-10:
            ratio = output_dist / input_dist
            max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def verify_contraction_condition(
    operator: np.ndarray,
    kappa_threshold: float = 1.0
) -> Tuple[bool, float]:
    """
    縮小条件 (C1') を検証
    
    κ < 1 であれば縮小写像
    """
    kappa = estimate_contraction_constant(operator)
    is_contraction = kappa < kappa_threshold
    
    return is_contraction, kappa


def spectral_analysis(operator: np.ndarray) -> SpectralAnalysis:
    """
    スペクトル解析
    
    固有値、スペクトル半径、スペクトルギャップを計算
    """
    eigenvalues = np.linalg.eigvals(operator)
    
    # 絶対値でソート
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    
    spectral_radius = np.abs(eigenvalues_sorted[0])
    
    if len(eigenvalues_sorted) > 1:
        second_eigenvalue = eigenvalues_sorted[1]
        spectral_gap = spectral_radius - np.abs(second_eigenvalue)
    else:
        second_eigenvalue = 0
        spectral_gap = spectral_radius
    
    return SpectralAnalysis(
        eigenvalues=eigenvalues,
        spectral_radius=spectral_radius,
        spectral_gap=spectral_gap,
        second_eigenvalue=second_eigenvalue
    )


# =============================================================================
# Picard反復による不動点探索
# =============================================================================

def picard_iteration(
    operator: np.ndarray,
    initial_state: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    normalize: bool = True
) -> ConvergenceResult:
    """
    Picard反復による不動点探索
    
    ψ_{n+1} = T(ψ_n)
    
    Args:
        operator: 演算子 T
        initial_state: 初期状態 ψ_0
        max_iterations: 最大反復回数
        tolerance: 収束閾値
        normalize: 正規化するか
    
    Returns:
        ConvergenceResult
    """
    history = [initial_state.copy()]
    errors = []
    
    state = initial_state.copy()
    converged = False
    
    for n in range(max_iterations):
        # 反復
        new_state = operator @ state
        
        if normalize:
            new_state = new_state / np.linalg.norm(new_state)
        
        # 誤差計算
        error = np.linalg.norm(new_state - state)
        errors.append(error)
        
        history.append(new_state.copy())
        
        # 収束判定
        if error < tolerance:
            converged = True
            break
        
        state = new_state
    
    # 縮小定数の推定
    kappa = estimate_contraction_constant(operator, n_samples=50)
    
    return ConvergenceResult(
        converged=converged,
        fixed_point=state,
        iterations=len(errors),
        history=history,
        errors=errors,
        final_error=errors[-1] if errors else float('inf'),
        contraction_constant=kappa
    )


def power_method(
    operator: np.ndarray,
    initial_state: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-10
) -> Tuple[float, np.ndarray, int]:
    """
    べき乗法による最大固有値・固有ベクトルの計算
    
    Returns:
        (最大固有値, 対応する固有ベクトル, 反復回数)
    """
    dim = operator.shape[0]
    
    if initial_state is None:
        v = np.random.randn(dim) + 1j * np.random.randn(dim)
    else:
        v = initial_state.copy()
    
    v = v / np.linalg.norm(v)
    
    for i in range(max_iterations):
        # 反復
        w = operator @ v
        
        # Rayleigh商
        eigenvalue = np.vdot(v, w)
        
        # 正規化
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-15:
            break
        
        w = w / w_norm
        
        # 収束判定
        if np.linalg.norm(w - v) < tolerance:
            return eigenvalue, w, i + 1
        
        v = w
    
    return eigenvalue, v, max_iterations


# =============================================================================
# 指数収束率の解析
# =============================================================================

def analyze_exponential_convergence(
    errors: List[float],
    skip_initial: int = 5
) -> Dict:
    """
    誤差履歴から指数収束率を解析
    
    ||ψ_n - ψ*|| ≤ C |μ_2|^n
    
    Args:
        errors: 誤差の履歴
        skip_initial: 初期の不安定な部分をスキップ
    
    Returns:
        解析結果
    """
    if len(errors) < skip_initial + 2:
        return {"exponential_fit": False}
    
    # 対数をとる
    log_errors = np.log(np.array(errors[skip_initial:]) + 1e-15)
    n_values = np.arange(skip_initial, len(errors))
    
    # 線形フィット
    if len(log_errors) > 1:
        coeffs = np.polyfit(n_values, log_errors, 1)
        rate = np.exp(coeffs[0])  # |μ_2|
        prefactor = np.exp(coeffs[1])  # C
        
        # フィットの良さ
        fitted = np.polyval(coeffs, n_values)
        residual = np.sum((log_errors - fitted)**2)
        
        return {
            "exponential_fit": True,
            "convergence_rate": rate,
            "prefactor": prefactor,
            "residual": residual,
            "mu_2_estimate": rate
        }
    
    return {"exponential_fit": False}


# =============================================================================
# 世界演算子の構築（簡略版）
# =============================================================================

def build_world_operator(
    dim: int,
    tau: float = 0.5,
    epsilon: float = 0.3,
    PFH: float = 0.2,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2
) -> np.ndarray:
    """
    世界構築演算子 T_World を構築
    
    T_World = T_I ∘ T_R ∘ T_C ∘ U_multi ∘ U_res
    """
    # 拡張ハミルトニアン
    H_base = np.diag(np.arange(dim, dtype=float))
    H_future = np.zeros((dim, dim), dtype=complex)
    H_entropy = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim - 1):
        H_future[i, i+1] = 0.5
        H_future[i+1, i] = 0.5
        H_entropy[i, i+1] = 0.3j
        H_entropy[i+1, i] = -0.3j
    
    H_extended = H_base + tau * H_future + epsilon * H_entropy + PFH * np.eye(dim)
    
    # 時間発展演算子
    U_res = expm(-1j * H_extended)
    U_multi = expm(-1j * (1 + tau) * H_extended * 0.5)
    
    # 変換演算子
    T_C = np.diag([np.exp(-alpha * i / dim) for i in range(dim)])
    T_C = T_C / np.linalg.norm(T_C, ord=2)
    
    T_R = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            T_R[i, j] = np.exp(-beta * abs(i - j))
    T_R = T_R / np.linalg.norm(T_R, ord=2)
    
    T_I = (1 - gamma) * np.eye(dim) + gamma * np.ones((dim, dim)) / dim
    
    # 合成
    T_World = T_I @ T_R @ T_C @ U_multi @ U_res
    
    return T_World


# =============================================================================
# 定理5.1の数値検証
# =============================================================================

def verify_theorem_5_1(
    dim: int = 8,
    tau: float = 0.5,
    epsilon: float = 0.3,
    PFH: float = 0.2,
    gamma_contraction: float = 0.5,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    verbose: bool = True
) -> Dict:
    """
    定理5.1の数値検証
    
    条件:
    (C1') 強縮小性: ||T(ψ) - T(φ)|| ≤ κ ||ψ - φ||, κ < 1
    (C2) 境界保存: T(∂B_r) ⊂ B_r
    (C3) 完備性: H_full は完備
    (C4) スペクトルギャップ: |λ_2| < 1
    
    結論:
    - 存在性: 一意な不動点 |I⟩ が存在
    - 収束性: lim_{n→∞} T^n |ψ_0⟩ = |I⟩
    - 収束率: ||T^n ψ_0 - I|| ≤ C |μ_2|^n
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("定理5.1の数値検証")
        print("=" * 60)
    
    # 世界演算子の構築
    T_World = build_world_operator(
        dim, tau, epsilon, PFH,
        alpha=0.5, beta=0.3, gamma=gamma_contraction
    )
    
    # 条件 (C1'): 縮小性
    is_contraction, kappa = verify_contraction_condition(T_World)
    results["C1_contraction"] = {
        "satisfied": is_contraction,
        "kappa": kappa
    }
    
    if verbose:
        print(f"\n[C1'] 縮小条件:")
        print(f"      κ = {kappa:.4f} {'< 1 ✓' if is_contraction else '>= 1 ✗'}")
    
    # 条件 (C4): スペクトル解析
    spectral = spectral_analysis(T_World)
    results["C4_spectral"] = {
        "spectral_radius": spectral.spectral_radius,
        "spectral_gap": spectral.spectral_gap,
        "second_eigenvalue": spectral.second_eigenvalue,
        "satisfied": np.abs(spectral.second_eigenvalue) < 1
    }
    
    if verbose:
        print(f"\n[C4] スペクトルギャップ:")
        print(f"      スペクトル半径: {spectral.spectral_radius:.4f}")
        print(f"      |λ_2| = {np.abs(spectral.second_eigenvalue):.4f}")
        print(f"      ギャップ: {spectral.spectral_gap:.4f}")
    
    # Picard反復
    initial_state = np.random.randn(dim) + 1j * np.random.randn(dim)
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    convergence = picard_iteration(
        T_World, initial_state,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    
    results["convergence"] = {
        "converged": convergence.converged,
        "iterations": convergence.iterations,
        "final_error": convergence.final_error
    }
    
    if verbose:
        print(f"\n[収束結果]")
        print(f"      収束: {'Yes ✓' if convergence.converged else 'No ✗'}")
        print(f"      反復回数: {convergence.iterations}")
        print(f"      最終誤差: {convergence.final_error:.2e}")
    
    # 指数収束率の解析
    if len(convergence.errors) > 10:
        exp_analysis = analyze_exponential_convergence(convergence.errors)
        results["exponential_convergence"] = exp_analysis
        
        if verbose and exp_analysis.get("exponential_fit"):
            print(f"\n[指数収束解析]")
            print(f"      収束率 |μ_2|: {exp_analysis['convergence_rate']:.4f}")
            print(f"      理論値との比較: {np.abs(spectral.second_eigenvalue):.4f}")
    
    # 不動点の検証
    if convergence.converged:
        fixed_point = convergence.fixed_point
        T_fixed = T_World @ fixed_point
        T_fixed = T_fixed / np.linalg.norm(T_fixed)
        
        fixed_point_error = np.linalg.norm(T_fixed - fixed_point)
        results["fixed_point"] = {
            "error": fixed_point_error,
            "is_valid": fixed_point_error < tolerance * 10
        }
        
        if verbose:
            print(f"\n[不動点検証]")
            print(f"      ||T(I) - I|| = {fixed_point_error:.2e}")
    
    # 総合判定
    all_satisfied = (
        results["C1_contraction"]["satisfied"] and
        results["C4_spectral"]["satisfied"] and
        results["convergence"]["converged"]
    )
    
    results["theorem_verified"] = all_satisfied
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"定理5.1: {'検証成功 ✓' if all_satisfied else '条件不成立 ✗'}")
        print(f"{'=' * 60}")
    
    return results


# =============================================================================
# 複数初期状態からの収束
# =============================================================================

def test_convergence_from_multiple_initial_states(
    T_World: np.ndarray,
    n_initial_states: int = 10,
    max_iterations: int = 500,
    tolerance: float = 1e-8
) -> Dict:
    """
    複数の初期状態から収束を検証
    
    不動点の一意性を確認
    """
    dim = T_World.shape[0]
    fixed_points = []
    all_converged = True
    
    for i in range(n_initial_states):
        # ランダムな初期状態
        initial = np.random.randn(dim) + 1j * np.random.randn(dim)
        initial = initial / np.linalg.norm(initial)
        
        result = picard_iteration(T_World, initial, max_iterations, tolerance)
        
        if result.converged:
            fixed_points.append(result.fixed_point)
        else:
            all_converged = False
    
    # 不動点の一意性確認
    if len(fixed_points) > 1:
        max_diff = 0
        for i in range(len(fixed_points)):
            for j in range(i + 1, len(fixed_points)):
                # 位相の不定性を考慮
                phase = np.vdot(fixed_points[i], fixed_points[j])
                phase = phase / np.abs(phase) if np.abs(phase) > 1e-10 else 1
                diff = np.linalg.norm(fixed_points[i] - phase * fixed_points[j])
                max_diff = max(max_diff, diff)
        
        unique = max_diff < tolerance * 100
    else:
        unique = True
        max_diff = 0
    
    return {
        "all_converged": all_converged,
        "n_converged": len(fixed_points),
        "unique_fixed_point": unique,
        "max_difference": max_diff
    }


# =============================================================================
# デモ
# =============================================================================

def demo():
    """自己参照的収束のデモ"""
    
    print("=" * 60)
    print("ReIG2/twinRIG 第5章")
    print("自己参照的収束シミュレーション")
    print("=" * 60)
    
    # 1. 基本的な収束テスト
    print("\n[1] 基本収束テスト")
    dim = 8
    T_World = build_world_operator(dim, tau=0.5, epsilon=0.3, PFH=0.2)
    
    initial = np.zeros(dim, dtype=complex)
    initial[0] = 1.0
    
    result = picard_iteration(T_World, initial, max_iterations=200)
    
    print(f"    収束: {result.converged}")
    print(f"    反復回数: {result.iterations}")
    print(f"    最終誤差: {result.final_error:.2e}")
    print(f"    縮小定数 κ: {result.contraction_constant:.4f}")
    
    # 2. スペクトル解析
    print("\n[2] スペクトル解析")
    spectral = spectral_analysis(T_World)
    
    print(f"    スペクトル半径: {spectral.spectral_radius:.4f}")
    print(f"    |λ_2|: {np.abs(spectral.second_eigenvalue):.4f}")
    print(f"    スペクトルギャップ: {spectral.spectral_gap:.4f}")
    
    # 3. 複数初期状態テスト
    print("\n[3] 複数初期状態からの収束（一意性検証）")
    multi_result = test_convergence_from_multiple_initial_states(T_World, n_initial_states=5)
    
    print(f"    全て収束: {multi_result['all_converged']}")
    print(f"    一意な不動点: {multi_result['unique_fixed_point']}")
    print(f"    不動点間の最大差: {multi_result['max_difference']:.2e}")
    
    # 4. パラメータ依存性
    print("\n[4] パラメータ依存性（γ_contraction）")
    
    for gamma in [0.1, 0.3, 0.5, 0.7]:
        T = build_world_operator(dim, gamma=gamma)
        is_contr, kappa = verify_contraction_condition(T)
        print(f"    γ={gamma:.1f}: κ={kappa:.4f} {'(縮小)' if is_contr else ''}")
    
    # 5. 定理5.1の完全検証
    print("\n[5] 定理5.1の完全検証")
    verify_theorem_5_1(dim=8, gamma_contraction=0.5, verbose=True)
    
    # 6. 指数収束の可視化
    print("\n[6] 収束履歴")
    if len(result.errors) > 0:
        print("    反復  |  誤差")
        print("    ------+----------")
        for i in [0, 5, 10, 20, 50, 100, min(150, len(result.errors)-1)]:
            if i < len(result.errors):
                print(f"    {i:5d} | {result.errors[i]:.2e}")


if __name__ == "__main__":
    demo()
