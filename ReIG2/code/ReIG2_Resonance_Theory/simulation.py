"""
reig2.simulation — シミュレーション・可視化
=============================================
共鳴事象作用素半群の時間発展シミュレーションと可視化ツール。
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable

from .state import SubjectState, Ensemble
from .semigroup import ResonanceSemigroup, ResonanceEvent
from .info_geometry import InformationGeometry


def run_resonance_simulation(
    N: int = 20,
    T_steps: int = 200,
    dim_z: int = 4,
    dim_theta: int = 3,
    R_c: float = 0.7,
    coupling_strength: float = 0.15,
    epsilon: float = 0.08,
    external_drive: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    共鳴シミュレーション実行

    Parameters
    ----------
    N : int — 主体数
    T_steps : int — ステップ数
    dim_z : int — 表出状態の次元
    dim_theta : int — パラメータ次元
    R_c : float — 臨界閾値
    coupling_strength : float — 結合強度
    epsilon : float — 可逆制御閾値
    external_drive : callable(t) -> np.ndarray or None — 外部駆動
    seed : int or None — 乱数シード

    Returns
    -------
    results : dict
    """
    if seed is not None:
        np.random.seed(seed)

    # 初期集合系
    ensemble = Ensemble.random(N=N, dim_z=dim_z, dim_theta=dim_theta)

    # 半群構成
    semigroup = ResonanceSemigroup(
        R_c=R_c,
        coupling_strength=coupling_strength,
        epsilon=epsilon,
    )

    # 時間発展
    final, history = semigroup.run(
        ensemble, T_steps=T_steps, u_func=external_drive
    )

    # 結果収集
    R_trace = [e.R for e in history]
    tau_trace = [e.tau for e in history]
    times = [e.t for e in history]

    # 情報幾何量の計算 (最初と最後の主体0)
    state_initial = history[0].ensemble_before[0]
    state_final = final[0]
    geo_distance = InformationGeometry.geodesic_distance(
        state_initial.theta, state_final.theta, state_initial.g
    )

    summary = semigroup.summary(history)
    summary.update({
        "R_trace": R_trace,
        "tau_trace": tau_trace,
        "times": times,
        "geodesic_distance_subject0": geo_distance,
        "final_ensemble": final,
        "history": history,
    })

    return summary


def print_simulation_report(results: dict):
    """シミュレーション結果のテキストレポート"""
    print("=" * 60)
    print("  ReIG2 共鳴事象シミュレーション結果")
    print("=" * 60)
    print(f"  総ステップ数:       {results['total_steps']}")
    print(f"  共鳴事象回数:       {results['resonance_events']}")
    print(f"  共鳴発火率:         {results['resonance_rate']:.2%}")
    print(f"  秩序パラメータ R:")
    print(f"    平均:             {results['R_mean']:.4f}")
    print(f"    最大:             {results['R_max']:.4f}")
    print(f"    最終:             {results['R_final']:.4f}")
    print(f"  測地線距離 (主体0): {results['geodesic_distance_subject0']:.4f}")
    print(f"  星点集合 P = {{t_k}}:")
    star = results['star_points']
    if len(star) <= 10:
        print(f"    {star}")
    else:
        print(f"    [{star[0]}, {star[1]}, ..., {star[-2]}, {star[-1]}]")
        print(f"    (計 {len(star)} 点)")
    print("=" * 60)


def plot_resonance_results(results: dict, save_path: Optional[str] = None):
    """
    シミュレーション結果の可視化 (matplotlib)

    4パネル:
      (1) 秩序パラメータ R(t) と臨界閾値
      (2) 共鳴事象の発火タイミング
      (3) 位相分布の時間発展 (初期 vs 最終)
      (4) 星点集合の可視化
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib が利用できないため可視化をスキップします")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'ReIG2 Resonance Event Simulation\n'
        r'$\mathfrak{R} = (\hat{\rho}_\epsilon \circ \hat{U}) '
        r'\circ \hat{M} \circ \hat{\tau} \circ \hat{A} '
        r'\circ \hat{E} \circ \hat{L} \circ \hat{C}$',
        fontsize=13,
    )

    times = results['times']
    R_trace = results['R_trace']
    tau_trace = results['tau_trace']
    star_points = results['star_points']
    history = results['history']

    # --- (1) 秩序パラメータ R(t) ---
    ax1 = axes[0, 0]
    ax1.plot(times, R_trace, 'b-', alpha=0.7, linewidth=1.2, label='$R(t)$')
    if history:
        R_c = history[0].R_c
        ax1.axhline(y=R_c, color='r', linestyle='--', alpha=0.7, label=f'$R_c = {R_c}$')
    # 共鳴点をマーク
    for tp in star_points:
        idx = int(tp)
        if 0 <= idx < len(R_trace):
            ax1.plot(tp, R_trace[idx], 'r*', markersize=8)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Order Parameter $R$')
    ax1.set_title('Order Parameter $R(t)$ & Threshold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # --- (2) 共鳴事象の発火 ---
    ax2 = axes[0, 1]
    fired_times = [t for t, tau in zip(times, tau_trace) if tau == 1]
    ax2.eventplot([fired_times], lineoffsets=0.5, linelengths=0.8, colors='red')
    ax2.set_xlabel('Time $t$')
    ax2.set_title(f'Resonance Events ($\\hat{{\\tau}} = 1$):  {len(fired_times)} events')
    ax2.set_yticks([])
    ax2.set_xlim(times[0], times[-1])
    ax2.grid(True, alpha=0.3, axis='x')

    # --- (3) 位相分布 (初期 vs 最終) ---
    ax3 = axes[1, 0]
    if history:
        phases_init = history[0].ensemble_before.phases()
        phases_final = results['final_ensemble'].phases()

        theta_grid = np.linspace(-np.pi, np.pi, 200)
        # KDE 的な可視化 (von Mises カーネル)
        kappa = 3.0
        kde_init = np.zeros_like(theta_grid)
        kde_final = np.zeros_like(theta_grid)
        for phi in phases_init:
            kde_init += np.exp(kappa * np.cos(theta_grid - phi))
        for phi in phases_final:
            kde_final += np.exp(kappa * np.cos(theta_grid - phi))
        kde_init /= len(phases_init)
        kde_final /= len(phases_final)

        ax3.plot(theta_grid, kde_init, 'b-', alpha=0.7, label='Initial')
        ax3.plot(theta_grid, kde_final, 'r-', alpha=0.7, label='Final')
        ax3.set_xlabel('Phase $\\phi$')
        ax3.set_ylabel('Density')
        ax3.set_title('Phase Distribution (Initial vs Final)')
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- (4) 星点集合の可視化 ---
    ax4 = axes[1, 1]
    if star_points:
        # 星点を極座標で可視化
        n_stars = len(star_points)
        angles = np.array(star_points) / max(times[-1], 1) * 2 * np.pi
        radii = np.linspace(0.3, 1.0, n_stars)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        ax4.scatter(x, y, c='gold', s=100, marker='*', edgecolors='orange',
                    zorder=5, label=f'$\\mathcal{{P}} = \\{{t_k\\}}$  ({n_stars} points)')
        # 軌道 (接続線)
        ax4.plot(x, y, 'gray', alpha=0.3, linewidth=0.8)
        ax4.set_xlim(-1.3, 1.3)
        ax4.set_ylim(-1.3, 1.3)
    else:
        ax4.text(0.5, 0.5, 'No resonance events', transform=ax4.transAxes,
                 ha='center', va='center', fontsize=14, color='gray')
    ax4.set_title('Star Points $\\mathcal{P} = \\{t_k\\}$\n'
                  '($\\gamma$ is shaped by $\\mathcal{P}$)')
    ax4.set_aspect('equal')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    else:
        plt.savefig('/tmp/resonance_simulation.png', dpi=150, bbox_inches='tight')

    plt.close()
    return fig
