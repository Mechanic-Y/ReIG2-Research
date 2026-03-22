"""
ReIG Dynamic Coupling Simulation
==================================
Exploring the ReIG3 → RIF transition:
  When the collective field reaches criticality,
  coupling structure itself changes (audience → performer opens).

Four scenarios:
  1. No feedback (K_aud→perf = 0)
  2. Weak feedback (gradual opening)
  3. Strong feedback (sharp opening at threshold)
  4. Full bidirectional (always open, dialogue-like)

Key new element: state-dependent coupling
  K_back(t) = K_max * sigmoid((E_mean_aud - E*) / w)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── Parameters ──
dt = 0.005
T = 50.0
steps = int(T / dt)
t_arr = np.linspace(0, T, steps)

kappa = 0.5
delta_cubic = 0.15
A0_rest = 1.0
sigma_amp = 0.6
mu_E = 0.15
eta_E = 0.25
lam_E = 0.5
A_sat = 4.0
E_sat = 8.0
E_c = 3.5  # individual criticality

N = 8  # 1 performer + 7 audience

np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_dynamic(K_back_max, E_star, w_transition, label=""):
    """
    Simulate with state-dependent feedback coupling.
    K_aud→perf = K_back_max * sigmoid((mean_E_aud - E_star) / w)
    """
    omega = np.concatenate([[1.0], 0.85 + 0.3 * np.random.RandomState(42).rand(7)])

    theta = np.zeros((steps, N))
    A = np.zeros((steps, N))
    E = np.zeros((steps, N))
    K_feedback = np.zeros(steps)  # track feedback coupling over time

    theta[0] = np.concatenate([[0.0], np.random.RandomState(42).uniform(0, 2*np.pi, 7)])
    A[0] = np.concatenate([[2.5], np.ones(7)])
    E[0] = np.concatenate([[2.0], 0.2 + 0.2*np.random.RandomState(42).rand(7)])

    for s in range(steps - 1):
        th = theta[s]
        am = A[s]
        en = E[s]

        # Dynamic feedback: audience mean energy → opens coupling
        mean_E_aud = np.mean(en[1:])
        k_back = K_back_max * sigmoid((mean_E_aud - E_star) / max(w_transition, 0.01))
        K_feedback[s] = k_back

        # Build coupling matrix at this timestep
        K = np.zeros((N, N))
        for j in range(1, N):
            K[j, 0] = 3.5       # performer → audience (always on)
        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    K[i, j] = 1.5   # audience ↔ audience
        # Dynamic: audience → performer
        for j in range(1, N):
            K[0, j] = k_back

        dth = th[None, :] - th[:, None]
        sin_dth = np.sin(dth)
        cos_dth = np.cos(dth)

        d_theta = omega + np.sum(K * sin_dth, axis=1) / N
        amp_coupling = np.sum(K * am[None, :] * cos_dth, axis=1) / N
        d_A = -kappa * (am - A0_rest) - delta_cubic * am**3 + sigma_amp * amp_coupling
        self_energy = eta_E * am**2 / (1.0 + (am / A_sat)**2)
        mutual = np.sum(K * am[:, None] * am[None, :] * cos_dth, axis=1) / N
        mutual_sat = lam_E * mutual / (1.0 + en / E_sat)
        d_E = -mu_E * en + self_energy + mutual_sat

        noise_A = 0.02 * np.random.randn(N)
        noise_E = 0.01 * np.random.randn(N)

        theta[s+1] = th + d_theta * dt
        A[s+1] = np.maximum(am + (d_A + noise_A) * dt, 0.05)
        E[s+1] = np.maximum(en + (d_E + noise_E) * dt, 0.0)

    K_feedback[-1] = K_feedback[-2]
    return theta, A, E, K_feedback


# ── Run four scenarios ──
# 1. No feedback
th1, am1, en1, kf1 = simulate_dynamic(K_back_max=0.0, E_star=2.0, w_transition=0.3,
                                        label="No feedback")
# 2. Weak feedback (gradual, low max, wide transition)
th2, am2, en2, kf2 = simulate_dynamic(K_back_max=1.5, E_star=2.0, w_transition=0.8,
                                        label="Weak feedback")
# 3. Strong feedback (sharp threshold)
th3, am3, en3, kf3 = simulate_dynamic(K_back_max=3.0, E_star=2.5, w_transition=0.2,
                                        label="Strong feedback")
# 4. Full bidirectional (always open)
th4, am4, en4, kf4 = simulate_dynamic(K_back_max=3.5, E_star=-10.0, w_transition=0.1,
                                        label="Full bidirectional")


# =====================================================================
# Plotting
# =====================================================================
fig = plt.figure(figsize=(20, 26))
gs = GridSpec(5, 4, figure=fig, hspace=0.40, wspace=0.28,
             top=0.94, bottom=0.03, left=0.07, right=0.97)

# Colors
c_perf = '#1a5276'
c_aud_mean = '#c0392b'
c_aud = ['#e74c3c', '#e67e22', '#27ae60', '#8e44ad',
         '#2980b9', '#d4ac0c', '#16a085']
c_kfeed = '#8e44ad'
c_crit = '#e74c3c'

scenario_titles = [
    '① フィードバックなし\n（従来のライブ）',
    '② 弱いフィードバック\n（じわじわ伝わる）',
    '③ 強いフィードバック\n（場が臨界で一気に）',
    '④ 完全双方向\n（対面対話相当）'
]

all_data = [
    (th1, am1, en1, kf1),
    (th2, am2, en2, kf2),
    (th3, am3, en3, kf3),
    (th4, am4, en4, kf4),
]

fig.suptitle('ReIG3 → RIF：場のフィードバックによる結合構造の相転移',
             fontsize=20, fontweight='bold', color='#1a1a3a', y=0.98)

def add_ec(ax, ymax=None):
    ax.axhline(E_c, color=c_crit, linestyle='--', alpha=0.4, linewidth=1)


# ─── Row 0: Feedback coupling K_back(t) ───
for col, (th, am, en, kf) in enumerate(all_data):
    ax = fig.add_subplot(gs[0, col])
    ax.fill_between(t_arr, 0, kf, color=c_kfeed, alpha=0.3)
    ax.plot(t_arr, kf, color=c_kfeed, linewidth=1.5)
    ax.set_title(scenario_titles[col], fontsize=10, fontweight='bold',
                 color='#1a5276', pad=8)
    ax.set_ylim(-0.1, 4.0)
    ax.set_xlabel('時間', fontsize=9)
    if col == 0:
        ax.set_ylabel('$K_{\\mathrm{back}}(t)$', fontsize=11)
    # Mark when feedback activates
    if kf.max() > 0.1:
        idx_half = np.where(kf > kf.max() * 0.5)[0]
        if len(idx_half) > 0:
            t_act = t_arr[idx_half[0]]
            ax.axvline(t_act, color=c_kfeed, linestyle=':', alpha=0.5)
            ax.text(t_act + 1, kf.max() * 0.85, f'開通\n$t$≈{t_act:.0f}',
                    fontsize=8, color=c_kfeed)

fig.text(0.02, 0.86, 'フ\nィ\n|\nド\nバ\nッ\nク\n結\n合', fontsize=10,
         fontweight='bold', color=c_kfeed, va='center', ha='center')


# ─── Row 1: Performer amplitude ───
for col, (th, am, en, kf) in enumerate(all_data):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(t_arr, am[:, 0], color=c_perf, linewidth=2, label='演者')
    ax.set_ylim(0, 3.5)
    ax.set_xlabel('時間', fontsize=9)
    if col == 0:
        ax.set_ylabel('演者の振幅 $A_0$', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

fig.text(0.02, 0.70, '演\n者\n振\n幅', fontsize=10,
         fontweight='bold', color=c_perf, va='center', ha='center')


# ─── Row 2: Performer internal energy ───
for col, (th, am, en, kf) in enumerate(all_data):
    ax = fig.add_subplot(gs[2, col])
    ax.plot(t_arr, en[:, 0], color=c_perf, linewidth=2.5, label='演者', zorder=5)
    add_ec(ax)
    ax.set_ylim(0, max(en[:, 0].max() * 1.15, E_c * 1.5))
    ax.set_xlabel('時間', fontsize=9)
    if col == 0:
        ax.set_ylabel('演者の内部エネルギー $E_0$', fontsize=11)
    # Check criticality
    perf_crit = np.any(en[:, 0] >= E_c)
    if perf_crit:
        tc = t_arr[np.where(en[:, 0] >= E_c)[0][0]]
        ax.plot(tc, E_c, 'o', color=c_perf, markersize=8, zorder=6)
        ax.annotate(f'演者臨界！ $t$={tc:.1f}', xy=(tc, E_c),
                   xytext=(tc + 3, E_c + 1.0),
                   fontsize=9, color=c_perf, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=c_perf, lw=1.5))
        # Shade above Ec
        ax.fill_between(t_arr, E_c, en[:, 0],
                        where=en[:, 0] >= E_c,
                        color=c_perf, alpha=0.15)
    else:
        ax.text(T * 0.5, E_c * 0.7, '臨界に達しない',
                fontsize=10, color='#999', ha='center', style='italic')
    ax.text(T * 0.98, E_c + 0.1, '$E_c$', ha='right', fontsize=9,
            color=c_crit, alpha=0.6)

fig.text(0.02, 0.53, '演\n者\nエ\nネ\nル\nギ\nー', fontsize=10,
         fontweight='bold', color=c_perf, va='center', ha='center')


# ─── Row 3: Audience mean energy + individual ───
for col, (th, am, en, kf) in enumerate(all_data):
    ax = fig.add_subplot(gs[3, col])
    mean_E_aud = np.mean(en[:, 1:], axis=1)
    ax.plot(t_arr, mean_E_aud, color=c_aud_mean, linewidth=2.5,
            label='聴衆平均', zorder=5)
    for j in range(1, N):
        ax.plot(t_arr, en[:, j], color=c_aud[j-1], linewidth=0.6,
                alpha=0.4)
    add_ec(ax)
    ax.set_ylim(0, max(en[:, 1:].max() * 1.15, E_c * 1.5))
    ax.set_xlabel('時間', fontsize=9)
    if col == 0:
        ax.set_ylabel('聴衆のエネルギー', fontsize=11)
    n_crit = sum(1 for j in range(1, N) if np.any(en[:, j] >= E_c))
    ax.text(T * 0.5, ax.get_ylim()[1] * 0.9,
            f'臨界達成: {n_crit}/{N-1}名',
            fontsize=10, color=c_aud_mean, ha='center', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.text(T * 0.98, E_c + 0.1, '$E_c$', ha='right', fontsize=9,
            color=c_crit, alpha=0.6)

fig.text(0.02, 0.35, '聴\n衆\nエ\nネ\nル\nギ\nー', fontsize=10,
         fontweight='bold', color=c_aud_mean, va='center', ha='center')


# ─── Row 4: Summary diagram ───
# Show the phase transition of coupling structure

ax = fig.add_subplot(gs[4, :])
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Four stages as boxes with arrows
stage_x = [0.5, 3.0, 5.5, 8.5]
stage_labels = [
    'フィードバック\nなし',
    '弱い\nフィードバック',
    '強い\nフィードバック\n（場の臨界）',
    '完全双方向\n（RIF状態）'
]
stage_sub = [
    '演者→聴衆のみ\n演者は臨界に達しない',
    '場の熱が少し戻る\n演者はまだ臨界以下',
    '場が臨界→結合が開く\n演者も臨界に達する',
    '全結合が常時開通\n場全体が共鳴状態'
]
stage_colors = ['#ebf5fb', '#d4efdf', '#fdebd0', '#fadbd8']
stage_edge = ['#85c1e9', '#82e0aa', '#f0b27a', '#f1948a']

# Check which scenarios have performer criticality
perf_crits = [np.any(en[:, 0] >= E_c) for (th, am, en, kf) in all_data]

for i, (x, lab, sub, fc, ec) in enumerate(
        zip(stage_x, stage_labels, stage_sub, stage_colors, stage_edge)):
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((x - 0.9, -0.9), 1.8, 3.0,
                           boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor=ec, linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, 1.3, lab, ha='center', va='center', fontsize=9,
            fontweight='bold', color='#2c3e50')
    ax.text(x, -0.15, sub, ha='center', va='center', fontsize=7.5,
            color='#555', linespacing=1.4)

    # Performer criticality indicator
    if perf_crits[i]:
        ax.text(x, 1.85, '★ 演者臨界', ha='center', fontsize=8,
                color='#c0392b', fontweight='bold')
    else:
        ax.text(x, 1.85, '× 演者は臨界以下', ha='center', fontsize=7.5,
                color='#999')

# Arrows between stages
for i in range(3):
    ax.annotate('', xy=(stage_x[i+1] - 1.0, 0.5),
               xytext=(stage_x[i] + 1.0, 0.5),
               arrowprops=dict(arrowstyle='->', color='#2c3e50',
                              lw=2, connectionstyle='arc3,rad=0'))

# Overall label
ax.text(4.5, 2.4, 'ReIG3 → RIF 相転移：結合構造が場の状態から創発する',
        ha='center', fontsize=13, fontweight='bold', color='#1a5276')


plt.savefig('/home/claude/reig_dynamic_coupling.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.savefig('/home/claude/reig_dynamic_coupling.pdf',
            bbox_inches='tight', facecolor='white')

# ── Print summary ──
print("=== Results ===")
for i, (label, (th, am, en, kf)) in enumerate(
        zip(['No feedback', 'Weak feedback', 'Strong feedback', 'Full bidirectional'],
            all_data)):
    perf_max_E = en[:, 0].max()
    perf_crit = np.any(en[:, 0] >= E_c)
    aud_crit = sum(1 for j in range(1, N) if np.any(en[:, j] >= E_c))
    kf_max = kf.max()
    print(f"\n{i+1}. {label}:")
    print(f"   Performer max E = {perf_max_E:.2f}, reaches Ec: {perf_crit}")
    print(f"   Audience members reaching Ec: {aud_crit}/{N-1}")
    print(f"   Max feedback coupling: {kf_max:.2f}")

print("\nDone!")
