"""
ReIG Resonance Dynamics Simulation (v2)
========================================
Three scenarios with saturation (nonlinear damping).

Model:
  dθ/dt = ω_i + (1/N) Σ K_ij sin(θ_j - θ_i)
  dA/dt = -κ(A - A0) - δA³ + (σ/N) Σ K_ij A_j cos(θ_j - θ_i)
  dE/dt = -μE + η A² / (1 + A²/A_sat²) + (λ/N) Σ K_ij A_i A_j cos(...) / (1 + E/E_sat)

The cubic damping -δA³ prevents amplitude blowup.
The saturation terms cap energy accumulation.
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
T = 40.0
steps = int(T / dt)
t = np.linspace(0, T, steps)

kappa = 0.5      # amplitude linear damping
delta = 0.15     # amplitude cubic damping (saturation)
A0_rest = 1.0    # resting amplitude
sigma = 0.6      # amplitude coupling
mu = 0.15        # energy damping
eta = 0.25       # self-energy from amplitude
lam = 0.5        # mutual energy amplification
A_sat = 4.0      # amplitude saturation scale
E_sat = 8.0      # energy saturation scale
E_c = 3.5        # critical energy threshold

np.random.seed(42)


def simulate(N, omega, K_matrix, theta0, A0_arr, E0):
    theta = np.zeros((steps, N))
    A = np.zeros((steps, N))
    E = np.zeros((steps, N))

    theta[0] = theta0
    A[0] = A0_arr
    E[0] = E0

    for s in range(steps - 1):
        th = theta[s]
        am = A[s]
        en = E[s]

        dth = th[None, :] - th[:, None]
        sin_dth = np.sin(dth)
        cos_dth = np.cos(dth)

        # dθ/dt
        d_theta = omega + np.sum(K_matrix * sin_dth, axis=1) / N

        # dA/dt with cubic saturation
        amp_coupling = np.sum(K_matrix * am[None, :] * cos_dth, axis=1) / N
        d_A = -kappa * (am - A0_rest) - delta * am**3 + sigma * amp_coupling

        # dE/dt with saturation
        self_energy = eta * am**2 / (1.0 + (am / A_sat)**2)
        mutual = np.sum(K_matrix * am[:, None] * am[None, :] * cos_dth, axis=1) / N
        mutual_sat = lam * mutual / (1.0 + en / E_sat)
        d_E = -mu * en + self_energy + mutual_sat

        noise_A = 0.03 * np.random.randn(N)
        noise_E = 0.01 * np.random.randn(N)

        theta[s+1] = th + d_theta * dt
        A[s+1] = np.maximum(am + (d_A + noise_A) * dt, 0.05)
        E[s+1] = np.maximum(en + (d_E + noise_E) * dt, 0.0)

    return theta, A, E


# =====================================================================
# Scenario 1: Symmetric (twinRIG - dialogue)
# =====================================================================
N1 = 2
omega1 = np.array([1.0, 1.2])
K1 = np.array([[0, 3.0],
               [3.0, 0]])
theta0_1 = np.array([0.0, np.pi * 0.6])
A0_1 = np.array([1.0, 1.0])
E0_1 = np.array([0.5, 0.3])

th1, am1, en1 = simulate(N1, omega1, K1, theta0_1, A0_1, E0_1)


# =====================================================================
# Scenario 2: Asymmetric (ReIG2 - listening to music)
# =====================================================================
N2 = 2
omega2 = np.array([1.0, 1.1])
K2 = np.array([[0, 0.0],
               [3.5, 0]])
theta0_2 = np.array([0.0, np.pi * 0.4])
A0_2 = np.array([2.5, 1.0])   # performer is expressive
E0_2 = np.array([2.0, 0.3])   # performer already energized

th2, am2, en2 = simulate(N2, omega2, K2, theta0_2, A0_2, E0_2)


# =====================================================================
# Scenario 3: Mixed (ReIG3→RIF - live venue)
# =====================================================================
N3 = 8
omega3 = np.concatenate([[1.0], 0.85 + 0.3 * np.random.rand(7)])

K3 = np.zeros((N3, N3))
for j in range(1, N3):
    K3[j, 0] = 3.5   # performer → audience
for i in range(1, N3):
    for j in range(1, N3):
        if i != j:
            K3[i, j] = 1.5  # audience ↔ audience

theta0_3 = np.concatenate([[0.0], np.random.uniform(0, 2*np.pi, 7)])
A0_3 = np.concatenate([[2.5], np.ones(7)])
E0_3 = np.concatenate([[2.0], 0.2 + 0.2*np.random.rand(7)])

th3, am3, en3 = simulate(N3, omega3, K3, theta0_3, A0_3, E0_3)


# =====================================================================
# Plotting
# =====================================================================
fig = plt.figure(figsize=(18, 24))
gs = GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.28,
             top=0.93, bottom=0.03, left=0.09, right=0.96)

c_A = '#1a5276'
c_B = '#c0392b'
c_perf = '#1a5276'
c_aud = ['#c0392b', '#e67e22', '#27ae60', '#8e44ad',
         '#2980b9', '#d4ac0c', '#16a085']
c_crit = '#e74c3c'

fig.suptitle('ReIG 共鳴ダイナミクス ─ 三つの結合構造の比較',
             fontsize=20, fontweight='bold', color='#1a1a3a', y=0.97)

titles = [
    '① 対称結合（twinRIG：対面対話）',
    '② 非対称結合（ReIG2：音楽を聴く）',
    '③ 混合結合（ReIG3→RIF：ライブ会場）'
]

def add_ec_line(ax):
    ax.axhline(E_c, color=c_crit, linestyle='--', alpha=0.5, linewidth=1.2)
    ax.text(T * 0.99, E_c + 0.15, '$E_c$（臨界）', ha='right',
            fontsize=9, color=c_crit, alpha=0.7)


# ─── Row 0: Phase synchronization ───
ax = fig.add_subplot(gs[0, 0])
pd1 = np.mod(th1[:, 1] - th1[:, 0] + np.pi, 2*np.pi) - np.pi
ax.plot(t, pd1, color=c_A, linewidth=1.2)
ax.set_ylabel('位相差 $\\Delta\\theta$', fontsize=11)
ax.set_title(titles[0], fontsize=11, fontweight='bold', color='#1a5276', pad=8)
ax.set_ylim(-np.pi, np.pi)
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('時間', fontsize=10)
ax.text(T*0.5, -2.5, '→ 位相が揃う（同期）', fontsize=9, color='#555', ha='center')

ax = fig.add_subplot(gs[0, 1])
pd2 = np.mod(th2[:, 1] - th2[:, 0] + np.pi, 2*np.pi) - np.pi
ax.plot(t, pd2, color=c_A, linewidth=1.2)
ax.set_title(titles[1], fontsize=11, fontweight='bold', color='#1a5276', pad=8)
ax.set_ylim(-np.pi, np.pi)
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('時間', fontsize=10)
ax.text(T*0.5, -2.5, '→ 聴き手が歌い手に引き込まれる', fontsize=9, color='#555', ha='center')

ax = fig.add_subplot(gs[0, 2])
for j in range(1, N3):
    pd = np.mod(th3[:, j] - th3[:, 0] + np.pi, 2*np.pi) - np.pi
    ax.plot(t, pd, color=c_aud[j-1], linewidth=0.9, alpha=0.75)
ax.set_title(titles[2], fontsize=11, fontweight='bold', color='#1a5276', pad=8)
ax.set_ylim(-np.pi, np.pi)
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('時間', fontsize=10)
ax.text(T*0.5, -2.5, '→ 聴衆全体が同期', fontsize=9, color='#555', ha='center')

fig.text(0.035, 0.82, '位\n相\n同\n期', fontsize=13, fontweight='bold',
         color='#1a5276', va='center', ha='center')


# ─── Row 1: Amplitude ───
ax = fig.add_subplot(gs[1, 0])
ax.plot(t, am1[:, 0], color=c_A, linewidth=1.8, label='主体A')
ax.plot(t, am1[:, 1], color=c_B, linewidth=1.8, label='主体B')
ax.set_ylabel('振幅 $A_i$（心の揺らぎ）', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlabel('時間', fontsize=10)

ax = fig.add_subplot(gs[1, 1])
ax.plot(t, am2[:, 0], color=c_A, linewidth=1.8, label='歌い手')
ax.plot(t, am2[:, 1], color=c_B, linewidth=1.8, label='聴き手')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlabel('時間', fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.plot(t, am3[:, 0], color=c_perf, linewidth=2.2, label='演者', zorder=5)
for j in range(1, N3):
    lbl = f'聴衆{j}' if j <= 3 else None
    ax.plot(t, am3[:, j], color=c_aud[j-1], linewidth=0.9, alpha=0.75, label=lbl)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.set_xlabel('時間', fontsize=10)

fig.text(0.035, 0.60, '振\n幅\n結\n合', fontsize=13, fontweight='bold',
         color='#1a5276', va='center', ha='center')


# ─── Row 2: Internal energy ───
def mark_criticality(ax, t_arr, E_arr, label, color, offset_x=2, offset_y=0.5):
    crit = np.where(E_arr >= E_c)[0]
    if len(crit) > 0:
        tc = t_arr[crit[0]]
        ax.plot(tc, E_c, 'o', color=color, markersize=7, zorder=5)
        ax.annotate(f'{label}臨界 $t$={tc:.1f}', xy=(tc, E_c),
                   xytext=(tc + offset_x, E_c + offset_y),
                   fontsize=8, color=color, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1))
        return tc
    return None

ax = fig.add_subplot(gs[2, 0])
ax.plot(t, en1[:, 0], color=c_A, linewidth=1.8, label='主体A')
ax.plot(t, en1[:, 1], color=c_B, linewidth=1.8, label='主体B')
add_ec_line(ax)
ax.set_ylabel('内部エネルギー $E_i$', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlabel('時間', fontsize=10)
tc_a = mark_criticality(ax, t, en1[:, 0], 'A', c_A, 3, 0.5)
tc_b = mark_criticality(ax, t, en1[:, 1], 'B', c_B, 3, -0.7)
# Highlight mutual criticality
if tc_a and tc_b:
    ax.text(T*0.5, max(en1.max()*0.85, E_c+1.5),
            '← 連鎖的に臨界', fontsize=10, color='#c0392b',
            ha='center', fontweight='bold')

ax = fig.add_subplot(gs[2, 1])
ax.plot(t, en2[:, 0], color=c_A, linewidth=1.8, label='歌い手')
ax.plot(t, en2[:, 1], color=c_B, linewidth=1.8, label='聴き手')
add_ec_line(ax)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlabel('時間', fontsize=10)
mark_criticality(ax, t, en2[:, 0], '歌い手', c_A, -12, 0.5)
mark_criticality(ax, t, en2[:, 1], '聴き手', c_B, 3, 0.5)

ax = fig.add_subplot(gs[2, 2])
ax.plot(t, en3[:, 0], color=c_perf, linewidth=2.2, label='演者', zorder=5)
for j in range(1, N3):
    lbl = f'聴衆{j}' if j <= 3 else None
    ax.plot(t, en3[:, j], color=c_aud[j-1], linewidth=0.9, alpha=0.75, label=lbl)
add_ec_line(ax)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.set_xlabel('時間', fontsize=10)
# Count audience criticality
n_crit = sum(1 for j in range(1, N3) if np.any(en3[:, j] >= E_c))
ax.text(T*0.65, max(en3[:, 1:].max()*0.85, E_c+1),
        f'聴衆 {n_crit}/{N3-1}名 臨界達成', fontsize=10,
        color='#c0392b', fontweight='bold', ha='center')

fig.text(0.035, 0.37, '内\n部\nエ\nネ\nル\nギ\nー', fontsize=13, fontweight='bold',
         color='#1a5276', va='center', ha='center')


# ─── Row 3: Coupling diagrams ───

# Scenario 1
ax = fig.add_subplot(gs[3, 0])
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.2, 1.5)
ax.set_aspect('equal')
ax.axis('off')

cA = plt.Circle((-0.6, 0), 0.38, facecolor='#d4e6f1', edgecolor=c_A, linewidth=2.5)
cB = plt.Circle((0.6, 0), 0.38, facecolor='#fadbd8', edgecolor=c_B, linewidth=2.5)
ax.add_patch(cA); ax.add_patch(cB)
ax.text(-0.6, 0, 'A', ha='center', va='center', fontsize=16, fontweight='bold', color=c_A)
ax.text(0.6, 0, 'B', ha='center', va='center', fontsize=16, fontweight='bold', color=c_B)
ax.annotate('', xy=(0.18, 0.1), xytext=(-0.18, 0.1),
           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
ax.annotate('', xy=(-0.18, -0.1), xytext=(0.18, -0.1),
           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
ax.text(0, 1.1, '双方向共鳴', ha='center', fontsize=13, fontweight='bold', color='#1a5276')
ax.text(0, 0.75, '$K_{AB} = K_{BA}$', ha='center', fontsize=11, color='#555')
ax.text(0, -0.75, '両者が臨界に達する', ha='center', fontsize=11, color='#555')

# Scenario 2
ax = fig.add_subplot(gs[3, 1])
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.2, 1.5)
ax.set_aspect('equal')
ax.axis('off')

cP = plt.Circle((-0.6, 0), 0.38, facecolor='#d4e6f1', edgecolor=c_A, linewidth=2.5)
cL = plt.Circle((0.6, 0), 0.38, facecolor='#fadbd8', edgecolor=c_B, linewidth=2.5)
ax.add_patch(cP); ax.add_patch(cL)
ax.text(-0.6, 0, '歌', ha='center', va='center', fontsize=14, fontweight='bold', color=c_A)
ax.text(0.6, 0, '聴', ha='center', va='center', fontsize=14, fontweight='bold', color=c_B)
ax.annotate('', xy=(0.18, 0), xytext=(-0.18, 0),
           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
ax.text(0, 1.1, '単方向伝搬', ha='center', fontsize=13, fontweight='bold', color='#1a5276')
ax.text(0, 0.75, '歌→聴のみ（一方向）', ha='center', fontsize=11, color='#555')
ax.text(0, -0.75, '聴き手のみ臨界に達する', ha='center', fontsize=11, color='#555')

# Scenario 3
ax = fig.add_subplot(gs[3, 2])
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

perf = plt.Circle((0, 1.0), 0.32, facecolor='#d4e6f1', edgecolor=c_A, linewidth=2.5)
ax.add_patch(perf)
ax.text(0, 1.0, '演', ha='center', va='center', fontsize=14, fontweight='bold', color=c_A)

aud_pos = []
for k in range(7):
    angle = np.pi * 0.12 + np.pi * 0.76 * k / 6
    x = 1.35 * np.cos(angle)
    y = -0.3 - 0.7 * np.sin(angle)
    aud_pos.append((x, y))
    ac = plt.Circle((x, y), 0.2, facecolor='#fadbd8',
                     edgecolor=c_aud[k], linewidth=1.5, alpha=0.85)
    ax.add_patch(ac)

for (x, y) in aud_pos:
    dx, dy = x - 0, y - 1.0
    dist = np.sqrt(dx**2 + dy**2)
    ax.annotate('', xy=(x - 0.22*dx/dist, y - 0.22*dy/dist),
               xytext=(0 + 0.35*dx/dist, 1.0 + 0.35*dy/dist),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2, alpha=0.6))

for i in range(len(aud_pos)):
    for j in range(i+1, min(i+3, len(aud_pos))):
        x1, y1 = aud_pos[i]
        x2, y2 = aud_pos[j]
        ax.plot([x1, x2], [y1, y2], '-', color='#e74c3c', alpha=0.35, linewidth=1.2)

ax.text(0, 1.6, '混合結合（場の空気）', ha='center', fontsize=13,
        fontweight='bold', color='#1a5276')
ax.text(0, -1.5, '場全体が連鎖的に臨界', ha='center', fontsize=11, color='#555')

plt.savefig('/home/claude/reig_resonance_v2.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.savefig('/home/claude/reig_resonance_v2.pdf',
            bbox_inches='tight', facecolor='white')
print("Done!")

# Print summary
print("\n=== Summary ===")
print(f"Scenario 1 (twinRIG): A max E={en1[:,0].max():.2f}, B max E={en1[:,1].max():.2f}")
print(f"  A reaches Ec={E_c}: {np.any(en1[:,0]>=E_c)}, B reaches Ec: {np.any(en1[:,1]>=E_c)}")
print(f"Scenario 2 (ReIG2): Singer max E={en2[:,0].max():.2f}, Listener max E={en2[:,1].max():.2f}")
print(f"  Singer reaches Ec: {np.any(en2[:,0]>=E_c)}, Listener reaches Ec: {np.any(en2[:,1]>=E_c)}")
n_aud_crit = sum(1 for j in range(1,N3) if np.any(en3[:,j]>=E_c))
print(f"Scenario 3 (Live): Performer max E={en3[:,0].max():.2f}")
print(f"  Audience members reaching Ec: {n_aud_crit}/{N3-1}")
