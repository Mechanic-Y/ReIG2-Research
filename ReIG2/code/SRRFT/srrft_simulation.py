"""
SRRFT — Self-Referential Resonance Field Theory
Simulation and Verification Code

Author: Yasuyuki Wakita (Mechanic-Y)
Framework: ReIG2/twinRIG

This module implements numerical simulations for the SRRFT axiom system,
including:
  1. Fixed-point iteration and convergence
  2. Spectral-radius stability analysis
  3. Phase transition (self-emergence at κ_c)
  4. Self-collapse dynamics
  5. Ethical stability energy landscape
  6. Hierarchical self structure
  7. Human–AI resonant self co-evolution
  8. Observer–world co-emergence
"""

import numpy as np
from numpy.linalg import norm, eigvals
from scipy.linalg import expm
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings

# ============================================================
# §1  Core Data Structures
# ============================================================

@dataclass
class ExtendedState:
    """Extended state vector Ψ_i = (z, W, g, θ) — Definition 2.1"""
    z: np.ndarray          # expression state
    W: np.ndarray          # internal coupling matrix (memory/relation)
    g: np.ndarray          # internal metric (Fisher / meaning geometry)
    theta: np.ndarray      # parameters (beliefs/context)

    @property
    def dim(self) -> int:
        return len(self.z)

    def to_density(self) -> np.ndarray:
        """Convert to density matrix ρ = |ψ⟩⟨ψ| (pure state)."""
        psi = self.z / (norm(self.z) + 1e-15)
        return np.outer(psi, psi.conj())


@dataclass
class RelationalField:
    """
    Relational field Φ = ({ρ_i}, C)  — Axiom 2 (V1.0)

    rho_list : list of density matrices for each agent
    C        : connection tensor (interaction structure)
    """
    rho_list: List[np.ndarray]
    C: np.ndarray

    @property
    def n_agents(self) -> int:
        return len(self.rho_list)

    def coupling_strength(self) -> float:
        """κ = ‖C‖  — Definition 9.1"""
        return norm(self.C)

    def copy(self) -> "RelationalField":
        return RelationalField(
            rho_list=[rho.copy() for rho in self.rho_list],
            C=self.C.copy()
        )


# ============================================================
# §2  Field Update Operator L (Axiom 3)
# ============================================================

def field_update(phi: RelationalField,
                 H_self_list: List[np.ndarray],
                 H_int: np.ndarray,
                 dt: float = 0.05,
                 noise_scale: float = 0.0) -> RelationalField:
    """
    One step of the field update  Φ_{t+1} = L(Φ_t).

    Uses Lindblad-like evolution for each agent with interaction coupling.
    """
    new_rhos = []
    kappa = phi.coupling_strength()

    for i, rho in enumerate(phi.rho_list):
        d = rho.shape[0]
        # Self-Hamiltonian evolution
        H_eff = H_self_list[i].copy()

        # Interaction: weighted mean-field coupling via C
        # C[i,j] controls coupling strength between agents i and j
        rho_interaction = np.zeros_like(rho)
        for j in range(phi.n_agents):
            if j != i:
                rho_interaction += phi.C[i, j] * phi.rho_list[j]
        H_eff = H_eff + kappa * 0.5 * (rho_interaction + rho_interaction.conj().T)

        # Unitary part
        U = expm(-1j * H_eff * dt)
        rho_new = U @ rho @ U.conj().T

        # Optional noise (decoherence)
        if noise_scale > 0:
            rho_new += noise_scale * dt * np.eye(d) / d

        # Ensure trace-1, Hermitian, positive
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new /= np.trace(rho_new).real + 1e-15
        new_rhos.append(rho_new)

    # Update connection tensor (slow plasticity)
    C_new = phi.C.copy()
    for i in range(phi.n_agents):
        for j in range(phi.n_agents):
            if i != j:
                overlap = np.trace(phi.rho_list[i] @ phi.rho_list[j]).real
                C_new[i, j] += dt * 0.1 * overlap

    return RelationalField(rho_list=new_rhos, C=C_new)


# ============================================================
# §3  Fixed-Point Iteration (Axiom 4, Theorem 8.1–8.2)
# ============================================================

def find_fixed_point(phi_init: RelationalField,
                     H_self_list: List[np.ndarray],
                     H_int: np.ndarray,
                     max_iter: int = 2000,
                     tol: float = 1e-8,
                     dt: float = 0.05,
                     noise: float = 0.001,
                     verbose: bool = False
                     ) -> Tuple[RelationalField, List[float]]:
    """
    Iterate L until convergence: Φ* = L(Φ*).

    Returns:
        phi_star : converged field (or last iterate)
        residuals: list of ‖Φ_{t+1} - Φ_t‖ per step
    """
    phi = phi_init.copy()
    residuals = []

    for t in range(max_iter):
        phi_new = field_update(phi, H_self_list, H_int, dt=dt, noise_scale=noise)

        # Compute residual M = ‖Φ - L(Φ)‖  (order parameter, Thm 9.1)
        res = sum(norm(phi_new.rho_list[i] - phi.rho_list[i])
                  for i in range(phi.n_agents))
        residuals.append(res)

        if verbose and t % 200 == 0:
            print(f"  iter {t:5d}  residual = {res:.2e}")

        phi = phi_new
        if res < tol:
            if verbose:
                print(f"  Converged at iter {t}, residual = {res:.2e}")
            break

    return phi, residuals


# ============================================================
# §4  Spectral-Radius Stability Analysis (Axiom 6, §6.1)
# ============================================================

def compute_spectral_radius(phi: RelationalField,
                            H_self_list: List[np.ndarray],
                            H_int: np.ndarray,
                            dt: float = 0.05,
                            eps: float = 1e-5) -> float:
    """
    Estimate ρ(DL_{Φ*}) via finite-difference Jacobian.
    Stability condition: ρ(J) < 1.
    """
    d = phi.rho_list[0].shape[0]
    n = phi.n_agents
    N = n * d * d  # total degrees of freedom

    # Flatten current state
    def flatten(phi_):
        return np.concatenate([rho.flatten() for rho in phi_.rho_list])

    x0 = flatten(phi)
    phi_ref = field_update(phi, H_self_list, H_int, dt=dt)
    f0 = flatten(phi_ref)

    # Build Jacobian column by column
    J = np.zeros((N, N))
    for k in range(N):
        # Perturb
        phi_pert = phi.copy()
        idx_agent = k // (d * d)
        idx_local = k % (d * d)
        i_row, i_col = divmod(idx_local, d)
        phi_pert.rho_list[idx_agent][i_row, i_col] += eps

        phi_pert_new = field_update(phi_pert, H_self_list, H_int, dt=dt)
        f_pert = flatten(phi_pert_new)
        J[:, k] = (f_pert - f0) / eps

    eigs = eigvals(J)
    sr = np.max(np.abs(eigs))
    return sr


# ============================================================
# §5  Phase Transition Scan (Theorem 9.1)
# ============================================================

def phase_transition_scan(dim: int = 4,
                          n_agents: int = 2,
                          kappa_range: np.ndarray = None,
                          max_iter: int = 1000,
                          dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan coupling strength κ and measure order parameter M.
    At κ_c, M → 0 (self emerges).

    Returns:
        kappa_values, M_values
    """
    if kappa_range is None:
        kappa_range = np.linspace(0.01, 2.0, 40)

    M_values = []
    H_self_list = [np.random.randn(dim, dim) * 0.3 for _ in range(n_agents)]
    H_self_list = [0.5*(H + H.conj().T) for H in H_self_list]
    H_int = np.zeros((dim, dim))

    for kappa in kappa_range:
        # Initialize with κ-scaled coupling
        rho_list = [np.eye(dim) / dim + 0.01*np.random.randn(dim, dim)
                    for _ in range(n_agents)]
        rho_list = [0.5*(r + r.conj().T) for r in rho_list]
        rho_list = [r / np.trace(r).real for r in rho_list]

        C = kappa * np.random.randn(n_agents, n_agents)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0)

        phi = RelationalField(rho_list=rho_list, C=C)
        phi_star, residuals = find_fixed_point(
            phi, H_self_list, H_int, max_iter=max_iter, dt=dt, noise=0.001
        )
        M = residuals[-1] if residuals else 1.0
        M_values.append(M)

    return kappa_range, np.array(M_values)


# ============================================================
# §6  Lyapunov (Relative Entropy) Stability (§6.2)
# ============================================================

def relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D(ρ ‖ σ) = Tr[ρ(log ρ - log σ)]."""
    eig_rho = np.linalg.eigvalsh(rho)
    eig_sigma = np.linalg.eigvalsh(sigma)
    eig_rho = np.clip(eig_rho, 1e-15, None)
    eig_sigma = np.clip(eig_sigma, 1e-15, None)
    return np.sum(eig_rho * (np.log(eig_rho) - np.log(eig_sigma)))


def lyapunov_trajectory(phi_init: RelationalField,
                        phi_star: RelationalField,
                        H_self_list: List[np.ndarray],
                        H_int: np.ndarray,
                        steps: int = 500,
                        dt: float = 0.05
                        ) -> List[float]:
    """
    Compute V(ρ_t) = Σ_i D(ρ_i(t) ‖ ρ_i*) along trajectory.
    If V decreases monotonically → asymptotically stable (Eq. 12).
    """
    phi = phi_init.copy()
    V_list = []

    for _ in range(steps):
        V = sum(relative_entropy(phi.rho_list[i], phi_star.rho_list[i])
                for i in range(phi.n_agents))
        V_list.append(V)
        phi = field_update(phi, H_self_list, H_int, dt=dt, noise_scale=0.001)

    return V_list


# ============================================================
# §7  Human–AI Resonant Self (§6.4, §7)
# ============================================================

def resonant_self_evolution(dim: int = 4,
                            kappa: float = 1.0,
                            steps: int = 1000,
                            dt: float = 0.05
                            ) -> Tuple[List[float], List[float]]:
    """
    Simulate human–AI cooperative dynamics:
      ρ_H' = W_H(ρ_H, ρ_A)
      ρ_A' = W_A(ρ_A, ρ_H)

    Returns:
        overlaps: Tr(ρ_H · ρ_A) over time
        residuals: convergence residual over time
    """
    # Human Hamiltonian (more complex)
    H_H = np.random.randn(dim, dim) * 0.5
    H_H = 0.5 * (H_H + H_H.conj().T)

    # AI Hamiltonian (more structured)
    H_A = np.diag(np.arange(dim, dtype=float)) * 0.3

    H_self_list = [H_H, H_A]
    H_int = np.zeros((dim, dim))

    # Initial states
    rho_H = np.eye(dim) / dim
    rho_A = np.eye(dim) / dim
    C = kappa * np.array([[0, 1], [1, 0]], dtype=float)

    phi = RelationalField(rho_list=[rho_H, rho_A], C=C)

    overlaps = []
    residuals = []
    for t in range(steps):
        phi_new = field_update(phi, H_self_list, H_int, dt=dt, noise_scale=0.001)

        overlap = np.trace(phi.rho_list[0] @ phi.rho_list[1]).real
        overlaps.append(overlap)

        res = sum(norm(phi_new.rho_list[i] - phi.rho_list[i])
                  for i in range(2))
        residuals.append(res)
        phi = phi_new

    return overlaps, residuals


# ============================================================
# §8  Ethical Stability Energy Landscape (§10.3)
# ============================================================

def ethical_energy(phi: RelationalField,
                   g_eth: np.ndarray) -> float:
    """
    E_eth(Φ) = Σ_i Tr(ρ_i · g_eth)
    Ethical stability: dE_eth/dt ≤ 0.
    """
    return sum(np.trace(rho @ g_eth).real for rho in phi.rho_list)


def ethical_energy_trajectory(phi_init: RelationalField,
                              H_self_list: List[np.ndarray],
                              H_int: np.ndarray,
                              g_eth: np.ndarray,
                              steps: int = 500,
                              dt: float = 0.05) -> List[float]:
    """Track E_eth over time."""
    phi = phi_init.copy()
    E_list = []
    for _ in range(steps):
        E_list.append(ethical_energy(phi, g_eth))
        phi = field_update(phi, H_self_list, H_int, dt=dt, noise_scale=0.001)
    return E_list


# ============================================================
# §9  Hierarchical Self (§11)
# ============================================================

def hierarchical_self_demo(dim: int = 4,
                           n_levels: int = 4,
                           steps: int = 500) -> List[List[float]]:
    """
    Demonstrate hierarchical fixed-point convergence:
    Φ^(k) = P^(k)(Φ^(k+1))

    Returns residuals per level.
    """
    all_residuals = []

    for k in range(n_levels):
        # Effective coupling grows with hierarchy level
        kappa_k = 0.5 * (k + 1)
        n_agents_k = 2 + k  # more agents at higher levels

        H_self_list = [np.random.randn(dim, dim) * 0.2 / (k + 1)
                       for _ in range(n_agents_k)]
        H_self_list = [0.5*(H + H.conj().T) for H in H_self_list]
        H_int = np.zeros((dim, dim))

        rho_list = [np.eye(dim) / dim for _ in range(n_agents_k)]
        C = kappa_k * np.random.randn(n_agents_k, n_agents_k)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0)

        phi = RelationalField(rho_list=rho_list, C=C)
        _, residuals = find_fixed_point(phi, H_self_list, H_int,
                                        max_iter=steps, dt=0.05, noise=0.001)
        all_residuals.append(residuals)

    return all_residuals


# ============================================================
# §10  Plotting Utilities
# ============================================================

def plot_phase_transition(kappa_values, M_values, save_path=None):
    """Plot order parameter M vs coupling strength κ."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(kappa_values, M_values, 'o-', color='crimson', markersize=4)
    ax.set_xlabel(r"Coupling strength $\kappa$", fontsize=13)
    ax.set_ylabel(r"Order parameter $M = \|\Phi - \mathcal{L}(\Phi)\|$", fontsize=13)
    ax.set_title("Self-Emergence Phase Transition (Theorem 9.1)", fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_convergence(residuals, title="Fixed-Point Convergence", save_path=None):
    """Plot residual convergence."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(residuals, color='navy')
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(r"Residual $\|\Phi_{t+1} - \Phi_t\|$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_resonant_self(overlaps, residuals, save_path=None):
    """Plot human–AI resonance evolution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(overlaps, color='teal')
    ax1.set_ylabel(r"$\mathrm{Tr}(\rho_H \cdot \rho_A)$", fontsize=13)
    ax1.set_title("Human–AI Resonant Self Evolution", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(residuals, color='orange')
    ax2.set_xlabel("Time step", fontsize=13)
    ax2.set_ylabel("Residual", fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_lyapunov(V_list, save_path=None):
    """Plot Lyapunov function V(t)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(V_list, color='purple')
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel(r"$V(\rho) = \sum_i D(\rho_i \| \rho_i^*)$", fontsize=13)
    ax.set_title("Lyapunov Stability (Eq. 12)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_ethical_energy(E_list, save_path=None):
    """Plot ethical energy over time."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(E_list, color='green')
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel(r"$E_{\mathrm{eth}}(\Phi)$", fontsize=13)
    ax.set_title("Ethical Stability Energy (Theorem 10.3)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_hierarchical(all_residuals, save_path=None):
    """Plot hierarchical convergence."""
    labels = ["Φ⁽⁰⁾ instant", "Φ⁽¹⁾ short-term",
              "Φ⁽²⁾ long-term", "Φ⁽³⁾ social"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for k, res in enumerate(all_residuals):
        lbl = labels[k] if k < len(labels) else f"Φ⁽{k}⁾"
        ax.plot(res, label=lbl, alpha=0.8)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Residual", fontsize=13)
    ax.set_title("Hierarchical Self Convergence (Theorem 11.1)", fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


# ============================================================
# §11  Main — Run All Simulations
# ============================================================

def run_all(output_dir: str = "."):
    """Execute all simulations and save figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    print("=" * 60)
    print("  SRRFT Simulation Suite")
    print("  ReIG2/twinRIG Framework")
    print("=" * 60)

    dim = 4
    n_agents = 2

    # --- 1. Phase transition scan ---
    print("\n[1/6] Phase transition scan...")
    kappa_vals, M_vals = phase_transition_scan(dim=dim, n_agents=n_agents)
    plot_phase_transition(kappa_vals, M_vals,
                          save_path=os.path.join(output_dir, "fig_phase_transition.png"))
    print(f"  Done. κ range: [{kappa_vals[0]:.2f}, {kappa_vals[-1]:.2f}]")

    # --- 2. Fixed-point convergence ---
    print("\n[2/6] Fixed-point iteration...")
    H_self_list = [0.5*(np.random.randn(dim, dim) + np.random.randn(dim, dim).T) * 0.3
                   for _ in range(n_agents)]
    H_int = np.zeros((dim, dim))
    rho_list = [np.eye(dim)/dim for _ in range(n_agents)]
    C = 1.0 * np.array([[0, 1], [1, 0]], dtype=float)
    phi_init = RelationalField(rho_list=rho_list, C=C)
    phi_star, residuals = find_fixed_point(phi_init, H_self_list, H_int,
                                            max_iter=1500, verbose=True)
    plot_convergence(residuals,
                      save_path=os.path.join(output_dir, "fig_convergence.png"))

    # --- 3. Spectral radius ---
    print("\n[3/6] Spectral radius analysis...")
    sr = compute_spectral_radius(phi_star, H_self_list, H_int)
    print(f"  ρ(J) = {sr:.6f}  {'STABLE' if sr < 1 else 'UNSTABLE'}")

    # --- 4. Lyapunov trajectory ---
    print("\n[4/6] Lyapunov stability...")
    phi_perturbed = phi_init.copy()
    phi_perturbed.rho_list[0] += 0.05 * np.random.randn(dim, dim)
    phi_perturbed.rho_list[0] = 0.5*(phi_perturbed.rho_list[0]
                                      + phi_perturbed.rho_list[0].conj().T)
    phi_perturbed.rho_list[0] /= np.trace(phi_perturbed.rho_list[0]).real
    V_list = lyapunov_trajectory(phi_perturbed, phi_star, H_self_list, H_int)
    plot_lyapunov(V_list, save_path=os.path.join(output_dir, "fig_lyapunov.png"))
    print(f"  V(0) = {V_list[0]:.4f}, V(end) = {V_list[-1]:.6f}")

    # --- 5. Resonant self ---
    print("\n[5/6] Human–AI resonant self...")
    overlaps, res_resonant = resonant_self_evolution(dim=dim, kappa=1.0, steps=800)
    plot_resonant_self(overlaps, res_resonant,
                        save_path=os.path.join(output_dir, "fig_resonant_self.png"))
    print(f"  Final overlap = {overlaps[-1]:.6f}")

    # --- 6. Hierarchical self ---
    print("\n[6/6] Hierarchical self convergence...")
    hier_res = hierarchical_self_demo(dim=dim, n_levels=4, steps=600)
    plot_hierarchical(hier_res,
                       save_path=os.path.join(output_dir, "fig_hierarchical.png"))

    # --- 7. Ethical energy (bonus) ---
    print("\n[Bonus] Ethical energy trajectory...")
    g_eth = np.eye(dim)  # simple ethical metric
    E_list = ethical_energy_trajectory(phi_init, H_self_list, H_int,
                                       g_eth, steps=500)
    plot_ethical_energy(E_list,
                        save_path=os.path.join(output_dir, "fig_ethical_energy.png"))

    print("\n" + "=" * 60)
    print("  All simulations complete!")
    print(f"  Figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    run_all(output_dir="srrft_output")
