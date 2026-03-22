"""
twinRIG V1 - Demonstration Script
==================================

This script demonstrates the core functionality of the twinRIG V1 framework:
1. Quantum state initialization
2. Extended time evolution with resonance parameters
3. Phase transition operations
4. Self-referential fixed point convergence

Reference: ReIG2_twinRIG_2025_December.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import QuantumState, TimeEvolutionEngine, WorldOperator
from operators import HamiltonianFactory, PhaseOperator, KrausChannel


def demo_1_basic_state():
    """Demonstrate basic quantum state operations."""
    print("\n" + "="*60)
    print("Demo 1: Basic Quantum State Operations")
    print("="*60)
    
    # Create a quantum state in 5-subspace Hilbert space
    subspace_dims = (3, 3, 3, 3, 3)  # 3^5 = 243 dimensions
    state = QuantumState(subspace_dims=subspace_dims)
    
    print(f"\nState Configuration:")
    print(f"  - Subspace dimensions: {subspace_dims}")
    print(f"  - Total dimension: {state.total_dim}")
    print(f"  - Initial norm: {state.norm():.6f}")
    
    # Measure each subspace
    print(f"\nSubspace Probability Distributions:")
    subspace_names = ['Meaning', 'Context', 'Ethics', 'Future', 'Stability']
    for i, name in enumerate(subspace_names):
        probs = state.measure_subspace(i)
        print(f"  {name}: {probs}")
    
    # Calculate fidelity with itself
    print(f"\nSelf-fidelity: {state.fidelity(state):.6f}")
    
    return state


def demo_2_hamiltonian_construction():
    """Demonstrate Hamiltonian construction."""
    print("\n" + "="*60)
    print("Demo 2: Hamiltonian Construction (Eq. 5)")
    print("="*60)
    
    dim = 243
    subspace_dims = (3, 3, 3, 3, 3)
    
    factory = HamiltonianFactory(dim, subspace_dims, coupling_strength=0.3, seed=42)
    hamiltonians = factory.create_all_hamiltonians()
    
    print("\nHamiltonian Components:")
    print("-" * 40)
    
    for name, H in hamiltonians.items():
        eigenvalues = np.linalg.eigvalsh(H)
        print(f"\n{name}:")
        print(f"  Shape: {H.shape}")
        print(f"  Hermitian: {np.allclose(H, H.conj().T)}")
        print(f"  Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
        print(f"  Non-zero elements: {np.count_nonzero(H)} / {H.size}")
    
    # Create extended Hamiltonian
    print("\n" + "-" * 40)
    print("Extended Hamiltonian (Eq. 5):")
    print("Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics")
    
    tau, epsilon, pfh = 0.5, 0.3, 0.2
    H_ext = factory.create_extended_hamiltonian(tau, epsilon, pfh)
    eigenvalues = np.linalg.eigvalsh(H_ext)
    
    print(f"\nParameters: τ={tau}, ε={epsilon}, PFH={pfh}")
    print(f"Extended H eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    
    return factory


def demo_3_time_evolution():
    """Demonstrate quantum time evolution."""
    print("\n" + "="*60)
    print("Demo 3: Quantum Time Evolution (Eq. 17)")
    print("="*60)
    
    dim = 243
    subspace_dims = (3, 3, 3, 3, 3)
    
    # Create Hamiltonians
    factory = HamiltonianFactory(dim, subspace_dims, seed=42)
    H_terms = list(factory.create_all_hamiltonians().values())
    
    # Create evolution engine with Trotter decomposition
    engine = TimeEvolutionEngine(H_terms, dt=0.01, trotter_order=2)
    
    # Initial state
    state = QuantumState(subspace_dims=subspace_dims)
    
    print(f"\nEvolution Parameters:")
    print(f"  Time step: {engine.dt}")
    print(f"  Trotter order: {engine.trotter_order}")
    print(f"  Initial norm: {state.norm():.6f}")
    
    # Evolve
    coefficients = [1.0, 0.5, 0.3, 0.2]  # [1, τ, ε, PFH]
    n_steps = 100
    
    print(f"\nEvolving for {n_steps} steps...")
    
    meaning_probs_history = []
    norms_history = []
    
    def callback(step, current_state):
        if step % 20 == 0:
            probs = current_state.measure_subspace(0)  # Meaning
            meaning_probs_history.append(probs[0])
            norms_history.append(current_state.norm())
    
    final_state, trajectory = engine.evolve(state, n_steps, coefficients, callback)
    
    print(f"\nResults:")
    print(f"  Final norm: {final_state.norm():.6f}")
    print(f"  Fidelity with initial: {final_state.fidelity(state):.6f}")
    
    # Show meaning subspace evolution
    print(f"\n  Meaning P(|0⟩) evolution:")
    for i, (prob, norm) in enumerate(zip(meaning_probs_history, norms_history)):
        print(f"    Step {i*20}: P(|0⟩)={prob:.4f}, norm={norm:.6f}")
    
    return trajectory


def demo_4_phase_transition():
    """Demonstrate phase transition operations."""
    print("\n" + "="*60)
    print("Demo 4: Phase Transition Operator G (Eq. 25)")
    print("="*60)
    
    dim = 243
    
    # Create phase operator
    phase_op = PhaseOperator(
        dim=dim,
        rotation_angle=np.pi/4,
        expansion_rate=0.1,
        jump_probability=0.3
    )
    
    print(f"\nPhase Operator Configuration:")
    print(f"  Rotation angle: π/4")
    print(f"  Expansion rate: 0.1")
    print(f"  Jump probability: 0.3")
    
    # Create random state
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state = state / np.linalg.norm(state)
    
    print(f"\nApplying G = P ∘ E ∘ R:")
    print(f"  Input norm: {np.linalg.norm(state):.6f}")
    
    # Apply each component
    state_r = phase_op.rotation(state)
    print(f"  After R (rotation): norm={np.linalg.norm(state_r):.6f}")
    
    state_e = phase_op.expansion(state_r)
    print(f"  After E (expansion): norm={np.linalg.norm(state_e):.6f}")
    
    state_p = phase_op.phase_jump(state_e, apply_stochastic=False)
    print(f"  After P (jump): norm={np.linalg.norm(state_p):.6f}")
    
    # Full application with normalization
    state_final = phase_op.apply(state, apply_jump_stochastic=False)
    print(f"  Final (normalized): norm={np.linalg.norm(state_final):.6f}")
    
    # Overlap with original
    overlap = np.abs(np.vdot(state, state_final))
    print(f"\n  Overlap |⟨ψ|Gψ⟩|: {overlap:.6f}")


def demo_5_fixed_point_convergence():
    """Demonstrate self-referential fixed point convergence."""
    print("\n" + "="*60)
    print("Demo 5: Fixed Point Convergence |I⟩ (Theorem 5.1)")
    print("="*60)
    
    subspace_dims = (3, 3, 3, 3, 3)
    dim = int(np.prod(subspace_dims))
    
    # Create world operator with contraction factor κ < 1
    world_op = WorldOperator(
        dim=dim,
        subspace_dims=subspace_dims,
        contraction_factor=0.9  # κ = 0.9 < 1
    )
    
    print(f"\nWorld Operator Configuration:")
    print(f"  Contraction factor κ: {world_op.contraction_factor}")
    print(f"  Condition: κ < 1 ✓ (Banach contraction)")
    
    # Initial state
    initial_state = QuantumState(subspace_dims=subspace_dims)
    
    print(f"\nFinding fixed point |I⟩ such that T̂_World|I⟩ = |I⟩...")
    
    fixed_point, iterations, history = world_op.find_fixed_point(
        initial_state,
        max_iterations=500,
        tolerance=1e-8
    )
    
    print(f"\nResults:")
    print(f"  Converged in {iterations} iterations")
    print(f"  Final distance: {history[-1]:.2e}")
    print(f"  Fixed point norm: {fixed_point.norm():.6f}")
    
    # Verify fixed point property
    applied = world_op.apply(fixed_point)
    residual = np.linalg.norm(applied.amplitudes - fixed_point.amplitudes)
    print(f"  Verification ||T̂|I⟩ - |I⟩||: {residual:.2e}")
    
    # Show convergence
    print(f"\n  Convergence history (selected steps):")
    steps_to_show = [0, 1, 5, 10, 50, 100, min(iterations-1, 200)]
    for step in steps_to_show:
        if step < len(history):
            print(f"    Step {step}: distance = {history[step]:.6e}")
    
    return history


def demo_6_kraus_channels():
    """Demonstrate quantum channels."""
    print("\n" + "="*60)
    print("Demo 6: Kraus Channels (Section 7.2)")
    print("="*60)
    
    dim = 16  # Smaller for clarity
    
    # Create a pure state density matrix
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state = state / np.linalg.norm(state)
    rho = np.outer(state, state.conj())
    
    print(f"\nInitial State:")
    print(f"  Purity Tr(ρ²): {np.real(np.trace(rho @ rho)):.6f}")
    print(f"  Trace: {np.real(np.trace(rho)):.6f}")
    
    # Dephasing channel
    print(f"\n1. Dephasing Channel (T2 process):")
    for gamma in [0.1, 0.3, 0.5]:
        kraus_ops = KrausChannel.dephasing_channel(dim, gamma)
        rho_out = KrausChannel.apply_channel(rho, kraus_ops)
        purity = np.real(np.trace(rho_out @ rho_out))
        print(f"   γ={gamma}: Purity={purity:.4f}, Trace={np.real(np.trace(rho_out)):.4f}")
    
    # Amplitude damping
    print(f"\n2. Amplitude Damping Channel (T1 process):")
    for gamma in [0.1, 0.3, 0.5]:
        kraus_ops = KrausChannel.amplitude_damping_channel(dim, gamma)
        rho_out = KrausChannel.apply_channel(rho, kraus_ops)
        purity = np.real(np.trace(rho_out @ rho_out))
        print(f"   γ={gamma}: Purity={purity:.4f}, Trace={np.real(np.trace(rho_out)):.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("twinRIG V1 - Comprehensive Demonstration")
    print("ReIG2/twinRIG: Quantum-Resonance AI Dialogue System")
    print("="*60)
    
    # Run demos
    demo_1_basic_state()
    demo_2_hamiltonian_construction()
    demo_3_time_evolution()
    demo_4_phase_transition()
    convergence_history = demo_5_fixed_point_convergence()
    demo_6_kraus_channels()
    
    print("\n" + "="*60)
    print("All demonstrations completed successfully!")
    print("="*60)
    
    # Optional: Plot convergence
    try:
        plt.figure(figsize=(10, 5))
        plt.semilogy(convergence_history)
        plt.xlabel('Iteration')
        plt.ylabel('Distance to fixed point')
        plt.title('Fixed Point Convergence (Theorem 5.1)')
        plt.grid(True, alpha=0.3)
        plt.savefig('convergence_v1.png', dpi=150, bbox_inches='tight')
        print("\nConvergence plot saved to: convergence_v1.png")
    except Exception as e:
        print(f"\n(Plotting skipped: {e})")


if __name__ == "__main__":
    main()
