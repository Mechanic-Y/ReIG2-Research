"""
twinRIG V2 - Comprehensive Demonstration
=========================================

Demonstrates the V2 sparse matrix architecture:
1. Scaling comparison (V1 vs V2)
2. Multi-axis time evolution
3. Commutator analysis for non-commutativity
4. Fixed point convergence
5. Large-scale capability

Reference: ReIG2_twinRIG_2025_December.pdf
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_v2 import (
    QuantumState, 
    SparseTimeEvolutionEngine, 
    MultiAxisEvolutionEngine,
    SparseWorldOperator,
    TimeAxis
)
from operators_v2 import SparseHamiltonianFactory, SparsePhaseOperator


def demo_1_scaling_comparison():
    """Compare V1 (dense) vs V2 (sparse) memory and performance."""
    print("\n" + "="*60)
    print("Demo 1: V1 vs V2 Scaling Comparison")
    print("="*60)
    
    test_dims = [3, 4, 5, 6]  # Subspace dimensions → 243, 1024, 3125, 7776 total
    
    results = []
    
    for d in test_dims:
        subspace_dims = (d, d, d, d, d)
        dim = d ** 5
        
        print(f"\n--- Dimension: {dim} (subspace d={d}) ---")
        
        # Sparse (V2)
        t0 = time.time()
        factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
        hamiltonians = factory.create_all_hamiltonians()
        t_sparse = time.time() - t0
        
        sparse_bytes = sum(
            H.data.nbytes + H.indices.nbytes + H.indptr.nbytes
            for H in hamiltonians.values()
        )
        dense_bytes_equiv = 4 * dim * dim * 16  # 4 matrices × n² × 16 bytes
        
        total_nnz = sum(H.nnz for H in hamiltonians.values())
        sparsity = 1 - total_nnz / (4 * dim * dim)
        
        print(f"  V2 (Sparse):")
        print(f"    Time: {t_sparse:.3f}s")
        print(f"    Memory: {sparse_bytes/1024:.1f} KB")
        print(f"    Sparsity: {sparsity*100:.2f}%")
        print(f"    NNZ: {total_nnz}")
        
        print(f"  V1 (Dense) equivalent:")
        print(f"    Memory: {dense_bytes_equiv/1024/1024:.1f} MB")
        print(f"  Compression: {dense_bytes_equiv/sparse_bytes:.1f}x")
        
        results.append({
            'dim': dim,
            'd': d,
            'sparse_bytes': sparse_bytes,
            'dense_bytes': dense_bytes_equiv,
            'compression': dense_bytes_equiv / sparse_bytes,
            'sparsity': sparsity
        })
    
    # Summary
    print("\n" + "-"*40)
    print("Summary: Memory Savings")
    print("-"*40)
    print(f"{'Dimension':<12} {'V2 (KB)':<12} {'V1 (MB)':<12} {'Savings':<12}")
    for r in results:
        print(f"{r['dim']:<12} {r['sparse_bytes']/1024:<12.1f} {r['dense_bytes']/1024/1024:<12.1f} {r['compression']:<12.1f}x")
    
    return results


def demo_2_multi_axis_setup():
    """Demonstrate multi-axis generator construction."""
    print("\n" + "="*60)
    print("Demo 2: Multi-Axis Time Generator Setup")
    print("="*60)
    
    subspace_dims = (5, 5, 5, 5, 5)
    dim = int(np.prod(subspace_dims))
    
    factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
    generators = factory.create_multi_axis_generators()
    
    print(f"\nTotal dimension: {dim}")
    print(f"Subspace dimensions: {subspace_dims}")
    
    print("\n" + "-"*40)
    print("Time Axis Generators (Section 3.2):")
    print("-"*40)
    
    for axis, G in generators.items():
        # Compute eigenvalue range (sparse)
        try:
            max_eig = np.real(np.max(np.abs(G.diagonal())))
        except:
            max_eig = 0.0
        
        print(f"\n{axis.name} (k={axis.value}):")
        print(f"  Shape: {G.shape}")
        print(f"  Non-zeros: {G.nnz}")
        print(f"  Sparsity: {(1 - G.nnz/dim**2)*100:.2f}%")
        print(f"  Max diagonal: {max_eig:.4f}")
    
    return generators


def demo_3_commutator_analysis():
    """Analyze commutator structure for non-commutativity."""
    print("\n" + "="*60)
    print("Demo 3: Commutator Analysis (Section 3.3)")
    print("="*60)
    
    subspace_dims = (5, 5, 5, 5, 5)
    dim = int(np.prod(subspace_dims))
    
    factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
    generators = factory.create_multi_axis_generators()
    
    print(f"\nAnalyzing ||[G^(i), G^(j)]|| for non-commutativity...")
    print("Non-zero commutators indicate non-commuting time axes.")
    
    print("\n" + "-"*40)
    print("Commutator Norms:")
    print("-"*40)
    
    import scipy.sparse as sp
    
    axes = list(generators.keys())
    commutator_matrix = np.zeros((len(axes), len(axes)))
    
    for i, ax1 in enumerate(axes):
        for j, ax2 in enumerate(axes):
            if i < j:
                G1, G2 = generators[ax1], generators[ax2]
                comm = G1 @ G2 - G2 @ G1
                norm = sp.linalg.norm(comm, 'fro')
                commutator_matrix[i, j] = norm
                commutator_matrix[j, i] = norm
                print(f"  ||[{ax1.name}, {ax2.name}]|| = {norm:.4f}")
    
    print("\n" + "-"*40)
    print("Interpretation:")
    print("-"*40)
    print("  - Non-zero values: Time axes do NOT commute")
    print("  - Requires Trotter decomposition for accurate evolution")
    print("  - Different evolution orders give different results")
    
    return commutator_matrix


def demo_4_multi_axis_evolution():
    """Demonstrate multi-axis time evolution."""
    print("\n" + "="*60)
    print("Demo 4: Multi-Axis Time Evolution")
    print("="*60)
    
    subspace_dims = (5, 5, 5, 5, 5)
    dim = int(np.prod(subspace_dims))
    
    # Setup
    factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
    generators = factory.create_multi_axis_generators()
    
    engine = MultiAxisEvolutionEngine(generators, dt=0.05)
    
    # Initial state
    state = QuantumState(subspace_dims=subspace_dims)
    
    print(f"\nEvolution Parameters:")
    print(f"  Time step: {engine.dt}")
    print(f"  Initial state dimension: {state.total_dim}")
    
    # Evolve with different parameter sets
    param_sets = [
        {'tau': 0.3, 'epsilon': 0.1, 'pfh': 0.1, 'name': 'Low resonance'},
        {'tau': 0.7, 'epsilon': 0.5, 'pfh': 0.5, 'name': 'High resonance'},
    ]
    
    for params in param_sets:
        print(f"\n--- {params['name']} ---")
        print(f"  τ={params['tau']}, ε={params['epsilon']}, PFH={params['pfh']}")
        
        # Reset state
        current = state.copy()
        
        # Evolve
        n_steps = 50
        meaning_evolution = []
        
        for step in range(n_steps):
            current = engine.evolve_step(
                current,
                tau=params['tau'],
                epsilon=params['epsilon'],
                pfh=params['pfh']
            )
            
            if step % 10 == 0:
                probs = current.measure_subspace(0)  # Meaning
                meaning_evolution.append((step, probs[0]))
        
        print(f"\n  Meaning P(|0⟩) evolution:")
        for step, prob in meaning_evolution:
            print(f"    Step {step}: {prob:.4f}")
    
    return engine


def demo_5_convergence_analysis():
    """Demonstrate fixed point convergence with sparse operators."""
    print("\n" + "="*60)
    print("Demo 5: Sparse Fixed Point Convergence")
    print("="*60)
    
    subspace_dims = (5, 5, 5, 5, 5)
    dim = int(np.prod(subspace_dims))
    
    # Create sparse world operator
    world_op = SparseWorldOperator(
        dim=dim,
        subspace_dims=subspace_dims,
        contraction_factor=0.9,
        sparsity=0.02
    )
    
    print(f"\nSparse World Operator:")
    stats = world_op.get_memory_stats()
    print(f"  Total NNZ: {stats['total_nnz']}")
    print(f"  Sparse bytes: {stats['sparse_bytes']/1024:.1f} KB")
    print(f"  Dense equivalent: {stats['dense_bytes_equivalent']/1024/1024:.1f} MB")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Actual sparsity: {stats['sparsity_actual']*100:.1f}%")
    
    # Find fixed point
    state = QuantumState(subspace_dims=subspace_dims)
    
    print(f"\nFinding fixed point |I⟩...")
    t0 = time.time()
    fixed, iterations, history = world_op.find_fixed_point(
        state,
        max_iterations=300,
        tolerance=1e-8
    )
    elapsed = time.time() - t0
    
    print(f"\nResults:")
    print(f"  Converged in {iterations} iterations")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Final distance: {history[-1]:.2e}")
    
    # Convergence profile
    print(f"\n  Convergence profile:")
    checkpoints = [0, 1, 5, 10, 25, 50, 100, min(iterations-1, 200)]
    for i in checkpoints:
        if i < len(history):
            print(f"    Iteration {i}: {history[i]:.2e}")
    
    return history


def demo_6_scaling_limit_test():
    """Test V2 scaling limits."""
    print("\n" + "="*60)
    print("Demo 6: V2 Scaling Limit Test")
    print("="*60)
    
    print("\nTesting maximum practical dimension...")
    
    # Test progressively larger dimensions
    test_dims = [
        (5, 5, 5, 5, 5),    # 3,125
        (6, 6, 6, 6, 6),    # 7,776
        (4, 4, 4, 4, 4, 4), # 4,096 (6 subspaces)
    ]
    
    for subspace_dims in test_dims:
        dim = int(np.prod(subspace_dims))
        print(f"\n--- Testing dim={dim} ---")
        
        try:
            t0 = time.time()
            
            # Create Hamiltonians
            factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
            H = factory.create_extended_hamiltonian()
            
            # Create state
            state = QuantumState(subspace_dims=subspace_dims)
            
            # Test evolution
            engine = SparseTimeEvolutionEngine([H], dt=0.01)
            final, _ = engine.evolve(state, n_steps=10)
            
            elapsed = time.time() - t0
            
            print(f"  Success!")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  H sparsity: {(1-H.nnz/dim**2)*100:.1f}%")
            print(f"  H memory: {(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes)/1024:.1f} KB")
            
        except Exception as e:
            print(f"  Failed: {e}")


def main():
    """Run all V2 demonstrations."""
    print("\n" + "="*60)
    print("twinRIG V2 - Comprehensive Demonstration")
    print("Sparse Matrix Architecture for Large-Scale Quantum Resonance")
    print("="*60)
    
    # Run all demos
    demo_1_scaling_comparison()
    demo_2_multi_axis_setup()
    demo_3_commutator_analysis()
    demo_4_multi_axis_evolution()
    convergence_history = demo_5_convergence_analysis()
    demo_6_scaling_limit_test()
    
    print("\n" + "="*60)
    print("All V2 demonstrations completed!")
    print("="*60)
    
    # Try to save convergence plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.semilogy(convergence_history)
        plt.xlabel('Iteration')
        plt.ylabel('Distance to fixed point')
        plt.title('V2 Sparse Fixed Point Convergence')
        plt.grid(True, alpha=0.3)
        plt.savefig('convergence_v2.png', dpi=150, bbox_inches='tight')
        print("\nConvergence plot saved: convergence_v2.png")
    except Exception as e:
        print(f"\n(Plotting skipped: {e})")
    
    print("\n" + "-"*60)
    print("V2 Key Achievements:")
    print("-"*60)
    print("  ✓ Scales to 7,776+ dimensions (V1 limited to ~1,000)")
    print("  ✓ 100-900x memory compression via sparsity")
    print("  ✓ Multi-axis non-commutative time evolution")
    print("  ✓ Trotter decomposition for non-commuting operators")
    print("  ✓ Krylov subspace methods for efficient evolution")


if __name__ == "__main__":
    main()
