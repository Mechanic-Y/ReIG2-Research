# ReIG2/twinRIG Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **A Rigorous Quantum-Mechanical Framework for Self-Reference and World Construction**

**🌐 Website**: [https://mechanic-y.github.io/ReIG2-Research/](https://mechanic-y.github.io/ReIG2-Research/)

---

## 📖 Overview

ReIG2/twinRIG is a comprehensive quantum-mechanical framework for modeling self-referential cognition and world construction. The theory progresses through four stages:

1. **Extended Time Evolution Operator** - Adding future possibility, fluctuation, and ethics to quantum mechanics
2. **Multi-dimensional Time Evolution** - Multiple time axes (physical, cultural, social, personal)
3. **Phase Transition Generation Operator** - Discrete state transitions and emergence
4. **World Generation Tensor System** - Unified framework with self-referential fixed points

### Key Mathematical Features

- ✅ **Fock Space Formalism**: Proper treatment of infinite-dimensional Hilbert spaces
- ✅ **Banach Fixed Point Theorem**: Rigorous proof of convergence with contraction κ < 1
- ✅ **Trotter-Suzuki Decomposition**: Explicit handling of non-commutative operators
- ✅ **Non-Unitary Processes**: Kraus operators, Lindblad master equations
- ✅ **Multi-Axis Non-Commutative Time**: Four temporal dimensions with ||[G^(i), G^(j)]|| > 0

---

## 📄 Publications

### Latest: December 2025 Comprehensive Edition

**Title**: ReIG2/twinRIG: 包括的フレームワーク — 時間発展から相転移生成へ

**Author**: Mechanic-Y / Yasuyuki Wakita

**Abstract**: This paper presents ReIG2/twinRIG, an integrated theoretical framework that progressively develops from standard quantum mechanical time evolution operators to extended time evolution operators, multidimensional time evolution operators, and phase transition generation operators.

📄 **[Download PDF](papers/ReIG2_twinRIG_2025_December.pdf)**

---

## 🏗️ Project Structure

```
ReIG2-Research/
├── papers/
│   └── ReIG2_twinRIG_2025_December.pdf    # Main paper (December 2025)
├── code/
|   ├── non_unitary_quantum.py      # Kraus, Lindblad implementations
│   ├── reig2_full_simulation.py    # Complete 3-qubit system
│   ├── quantum_circuit_implementation.py  # Qiskit circuits
│   ├── figure_generation.py        # Reproducible visualizations 
│   ├── v1/                                 # Dense matrix implementation
│   │   ├── engine.py                       # Quantum state & evolution
│   │   ├── operators.py                    # Hamiltonians & phase operators
│   │   └── demo.py                         # V1 demonstration
│   └── v2/                                 # Sparse matrix implementation
│       ├── engine_v2.py                    # Scalable sparse engine
│       ├── operators_v2.py                 # Sparse operators
│       └── demo_v2.py                      # V2 demonstration
├── images/
│   ├── fig1_system_architecture.png
│   ├── fig2_convergence.png
│   ├── fig3_nonunitary.png
│   ├── fig4_circuit.png
│   └── fig5_functor.png
├── docs/
|   ├── mathematical_proofs.md      # Detailed theorem proofs
│   ├── implementation_notes.md     # Code documentation
│   ├── hardware_requirements.md    # Quantum hardware specs
│   └── v3/                                 # V3 AI Partner Framework
│       ├── 01_V3_Architecture.md           # System architecture
│       ├── 02_WorldTensor_Core.md          # World tensor core
│       ├── 03_ReIG2_V3_Interaction_Model.md
│       ├── 04_V1V2_Safety_Guide.md
│       ├── 05_V3_Response_Framework.md
│       ├── 06_V3_Examples_and_Patterns.md
│       └── 07_V3_Limitations_and_Policies.md
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run V1 Demo (Dense Implementation)

```bash
cd code/v1
python demo.py
```

**V1 Features**:
- Dense matrix operations
- Suitable for dimensions ≤ 1,000
- Clear, educational implementation
- Full Hamiltonian construction

### Run V2 Demo (Sparse Implementation)

```bash
cd code/v2
python demo_v2.py
```

**V2 Features**:
- Sparse matrix operations (CSR format)
- Scales to 30,000+ dimensions
- Multi-axis time evolution
- Krylov subspace methods
- 100-900x memory compression

---

## 📐 Mathematical Framework

### Extended Time Evolution Operator (Section 2)

Standard quantum mechanics:
```
U(t) = exp(-iHt/ℏ)
```

Extended with three resonance parameters:
```
Û_res(t; τ, ε, PFH) = exp(-iĤ(t, τ, ε, PFH)/ℏ)

Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics
```

### Multi-dimensional Time (Section 3)

Multiple time axes with weight functions:
```
Û_multi = exp(-i Σₖ Ĥₖ fₖ(τ, ε, PFH) / ℏ)
```

Where:
- k=0: Physical time
- k=1: Cultural time
- k=2: Social time
- k=3: Personal time

Non-commutativity: ||[Ĥₖ, Ĥₖ']|| > 0

### Phase Transition Operator (Section 4)

Discrete state transitions:
```
G = P ∘ E ∘ R

R: Torsion (rotation)
E: Expansion
P: Phase jump
```

### World Construction Operator (Section 5)

Complete transformation chain:
```
T̂_World = T_I ∘ T_R ∘ T_C ∘ Û_multi ∘ Û_res
```

Fixed point convergence (Theorem 5.1):
```
lim_{N→∞} T̂_Self^(N) |Ψ₀⟩ = |I⟩
```

---

## 📊 Performance Comparison

| Metric | V1 (Dense) | V2 (Sparse) |
|--------|-----------|-------------|
| Max Dimension | ~1,000 | 30,000+ |
| Memory | O(n²) | O(nnz) |
| Evolution | O(n³) | O(nnz·m) |
| Sparsity Support | ❌ | ✅ |
| Multi-axis Time | ❌ | ✅ |

**Typical Compression**: 100-900x for dimensions > 1000

---

## 🤖 V3: AI Thinking Partner Framework

ReIG2 V3 extends the quantum-resonance framework into an AI dialogue system architecture:

- **World Tensor Layer**: Integrates meaning, context, ethics, future, and stability spaces
- **Cognitive Partnership**: AI as a thinking partner, not an answer machine
- **Safety Integration**: Twin-gate system (Intent Gate + Content Gate)

See `docs/v3/` for complete documentation.

---

## 📚 References

1. Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010)
2. Hofstadter, *Gödel, Escher, Bach: An Eternal Golden Braid* (1979)
3. Friston, "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience* (2010)
4. Banach, "Sur les opérations dans les ensembles abstraits" (1922)
5. Trotter, "On the product of semi-groups of operators" (1959)
6. Lindblad, "On the generators of quantum dynamical semigroups" (1976)

---

## 🔮 Future Directions

### Theoretical
- Non-commutative generator extensions
- Full Lindblad integration
- Tensor network methods

### Experimental
- Cognitive neuroscience validation (fMRI, EEG)
- Quantum hardware implementation (IBM Quantum, IonQ)
- Parameter estimation from behavioral data

### Applications
- Variational quantum algorithms
- Quantum-inspired machine learning
- Consciousness modeling

---

## 📝 Citation

```bibtex
@article{wakita2025reig2,
  title={ReIG2/twinRIG: A Comprehensive Framework from Time Evolution to Phase Transition Generation},
  author={Wakita, Yasuyuki (Mechanic-Y)},
  year={2025},
  month={December}
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Mechanic-Y / Yasuyuki Wakita**

- GitHub: [@Mechanic-Y](https://github.com/Mechanic-Y)
- Website: [mechanic-y.github.io](https://mechanic-y.github.io)

---

## 🙏 Acknowledgments

The development of this framework was significantly aided by dialogue with ChatGPT（openAI）Gemini（Google）Claude (Anthropic). The mathematical rigorization, implementation verification, and integration of perspectives at each stage benefited from valuable insights.

---

*Built with ❤️ for quantum cognitive science*

---

<p align="center">
  <b>Built with ❤️ for quantum cognitive science</b>
</p>
