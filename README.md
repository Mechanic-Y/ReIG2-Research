# ReIG2/twinRIG Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

> A Rigorous Quantum-Mechanical Framework for Self-Reference and World Construction

**🌐 Website**: [https://mechanic-y.github.io/ReIG2-Research/](https://mechanic-y.github.io/ReIG2-Research/)

---

## 📖 Overview

ReIG2/twinRIG is a quantum-mechanical model of self-referential cognition and world construction. This repository contains the complete implementation, documentation, and supplementary materials for the revised edition (2025).

### Key Features

- ✅ **Fock Space Formalism**: Proper treatment of infinite-dimensional Hilbert spaces
- ✅ **Banach Fixed Point Theorem**: Rigorous proof of convergence with contraction κ < 1
- ✅ **Trotter Decomposition**: Explicit handling of non-commutativity
- ✅ **Non-Unitary Processes**: Kraus operators, Lindblad master equations
- ✅ **Complete Simulations**: Python implementations with visualization
- ✅ **Quantum Circuits**: Qiskit-compatible implementations for real hardware

---

## 📄 Publications

### Latest: Revised Edition (2025.11.29)

**Title**: ReIG2/twinRIG: A Rigorous Quantum-Mechanical Framework for Self-Reference and World Construction

**Author**: Mechanic-Y / Yasuyuki Wakita

**Abstract**: We present a mathematically rigorous reformulation addressing all major criticisms of the original framework...

**Links**:
- [📄 PDF](papers/reig2_revised_2025.pdf)
- [🎤 Slides](slides/reig2_presentation.pdf)
- [📊 arXiv](https://arxiv.org/abs/xxxx.xxxxx) (Coming Soon)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mechanic-y/ReIG2-Research.git
cd ReIG2-Research
pip install -r requirements.txt
```

### Run Simulations

```bash
# Complete 3-qubit simulation
python code/reig2_full_simulation.py

# Non-unitary dynamics
python code/non_unitary_quantum.py

# Generate paper figures
python code/figure_generation.py
```

---

## 📊 Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| O_M(N=100) | 0.951 → 1 | Meaning observable convergence |
| L(world) | 0.012 → 0 | World distance (exponential decay) |
| Contraction κ | < 1 | Proven rigorously via Banach theorem |
| Convergence Rate | ~0.95^N | Exponential with spectral gap |

---

## 📁 Repository Structure

```
ReIG2-Research/
├── papers/
│   ├── reig2_revised_2025.pdf      # Main paper (21 pages)
│   └── reig2_original.pdf          # Original paper
├── slides/
│   └── reig2_presentation.pdf      # Beamer presentation
├── code/
│   ├── non_unitary_quantum.py      # Kraus, Lindblad implementations
│   ├── reig2_full_simulation.py    # Complete 3-qubit system
│   ├── quantum_circuit_implementation.py  # Qiskit circuits
│   └── figure_generation.py        # Reproducible visualizations
├── images/
│   ├── fig1_system_architecture.png
│   ├── fig2_convergence.png
│   ├── fig3_nonunitary.png
│   ├── fig4_circuit.png
│   └── fig5_functor.png
├── docs/
│   ├── mathematical_proofs.md      # Detailed theorem proofs
│   ├── implementation_notes.md     # Code documentation
│   └── hardware_requirements.md    # Quantum hardware specs
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 💻 Code Examples

### Example 1: Non-Unitary Evolution

```python
from non_unitary_quantum import dephasing_channel, amplitude_damping
import numpy as np

# Initial state
rho = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|

# Apply decoherence
gamma = 0.1
rho_dephased = dephasing_channel(rho, gamma)

print(f"Coherence decay: {np.abs(rho_dephased[0,1]):.4f}")
```

### Example 2: Quantum Circuit

```python
from quantum_circuit_implementation import ReIG2Circuit

# Initialize 3-qubit system
circuit = ReIG2Circuit(omega_M=1.0, omega_C=0.7, omega_O=0.5)

# Build and execute
qc = circuit.build_full_circuit(N=10, dt=0.1)
```

---

## 🔬 Theoretical Background

### Fock Space Formulation

```
ℱ = ⊕_{n=0}^∞ H_rec^⊗n = ℂ ⊕ H_rec ⊕ (H_rec ⊗ H_rec) ⊕ ...
```

Inner product: `⟨Ψ|Φ⟩_ℱ = Σ_{n=0}^∞ ⟨ψ_n|φ_n⟩`

### Fixed Point Theorem

**Theorem**: Under conditions (C1')-(C4), the system converges:

```
lim_{N→∞} T̂_Self^(N) |Ψ₀⟩ = |I⟩
```

with exponential rate `C|λ₂|^N`.

### Free Energy Principle Connection

| FEP Concept | ReIG2 Correspondence |
|-------------|---------------------|
| Internal states μ | H_M ⊗ H_C |
| Sensory input s | H_O |
| Free energy F | L(world) + λD_KL |

---

## 🖼️ Figures

<details>
<summary>Click to expand figures</summary>

### System Architecture
![System Architecture](images/fig1_system_architecture.png)

### Convergence
![Convergence](images/fig2_convergence.png)

### Quantum Circuit
![Circuit](images/fig4_circuit.png)

</details>

---

## 🛠️ Hardware Requirements

### Quantum Hardware Specifications

| Requirement | Specification |
|-------------|--------------|
| Single-qubit fidelity | F > 99.9% |
| Two-qubit fidelity | F > 99% |
| T1 (relaxation) | > 100 μs |
| T2 (coherence) | > 50 μs |
| Circuit depth | < 1000 gates |

**Recommended Platforms**:
- IBM Quantum (ibm_kyoto, ibm_osaka)
- IonQ
- Google Sycamore

---

## 📚 Citation

```bibtex
@article{wakita2025reig2,
  title={ReIG2/twinRIG: A Rigorous Quantum-Mechanical Framework for Self-Reference and World Construction},
  author={Wakita, Yasuyuki},
  journal={GitHub Pages},
  year={2025},
  note={Revised Edition},
  url={https://mechanic-y.github.io/ReIG2-Research/}
}
```

### APA Format
```
Wakita, Y. (2025). ReIG2/twinRIG: A Rigorous Quantum-Mechanical Framework for 
Self-Reference and World Construction (Revised Edition). GitHub Pages. 
https://mechanic-y.github.io/ReIG2-Research/
```

---

## 🗺️ Roadmap

- [x] Mathematical rigor enhancement (2025.11.29)
- [ ] arXiv submission (2026 Q1)
- [ ] IBM Quantum experiments (2026 Q2)
- [ ] Journal publication (2026 Q3)
- [ ] Interactive web demos (2026 Q4)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Code optimization
- Additional simulations
- Hardware implementations
- Documentation improvements
- Bug reports

---

## 📧 Contact

- **Author**: Mechanic-Y / Yasuyuki Wakita
- **GitHub**: [@mechanic-y](https://github.com/mechanic-y)
- **Email**: (Add if desired)
- **Original Site**: [ReIG2-twinRIG-Core](https://mechanic-y.github.io/ReIG2-twinRIG-Core/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Theoretical foundations: Banach, Hofstadter, Friston
- Quantum computing: Nielsen & Chuang
- Community feedback and support

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/mechanic-y/ReIG2-Research?style=social)
![GitHub forks](https://img.shields.io/github/forks/mechanic-y/ReIG2-Research?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/mechanic-y/ReIG2-Research?style=social)

---

<p align="center">
  <b>Built with ❤️ for quantum cognitive science</b>
</p>
