# ReIG2 共鳴事象統合理論 — 実装パッケージ

**Resonance as Event-Update: A Nonlinear Operator Semigroup and Information-Geometric Integration in ReIG2**

## 概要

共鳴を「状態」ではなく「事象」として定義し、8つの非線形作用素の半群合成として形式化した理論の Python 実装。

```
𝔑 = (ρ̂ε ∘ Û) ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ
```

## パッケージ構造

```
reig2_resonance/
├── reig2/                    # メインパッケージ
│   ├── __init__.py           # エクスポート
│   ├── state.py              # §2 状態空間 (Ψ_i, Ensemble)
│   ├── operators.py          # §3 8基本作用素 (Ĉ, L̂, Ê, Â, τ̂, M̂, Û, ρ̂ε)
│   ├── semigroup.py          # §4 合成構造 (半群 𝔑)
│   ├── alignment.py          # §3.4 3軸整合 (周波数・テンソル・位相)
│   ├── info_geometry.py      # §6 情報幾何 (Fisher計量・曲率・測地線)
│   ├── ai_agent.py           # §8 AI実装 (目的関数・安全ゲート)
│   └── simulation.py         # シミュレーション・可視化
├── tests/
│   └── test_operators.py     # 単体テスト (20テスト)
├── demo.py                   # デモスクリプト
└── README.md
```

## モジュール対応表

| モジュール | 論文セクション | 内容 |
|---|---|---|
| `state.py` | §2 | 主体状態 Ψ_i = (z, W, g, θ)、集合系、秩序パラメータ |
| `operators.py` | §3 | 8作用素の完全実装 |
| `semigroup.py` | §4 | 半群合成 𝔑、時間発展、星点集合 P |
| `alignment.py` | §3.4, App.C | 3軸整合 (Align_f, Align_T, Align_φ) |
| `info_geometry.py` | §6 | Fisher計量、Christoffel接続、曲率テンソル、測地線、自由エネルギー、自然勾配 |
| `ai_agent.py` | §8, §9 | 目的関数 J(θ)、リスク項、安全ゲート、二段階更新 |
| `simulation.py` | — | シミュレーション実行、4パネル可視化 |

## 作用素一覧

| 記号 | クラス名 | 役割 |
|---|---|---|
| Ĉ | `ContactOperator` | 接触・干渉生成 |
| L̂ | `CooperationLayerOperator` | 協力層（場）生成 |
| Ê | `EnvironmentShareOperator` | 環境共有 |
| Â | `AlignmentOperator` | 3軸整合（Kuramoto同期） |
| τ̂ | `ThresholdGate` | 臨界判定（相転移スイッチ） |
| M̂ | `EmpathyOperator` | 共感（視点交換、τ̂=1のみ） |
| Û | `UpdateOperator` | 更新（塑性） |
| ρ̂ε | `RelaxationOperator` | 可逆制御（緩和/固定） |

## 使用方法

```python
from reig2 import Ensemble, ResonanceSemigroup

# 25体の集合系を初期化
ensemble = Ensemble.random(N=25, dim_z=4, dim_theta=3)

# 半群を構成
semigroup = ResonanceSemigroup(R_c=0.65, coupling_strength=0.18)

# 時間発展 (200ステップ)
final, history = semigroup.run(ensemble, T_steps=200)

# 星点集合
print(semigroup.star_points)  # P = {t_k}
```

## テスト

```bash
cd reig2_resonance
python -m tests.test_operators
```

## デモ

```bash
python demo.py
```

## 依存パッケージ

- `numpy` (必須)
- `matplotlib` (可視化、オプション)

## 著者

Yasuyuki Wakita（脇田泰行）

## ライセンス

MIT License

## 引用 (Citation)

```bibtex
@software{wakita2026reig2resonance,
  author       = {Wakita, Yasuyuki},
  title        = {ReIG2 Resonance Operator Theory},
  version      = {1.0.0},
  year         = {2026},
  license      = {MIT},
  url          = {https://github.com/Mechanic-Y/reig2-resonance}
}
```

詳細は `CITATION.cff` を参照してください。
