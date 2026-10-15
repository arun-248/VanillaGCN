# 🕸️ Three-Layer Vanilla Graph Convolutional Network (GCN)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/From%20Scratch-NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)](https://numpy.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Cora-7C5CFC?style=for-the-badge)](https://github.com/kimiyoung/planetoid)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white&style=for-the-badge)](https://github.com/features/actions)

**🎓 A derivation and from-scratch implementation of a GCN with exactly three graph-convolution layers, evaluated on the Cora citation network for semi-supervised node classification**

**Derive · Implement · Verify · Evaluate**

<a href="https://github.com/arun-248/VanillaGCN"><strong>📂 Source Code</strong></a>
·
<a href="docs/GCN_Two_and_Three_Layer_Assignment.pdf"><strong>📄 Assignment Document</strong></a>

</div>

---

## 🌟 Project Highlights

> **Every number in this README is copied from files in `outputs/`, produced by running the code.** The fixed-weight numerical demonstration is kept clearly separate from the trainable model.

**🎯 What makes this special:**
- **Three graph-convolution layers**, with every mathematical step implemented explicitly in NumPy
- **Hand-derived back-propagation** verified by finite-difference gradient checks (worst relative error 2.3e-08)
- **Adam optimiser, Xavier initialisation, dropout, L2 and checkpoint selection** all written from scratch
- **Reproducible protocol**: 10 seeds, public Planetoid split, GCN compared against a no-graph baseline
- **Scientific honesty built in**: limitations and the unexecuted PyTorch version are stated openly
- **Full documentation**: three PDF documents covering derivation, code and outputs

> ⚠️ **Educational / assignment project.** Results come from one dataset and one fixed split, with no systematic hyper-parameter tuning.

---

## 🚀 Key Features

**📐 Mathematical Derivation**
* Spectral filter → Chebyshev K = 1 → renormalisation trick
* Pseudocode and a numerical worked example (PDF, Section 6)

**🧮 From-Scratch Implementation**
* One function per formula: self-loops, degrees, Â (dense and sparse), ReLU, softmax, layer, loss
* Manual back-propagation for both the two-layer and three-layer GCN
* Macro precision / recall / F1 metrics

**✅ Verification**
* Finite-difference gradient checks for the two-layer and three-layer networks
* Fixed-weight demo with no training, printing every intermediate matrix

**📊 Experiments**
* Cora node classification over 10 seeds
* No-graph baseline (Â = I) for comparison
* Two-layer example on Zachary's Karate Club (2 labelled nodes)

---

## 🏗️ System Architecture

```
 Cora (Planetoid files)
 ┌──────────────────────────────┐
 │ 2,708 nodes · 1,433 features │
 │ 7 classes · 140/500/1000     │
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ Ã = A + I                    │
 │ Â = D̃^(-1/2) Ã D̃^(-1/2)      │
 └──────────────┬───────────────┘
                ▼
 X ─▶ Layer 1 (1433→64) ─ReLU─▶ Layer 2 (64→32) ─ReLU─▶ Layer 3 (32→7) ─softmax─▶ P
                                                                                  │
                           Cross-entropy on labelled nodes ◀──────────────────────┘
                                         │
                                         ▼
              Manual back-propagation → Adam → Best-validation checkpoint
```

---

## 📊 How the System Works

**1. 📥 Data Loading**
* The Cora loader reads Planetoid files into a sparse adjacency matrix, features, labels and masks

**2. 🔗 Graph Normalisation**
* Self-loops are added and the adjacency matrix is symmetrically normalised

**3. ➡️ Forward Pass**
* Three graph-convolution layers with ReLU between them and softmax at the output

**4. ⬅️ Back-Propagation**
* Gradients are derived by hand and checked against finite differences

**5. 🏋️ Training**
* Adam with dropout 0.5, learning rate 0.01, weight decay 5e-4 (layer 1) and 200 epochs

**6. 💾 Checkpoint Selection**
* The checkpoint with the lowest validation cross-entropy is used for testing

---

## 🧮 Mathematical Background

| Quantity | Formula |
|---|---|
| Self-loops | Ã = A + I |
| Degrees | D̃ᵢᵢ = Σⱼ Ãᵢⱼ |
| Symmetric normalisation | Â = D̃^(-1/2) Ã D̃^(-1/2) |
| Layer 1 | H¹ = ReLU(Â X W¹ + b¹) |
| Layer 2 | H² = ReLU(Â H¹ W² + b²) |
| Layer 3 | Z = Â H² W³ + b³, P = softmax(Z) |
| Loss | Cross-entropy over the labelled training nodes |
| Output gradient | δ³ = M ⊙ (P − Y) / \|train\| |

The full derivation, including the remaining gradients, is in the PDF documents.

---

## 🗂️ Dataset

### Cora (Planetoid format)

| Property | Value |
|---|---|
| Location | `data/cora/` (from [kimiyoung/planetoid](https://github.com/kimiyoung/planetoid)) |
| Papers (nodes) | 2,708 |
| Unique undirected citations | 5,278 |
| Word features | 1,433 (row-normalised) |
| Classes | 7 |
| Public split | 140 train / 500 validation / 1,000 test |
| Edge homophily | 0.810 |

Two small graphs support the demos: a **4-node toy graph** for the fixed-weight demonstration and a **60-node synthetic block-model graph** for the gradient check.

---

## 📈 Results

> From `outputs/cora_output.txt`; 1,000 test nodes; mean ± sample standard deviation over 10 seeds.

| Metric | No graph (Â = I) | **3-layer GCN** |
|---|---:|---:|
| Accuracy | 0.5459 ± 0.0197 | **0.8160 ± 0.0082** |
| Macro precision | 0.5264 ± 0.0202 | **0.7991 ± 0.0071** |
| Macro recall | 0.5538 ± 0.0140 | **0.8129 ± 0.0109** |
| Macro F1 | 0.5303 ± 0.0193 | **0.8039 ± 0.0077** |
| Test loss | 1.3453 ± 0.0479 | **0.6032 ± 0.0180** |

### ✅ Gradient checks (finite differences)

| Network | Worst relative error | Status |
|---|---:|---|
| Three-layer | 2.3e-08 | PASSED |
| Two-layer | 2.4e-09 | PASSED |

### 🥋 Two-layer example (Zachary's Karate Club)

* 2 labelled nodes, seed 42, final epoch: accuracy on the 32 unlabelled nodes is **0.8125**
* Over seeds 0–9: mean **0.9406 ± 0.0453** (see `outputs/two_layer_example_output.txt`)
* This is a tiny graph and is **not comparable** with the Cora numbers

Per-seed results, confusion matrices and learning curves are in the PDF.

---

## ⚙️ Experimental Settings

| Setting | Value |
|---|---|
| Layer widths | 1433 → 64 → 32 → 7 |
| Dropout | 0.5 |
| Optimiser | Adam, learning rate 0.01 |
| Weight decay | 5e-4 (layer 1) |
| Epochs | 200 |
| Checkpoint | Lowest validation cross-entropy |
| Seeds | 42, 123, 456, 7, 8, 9, 10, 11, 12, 13 |

Hyper-parameters were **not** tuned systematically.

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt

bash run_all.sh                        # demo, gradient check, Cora experiment (~12 min on 1 CPU core), figures, PDF
python src/run_cora.py                 # only the main experiment
python src/run_two_layer_example.py    # two-layer examples (Karate Club)
```

---

## 📁 Project Structure

```
VanillaGCN/
│
├── .github/workflows/          # GitHub Actions workflows
├── data/cora/                  # Cora dataset (Planetoid format)
├── docs/                       # PDF documents (assignment, report, two-layer chapter)
├── outputs/                    # Saved outputs that every reported number comes from
├── src/                        # Source code (see table below)
├── tests/                      # Gradient checks and PyTorch equivalence test
├── .gitignore
├── requirements.txt
├── run_all.sh                  # Runs demo, gradient check, experiment, figures, PDF
└── README.md                   # This file
```

### 💻 Source files

| File | Content |
|---|---|
| `src/cora_data.py` | Cora loader (Planetoid files → sparse A, features, labels, masks) |
| `src/graph_data.py` | Toy graph and synthetic 3-community graph |
| `src/gcn_ops.py` | One function per formula: self-loops, degrees, Â (dense and sparse), ReLU, softmax, layer, loss |
| `src/gcn_two_layer.py` | Two-layer GCN in NumPy with hand-derived back-propagation, Adam and checkpoint rules |
| `src/run_two_layer_example.py` | Fixed-weight forward pass and a trained run on Zachary's Karate Club |
| `src/gcn_numpy.py` | Trainable three-layer GCN: manual back-propagation, Adam, Xavier init, dropout, L2, checkpoint selection, macro metrics |
| `src/demo_fixed_weights.py` | Fixed-weight demo with no training; prints every intermediate matrix |
| `src/run_cora.py` | Main experiment: 10 seeds, GCN vs. no-graph baseline |
| `src/run_training.py` | Supplementary run on the small synthetic graph |
| `src/gcn_torch.py` | PyTorch version (not executed, see note) |
| `src/make_figures.py`, `src/make_cora_figures.py`, `src/make_two_layer_figures.py` | Figures |
| `src/build_report.py`, `src/build_two_layer_doc.py`, `src/build_assignment_doc.py` | The three PDF documents |
| `tests/gradient_check.py` | Finite-difference test of three-layer back-propagation |
| `tests/gradient_check_two_layer.py` | Finite-difference test of two-layer back-propagation |
| `tests/test_torch_matches_numpy.py` | PyTorch equivalence test (not executed) |

> **PyTorch note.** The environment used to prepare this package could not install PyTorch (no disk space), so `gcn_torch.py` and `tests/test_torch_matches_numpy.py` were written but never run. **No result here comes from them.** Please run the test locally first.

---

## 📚 Documentation

| Document | Pages | Content |
|---|---:|---|
| [`GCN_Two_and_Three_Layer_Assignment.pdf`](docs/GCN_Two_and_Three_Layer_Assignment.pdf) | 27 | Assignment document: two-layer + three-layer |
| [`GCN_Assignment_Report.pdf`](docs/GCN_Assignment_Report.pdf) | 16 | Three-layer report (Phases 1–3) |
| [`Two_Layer_GCN_and_Codes.pdf`](docs/Two_Layer_GCN_and_Codes.pdf) | 25 | Two-layer chapter with all codes and outputs |

---

## ⚖️ Limitations

1. **One dataset and one fixed split.** Seeds vary only initialisation and dropout, so the standard deviations are not confidence intervals for other splits or datasets.
2. **Strong homophily.** Cora favours neighbourhood averaging.
3. **No systematic hyper-parameter search.**
4. **No comparison** with other architectures or depths (depth is fixed at three by the assignment).
5. **t-SNE is illustrative only.** No cluster-quality metric was computed.
6. **PyTorch implementation not executed** by the package author.

---

## 🔄 Future Enhancements

<details>
<summary><strong>🎯 Near-term</strong></summary>

- [ ] Run and verify the PyTorch equivalence test locally
- [ ] Add a systematic hyper-parameter search

</details>

<details>
<summary><strong>🌟 Research roadmap</strong></summary>

- [ ] Evaluate on additional datasets and random splits
- [ ] Compare against other architectures and depths
- [ ] Add a cluster-quality metric for the t-SNE visualisation

</details>

---

## 📖 References

* Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017
* Yang, Cohen & Salakhutdinov, *Revisiting Semi-Supervised Learning with Graph Embeddings*, ICML 2016

---

<div align="center">

## 🎓 Understanding graph learning from first principles

**Every formula implemented, every gradient verified, every number traceable to an output file.**

---

**Made with ❤️ by [Arun Chinthalapally](https://github.com/arun-248)**

[⬆ Back to Top](#️-three-layer-vanilla-graph-convolutional-network-gcn)

</div>
