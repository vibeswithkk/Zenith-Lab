<p align="center">
  <img src="assets/zenith_v0.3.0_benchmark.png" alt="Zenith Benchmark" width="600"/>
</p>

<h1 align="center">Zenith Performance Suite</h1>

<p align="center">
  <strong>High-Performance Inference Compiler for PyTorch</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyzenith/">
    <img src="https://img.shields.io/pypi/v/pyzenith?color=blue&label=PyPI" alt="PyPI Version"/>
  </a>
  <a href="https://github.com/vibeswithkk/zenith-performance-suite/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  </a>
  <a href="https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Production_RealWorld_Tests/Zenith_v0.3.0_RealWorld_Benchmark.ipynb">
    <img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab" alt="Open in Colab"/>
  </a>
</p>

<p align="center">
  <code>+69% Inference Speedup</code> | <code>-87% Energy Consumption</code> | <code>0.000 MSE</code>
</p>

---

## What is Zenith?

**Zenith** is a high-performance inference compiler that integrates seamlessly with PyTorch via `torch.compile()`. It optimizes your models for faster inference while maintaining perfect numerical precision.

```python
import torch
import zenith  # Registers the backend automatically

model = YourModel()
optimized_model = torch.compile(model, backend="zenith")

# 1.7x faster inference with identical outputs
output = optimized_model(input_tensor)
```

---

## Quick Start

### Installation

```bash
pip install pyzenith
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

---

## Benchmark Results (v0.3.0)

All benchmarks were conducted on **Google Colab** with a **Tesla T4 GPU** (15GB VRAM).

### Hero Metrics

| Metric | PyTorch | Zenith | Improvement |
| :--- | :--- | :--- | :--- |
| **Inference Speed** | 15.37 tok/s | 26.02 tok/s | **+69.21%** |
| **Energy Consumption** | 49.34 J | 6.09 J | **-87.66%** |
| **Numerical Precision** | Baseline | 0.000000 MSE | **PERFECT** |

---

## Scientific Validation Suite

Zenith has been rigorously tested across multiple dimensions to ensure production readiness:

### 1. Inference Performance

Text generation throughput comparison using TinyLlama-1.1B.

| Metric | PyTorch | Zenith | Delta |
| :--- | :--- | :--- | :--- |
| **Throughput (TPS)** | 15.37 | 26.02 | **+69.21%** |

[View Full Log](assets/HASIL_COLAB/Production_RealWorld_Tests/Result_Zenith_v0.3.0_RealWorld_Benchmark.md)

---

### 2. Numerical Precision

Bitwise accuracy verification across critical operations.

| Operation | Max Difference | Status |
| :--- | :--- | :--- |
| Softmax | 0.000000000 | PASSED |
| LayerNorm | 0.000000000 | PASSED |
| GELU | 0.000000000 | PASSED |

[View Precision Log](assets/HASIL_COLAB/Production_RealWorld_Tests/Result_Zenith_Scientific_Precision_Test.md)

---

### 3. Dynamic Shape Adaptability

Variable-length input handling without recompilation overhead.

<p align="center">
  <img src="assets/zenith_dynamic_shape_chart.png" alt="Dynamic Shape Performance" width="500"/>
</p>

| Metric | Result |
| :--- | :--- |
| Recompilation Events | None Detected |
| Latency Scaling | Linear |

[View Dynamic Shape Log](assets/HASIL_COLAB/Production_RealWorld_Tests/Result_Zenith_Scientific_DynamicShape_Test.md)

---

### 4. Energy Efficiency

Power consumption measurement for sustainable AI.

<p align="center">
  <img src="assets/zenith_energy_chart.png" alt="Energy Efficiency" width="500"/>
</p>

| Metric | PyTorch | Zenith | Savings |
| :--- | :--- | :--- | :--- |
| Duration | 1.85s | 0.12s | 15x Faster |
| Energy | 49.34 J | 6.09 J | **-87.66%** |

[View Energy Log](assets/HASIL_COLAB/Production_RealWorld_Tests/Result_Zenith_Scientific_Energy_Test.md)

---

### 5. Numerical Stability

Determinism verification over 500 inference iterations.

<p align="center">
  <img src="assets/zenith_stability_chart.png" alt="Stability Chart" width="500"/>
</p>

| Metric | PyTorch | Zenith |
| :--- | :--- | :--- |
| Max Drift | 0.000000000 | 0.000000000 |
| Determinism | 100% | 100% |

[View Stability Log](assets/HASIL_COLAB/Production_RealWorld_Tests/Result_Zenith_Scientific_Stability_Test.md)

---

## Final Verdict

| Aspect | Status | Notes |
| :--- | :--- | :--- |
| Inference Speed | **EXCEPTIONAL** | +69% over PyTorch |
| Numerical Precision | **PERFECT** | Zero divergence |
| Dynamic Shapes | **GOOD** | No recompile overhead |
| Energy Efficiency | **EXCEPTIONAL** | -87% consumption |
| Stability | **PERFECT** | 100% deterministic |

---

## Repository Structure

```
zenith-performance-suite/
    Production_RealWorld_Tests/     # Real benchmark notebooks
    Legacy_Dummy_Tests/             # Historical test files
    assets/
        HASIL_COLAB/
            Production_RealWorld_Tests/   # Detailed result logs
            Legacy_Dummy_Tests/           # Historical logs
        *.png                             # Benchmark charts
    Zenith_Scientific_*.ipynb       # Scientific validation notebooks
    README.md
    LICENSE
```

---

## Run Your Own Benchmarks

Open any notebook directly in Google Colab:

| Test | Colab Link |
| :--- | :--- |
| Real-World Benchmark | [Open](https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Production_RealWorld_Tests/Zenith_v0.3.0_RealWorld_Benchmark.ipynb) |
| Precision Test | [Open](https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Zenith_Scientific_Precision_Test.ipynb) |
| Dynamic Shape Test | [Open](https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Zenith_Scientific_DynamicShape_Test.ipynb) |
| Energy Test | [Open](https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Zenith_Scientific_Energy_Test.ipynb) |
| Stability Test | [Open](https://colab.research.google.com/github/vibeswithkk/zenith-performance-suite/blob/main/Zenith_Scientific_Stability_Test.ipynb) |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

---

## License

MIT License - Copyright (c) 2025 Wahyu Ardiansyah

---

<p align="center">
  <sub>Built with precision. Validated with science. Ready for production.</sub>
</p>
