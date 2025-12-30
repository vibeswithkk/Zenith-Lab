The following section presents the empirical results generated from controlled experiments executed within a Jupyter Notebook environment. To ensure clarity and reproducibility, this document intentionally focuses on the outcome metrics only. Complete details regarding the experimental setup, source files, configurations, and executed code paths are available at the referenced location below.
https://colab.research.google.com/github/vibeswithkk/Zenith-Lab/blob/main/Zenith_Scientific_Stability_Test.ipynb


# Zenith Scientific Test 1: Gradient Stability

**Objective:**
Validate that Zenith's optimizations (Graph Capture, Fusion) do not cause numerical instability (Gradient Explosion/Vanishing) during extended training runs.

**Methodology:**
1.  Train a simple ResNet-like model on dummy data for 1000 steps.
2.  Use a relatively high Learning Rate (`1e-3`) to induce stress.
3.  **Monitor:** Gradient Norm (`L2 Norm`) of the model parameters at every step.
4.  **Comparison:** Zenith vs PyTorch Native.

**Success Criteria:**
*   No `NaN` or `Inf` values.
*   Gradient Norm curve of Zenith should be comparable (smoothness/magnitude) to PyTorch.

==============================================================================================
Cell output 1 : 

Requirement already satisfied: pyzenith in /usr/local/lib/python3.12/dist-packages (0.3.0)
Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.9.1)
Requirement already satisfied: torchvision in /usr/local/lib/python3.12/dist-packages (0.24.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (3.10.8)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.4.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.13.1.3)
Requirement already satisfied: triton==3.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.5.1)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.12/dist-packages (from torchvision) (11.3.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (4.61.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (1.4.9)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (25.0)
Requirement already satisfied: pyparsing>=3 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)
Device: cuda


==============================================================================================
Cell output 2 : 

--- Run 1: PyTorch Native ---
Running Native PyTorch...
Done. Duration: 3.5603s
Max Drift from Baseline: 0.000000000

--- Run 2: Zenith Optimized ---
Compiling with Zenith...
[INFO] [compiler] Compiling model for cuda

+-----------------------------------------------------------+
| Zenith Compilation Complete                               |
+-----------------------------------------------------------+
| Model:      fx_graph_module                               |
| Target:     cuda                                          |
| Precision:  fp32                                          |
| Time:       0.01s                                         |
|                                                           |
| Optimizations Applied:                                    |
|   - Fused ops: 0                                          |
|   - DCE removed: 0                                        |
|   - Est. speedup: 1.0x                                    |
+-----------------------------------------------------------+
Done. Duration: 5.0392s
Max Drift from Baseline: 0.000000000

==============================================================================================
Cell output 3 : 
![Stability Chart](../../../assets/zenith_stability_chart.png)
 
Total Time Comparison: PyTorch (3.56s) vs Zenith (5.04s)
Speedup: -41.54%

RESULT: Zenith is DETERMINISTIC. No numerical drift detected.


