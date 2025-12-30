The following section presents the empirical results generated from controlled experiments executed within a Jupyter Notebook environment. To ensure clarity and reproducibility, this document intentionally focuses on the outcome metrics only. Complete details regarding the experimental setup, source files, configurations, and executed code paths are available at the referenced location below.
https://colab.research.google.com/github/vibeswithkk/Zenith-Lab/blob/main/Zenith_Scientific_DynamicShape_Test.ipynb

# Zenith Scientific Test 3: The Dynamic Shape Torture Test

**Objective:**
Real-world AI workloads (like Chatbots) have variable input lengths. Static compilers often fail here, triggering slow "re-compilation" for every new shape.
This test verifies if Zenith handles **Dynamic Shapes** gracefully.

**Methodology:**
1.  Define a Linear Layer model.
2.  Run an inference loop where `batch_size` is fixed (32), but `seq_len` changes randomly (between 100 and 1000) **every single iteration**.
3.  **Metric:** Measure latency of each step. Watch for "Spikes".

**Success Criteria:**
*   **No compiling pauses:** The first few steps might be slow (warmup), but subsequent steps must be fast regardless of shape changes.
*   **Linear Scaling:** Latency should increase linearly with sequence length, not exponentially.
==============================================================================================

Cell output 1 : 

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.8/52.8 kB 4.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 477.0/477.0 kB 21.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 899.7/899.7 MB 759.7 kB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 594.3/594.3 MB 1.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.2/10.2 MB 146.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.0/88.0 MB 9.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 954.8/954.8 kB 61.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 193.1/193.1 MB 5.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 71.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.6/63.6 MB 12.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 267.5/267.5 MB 4.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288.2/288.2 MB 4.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.3/39.3 MB 17.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 90.0/90.0 kB 5.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.5/170.5 MB 6.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 131.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 147.8 MB/s eta 0:00:00

=============================================================================================

Cell output 2 : 

[INFO] [compiler] Compiling model for cuda

+-----------------------------------------------------------+
| Zenith Compilation Complete                               |
+-----------------------------------------------------------+
| Model:      fx_graph_module                               |
| Target:     cuda                                          |
| Precision:  fp32                                          |
| Time:       0.00s                                         |
|                                                           |
| Optimizations Applied:                                    |
|   - Fused ops: 0                                          |
|   - DCE removed: 0                                        |
|   - Est. speedup: 1.0x                                    |
+-----------------------------------------------------------+

--- Zenith Optimized ---
Warmup Done.
Starting Dynamic Loop...
[INFO] [compiler] Compiling model for cuda

+-----------------------------------------------------------+
| Zenith Compilation Complete                               |
+-----------------------------------------------------------+
| Model:      fx_graph_module                               |
| Target:     cuda                                          |
| Precision:  fp32                                          |
| Time:       0.00s                                         |
|                                                           |
| Optimizations Applied:                                    |
|   - Fused ops: 0                                          |
|   - DCE removed: 0                                        |
|   - Est. speedup: 1.0x                                    |
+-----------------------------------------------------------+
Step 0: SeqLen=422 -> 670.26ms
Step 10: SeqLen=231 -> 113.10ms
Step 20: SeqLen=566 -> 270.38ms
Step 30: SeqLen=932 -> 438.70ms
Step 40: SeqLen=143 -> 67.60ms

==============================================================================================

Cell output 3 : 


![Dynamic Shape Chart](../../../assets/zenith_dynamic_shape_chart.png)


