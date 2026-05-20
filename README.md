# SAE Structural Steering: Diagnosing Hierarchy in GPT-2

**Objective:** To empirically test whether Sparse Autoencoders (SAEs) can recover true hierarchical structure (specifically, bracket nesting depth) from LLM activations, and to verify this causality via feature steering. 

This repository demonstrates a complete Mechanistic Interpretability pipeline using `TransformerLens` and `SAELens` to audit a model's internal structural logic, clamping specific neural features to induce targeted structural amnesia.

## 🔬 Key Findings

1. **Feature Identification:** Isolated a distinct structural "depth feature" (**Feature ID 22574**, Layer 4 of `gpt2-small-res-jb`).
2. **Linear Scaling:** The feature's activation magnitude scales almost perfectly linearly with the depth of unclosed brackets, proving the SAE recovered a hierarchical tracking mechanism rather than a flat token representation.
3. **Causal Steering:** Artificially clamping this feature to `0.0` during the forward pass successfully degraded the model's structural logic, preventing it from closing nested sequences at shallow depths. Deeper depths survived, suggesting structural feature redundancy.

## 🛠️ Tech Stack
* **Model:** `gpt2-small`
* **Mechanistic Interpretability:** `TransformerLens`, `SAELens`
* **Framework:** PyTorch

## 📊 Empirical Results

The pipeline evaluates the model against a few-shot prompted bracket sequence. After verifying 100% baseline structural accuracy, an intervention hook suppresses Feature 22574.

```text
============================================================
STEERING EXPERIMENT: Clamping Depth Feature 22574 to 0.0
============================================================

Depth 1: 
    Expected:   ')'
    Baseline:   ' )' ✅
    Steered:    ' (' ❌

Depth 2: 
    Expected:   ')'
    Baseline:   ' )' ✅
    Steered:    ' (' ❌

Depth 3: 
    Expected:   ')'
    Baseline:   ' )' ✅
    Steered:    ' )' ✅

Depth 4: 
    Expected:   ')'
    Baseline:   ' )' ✅
    Steered:    ' )' ✅

============================================================
SUMMARY
============================================================
Baseline accuracy: 4/4 (100%)
Steered accuracy:  2/4 (50%)
Degradation:       2 failures induced

✅ SUCCESS: Steering suppressed the depth feature, breaking bracket prediction.
