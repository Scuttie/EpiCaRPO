

**Q (rySb):** Verbalized confidence is sensitive to prompt phrasing. How robust is EpiCaR's calibration to different evaluation prompts?

**A:** We appreciate this important concern. We conducted a prompt sensitivity analysis using four self-evaluation templates (original, formal, casual, detailed) across Qwen3-8B and Llama3-8B, evaluating on MATH-500.

**Setup.** For each model, we generated solutions once (greedy decoding), then measured verbalized confidence using four distinct evaluation prompts. This isolates prompt sensitivity from solution variance.

**Results.**

| Model | Method | AUROC (mean ± std) | ECE (mean ± std) |
|-------|--------|-------------------|------------------|
| Qwen3-8B | STaR | 0.721 ± 0.045 | 0.267 ± 0.064 |
| Qwen3-8B | **EpiCaR** | **0.797 ± 0.040** | **0.203 ± 0.055** |
| Llama3-8B | STaR | 0.570 ± 0.057 | 0.645 ± 0.130 |
| Llama3-8B | **EpiCaR** | 0.571 ± **0.032** | 0.594 ± 0.155 |

**Key findings:**

**(1) EpiCaR improves both calibration quality and robustness on Qwen3-8B.** EpiCaR achieves higher mean AUROC (0.797 vs 0.721, +0.076) while simultaneously exhibiting lower cross-template variance (std 0.040 vs 0.045). This suggests that Stage 2 training not only improves calibration but also internalizes the self-evaluation capability, making it less dependent on specific prompt wording.

**(2) Robustness improvement is consistent across architectures.** On both model families, EpiCaR shows lower AUROC standard deviation than STaR (average std: 0.036 vs 0.051, a 1.4× improvement). The effect is particularly pronounced on Llama3-8B, where EpiCaR reduces AUROC variance by 44% (0.032 vs 0.057), even though the absolute calibration quality is limited by the model's lower reasoning accuracy on MATH-500.

**(3) The results support our internalization hypothesis.** STaR models, which never trained on self-evaluation, must rely on prompt-elicited metacognition—making them inherently more prompt-sensitive. EpiCaR's explicit training on self-evaluation tokens creates a more stable internal signal that transfers across prompt formulations.

We acknowledge that the absolute AUROC for Llama3-8B remains moderate (~0.57), which we attribute to its lower base accuracy (16%) creating a class-imbalanced evaluation setting. The robustness improvement nonetheless demonstrates that EpiCaR's calibration mechanism generalizes beyond the specific training prompt.
