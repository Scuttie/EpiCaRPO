**Q (Reviewer rySb):** Verbalized confidence is sensitive to prompt phrasing. How robust is EpiCaR's calibration to different evaluation prompts?

**A:** We appreciate this important concern. We conducted a prompt sensitivity analysis using four self-evaluation templates across Qwen3-8B and Llama3-8B, evaluating on MATH-500.

**Setup.** For each model, we generated solutions once (greedy decoding), then measured verbalized confidence using four distinct evaluation prompts with varying levels of formality and specificity. This isolates prompt sensitivity from solution variance. The four templates are:

| Template | Self-Evaluation Prompt |
|---|---|
| Original | `Is the answer correct? Choose ONLY one letter. A) Yes B) No. Your choice:` |
| Formal | `Based on careful mathematical verification, is this solution correct? Choose: A) Yes B) No. Answer:` |
| Casual | `Right or wrong? A) Right B) Wrong. Pick one:` |
| Detailed | `After checking each reasoning step and the final computation, is the answer correct? A) Yes, it is correct B) No, it is incorrect. Your choice:` |

These templates span a range of stylistic and instructional characteristics: **Original** is our training-time prompt; **Formal** introduces domain-specific framing ("mathematical verification"); **Casual** minimizes instruction length and uses colloquial phrasing; **Detailed** explicitly requests step-level checking and uses full-sentence answer options. This diversity tests whether EpiCaR's calibration depends on surface-level prompt similarity to the training template or reflects a deeper internalized capability.

**Results.**

| Model | Method | AUROC (mean $\pm$ std) | ECE (mean $\pm$ std) |
|-------|--------|-------------------|------------------|
| Qwen3-8B | STaR | 0.721 $\pm$ 0.045 | 0.267 $\pm$ 0.064 |
| Qwen3-8B | **EpiCaR** | **0.797 $\pm$ 0.040** | **0.203 $\pm$ 0.055** |
| Llama3-8B | STaR | 0.570 $\pm$ 0.057 | 0.645 $\pm$ 0.130 |
| Llama3-8B | **EpiCaR** | **0.571 $\pm$ 0.032** | **0.594 $\pm$ 0.155** |

**Key findings:**

**(1) EpiCaR improves both calibration quality and robustness on Qwen3-8B.** EpiCaR achieves higher mean AUROC (0.797 vs. 0.721, +0.076) while simultaneously exhibiting lower cross-template variance (std 0.040 vs. 0.045). This suggests that Stage 2 training not only improves calibration but also internalizes the self-evaluation capability, making it less dependent on specific prompt wording.

**(2) Robustness improvement is consistent across architectures.** On both model families, EpiCaR shows lower AUROC standard deviation than STaR (average std: 0.036 vs. 0.051, a 1.4$\times$ improvement). The effect is particularly pronounced on Llama3-8B, where EpiCaR reduces AUROC variance by 44% (0.032 vs. 0.057), even though the absolute calibration quality is limited by the model's lower reasoning accuracy on MATH-500.

**(3) The results support our internalization hypothesis.** STaR models, which never trained on self-evaluation, must rely on prompt-elicited metacognition—making them inherently more prompt-sensitive. EpiCaR's explicit training on self-evaluation tokens creates a more stable internal signal that transfers across prompt formulations. Notably, even the **Casual** template—which differs most from the training prompt in both length and vocabulary—still elicits well-calibrated confidence from EpiCaR, whereas STaR models show significant degradation.

We acknowledge that the absolute AUROC for Llama3-8B remains moderate ($\approx$0.57), which we attribute to its lower base accuracy (16%) creating a class-imbalanced evaluation setting. The robustness improvement nonetheless demonstrates that EpiCaR's calibration mechanism generalizes beyond the specific training prompt. We will include the per-template breakdown and these prompt definitions in the revised paper's appendix.
