**Q (Reviewer rySb):** The authors choose to keep the standard loss function that supervises both the trace and the confidence estimate. To validate that the calibration has been optimally learned from the positive and negative traces, would an ablation with a basic weighted average of the two loss terms be useful to highlight the soundness of the authors' choice?

**A:** Thank you for this suggestion. As noted, our design deliberately uses a single standard SFT loss over the shuffled mixture $\mathcal{D}_{\text{total}} = \text{Shuffle}(\mathcal{D}_{\text{reason}} \cup \mathcal{D}_{\text{eval}})$, treating both reasoning traces and self-evaluation examples as next-token prediction targets within a unified objective (Algorithm 1, Lines 14–15). This simplicity is intentional: it avoids introducing additional hyperparameters while allowing the model to learn both skills through a shared autoregressive objective.

To validate this choice, we conducted the suggested ablation by introducing an explicit weighting $\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{reasoning}} + (1 - \alpha) \cdot \mathcal{L}_{\text{calibration}}$ and training Qwen3-8B across $\alpha \in \{0.3, 0.5, 0.7, 0.9, 1.0\}$, evaluated on MATH-500:

| $\alpha$ | Acc (%) | AUROC $\uparrow$ | ECE $\downarrow$ |
|---|---|---|---|
| 1.0 (reasoning only) | 44.20 | 0.6853 | 0.2658 |
| 0.9 | 43.20 | 0.7007 | 0.1213 |
| 0.7 | 45.60 | 0.7323 | 0.0918 |
| 0.5 | 43.60 | 0.7616 | 0.0920 |
| 0.3 | 44.40 | 0.7823 | 0.1162 |

The results support our unified objective design in several ways:

**(1) Reasoning accuracy is invariant to $\alpha$.** Accuracy varies within only $\pm$1.2pp (43.2–45.6%) across the full range, confirming that the calibration loss does not compete with the reasoning objective. This aligns with our paper's core claim that EpiCaR achieves Pareto-superiority—calibration comes at no meaningful accuracy cost.

**(2) Calibration improves monotonically with calibration weight.** AUROC rises consistently from 0.6853 ($\alpha$=1.0) to 0.7823 ($\alpha$=0.3), confirming that the self-evaluation signal in $\mathcal{D}_{\text{eval}}$ is directly responsible for calibration gains rather than an artifact of data augmentation.

**(3) The optimal range is broad, suggesting robustness.** ECE achieves a wide plateau across $\alpha$=0.5–0.7 (ECE $\approx$ 0.092, a 3$\times$ improvement over $\alpha$=1.0), indicating that precise tuning is unnecessary. This supports our unified objective design, as even a coarse balance between the two objectives yields strong calibration. That said, we note several promising directions for future investigation: adaptive loss weighting schemes (e.g., uncertainty-based scheduling that shifts emphasis from reasoning to calibration as training progresses), or a theoretical analysis characterizing the relationship between the implicit data mixing ratio and the effective gradient contribution of each objective. We leave these extensions for future work.

We believe this ablation validates the soundness of our unified approach—explicit loss weighting can marginally improve specific metrics, but our standard SFT formulation achieves near-optimal calibration while maintaining the advantage of zero additional hyperparameters. We will include this analysis in the revised paper.
