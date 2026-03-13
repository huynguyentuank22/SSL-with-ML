

















<!-- RQ1
Which encoder architecture produces the most informative embeddings for HPC workload representation?

RQ2
Which encoder layer provides the most predictive representation for downstream ML models?

RQ3
Do learned embeddings outperform raw tabular features for resource prediction?

RQ4
Does contrastive learning improve representation quality compared to purely supervised training?

RQ5
Do contrastive representations improve generalization across unseen workloads?

---

## Triplet Encoder Results

Metrics evaluated on the validation set after training with triplet loss.
- **alignment_after** ↓ lower is better (anchor–positive pairs are closer)
- **uniformity_after** ↓ lower (more negative) is better (embeddings spread more uniformly)
- **cosine_gap_after** ↑ higher is better (larger margin between positive and negative pairs)

Model parameter counts (INPUT_DIM=121, EMB_DIM=64):

| Architecture | # Params |
|---|---:|
| mlp_small | 11,968 |
| mlp_medium | 40,384 |
| mlp_large | 72,384 |
| mlp_deep | 138,176 |
| tabtransformer_small | 23,136 |
| tabtransformer_medium | 112,448 |
| tabtransformer_large | 211,968 |

---

### Task: `avgpcon`

| Architecture | # Params | best_val_loss | cosine_gap_after | alignment_after | uniformity_after |
|---|---:|---:|---:|---:|---:|
| mlp_small | 11,968 | 0.014422 | 1.538659 | 0.338951 | -1.882580 |
| mlp_medium | 40,384 | 0.009218 | 1.569577 | 0.311508 | -1.969187 |
| mlp_large | 72,384 | 0.008221 | 1.594709 | 0.296217 | -2.014527 |
| mlp_deep | 138,176 | 0.007736 | 1.580352 | 0.297623 | -2.047857 |
| tabtransformer_small | 23,136 | 0.017235 | 1.532222 | 0.325203 | -1.886634 |
| tabtransformer_medium | 112,448 | 0.012807 | 1.563049 | 0.308876 | -1.940159 |
| tabtransformer_large | 211,968 | 0.011275 | 1.566984 | 0.314306 | -1.953741 |

---

### Task: `duration`

| Architecture | # Params | best_val_loss | cosine_gap_after | alignment_after | uniformity_after |
|---|---:|---:|---:|---:|---:|
| mlp_small | 11,968 | 0.033749 | 1.425468 | 0.465652 | -1.870254 |
| mlp_medium | 40,384 | 0.024032 | 1.479785 | 0.416667 | -2.008016 |
| mlp_large | 72,384 | 0.022057 | 1.487571 | 0.418104 | -2.042227 |
| mlp_deep | 138,176 | 0.020319 | 1.496454 | 0.409658 | -2.093574 |
| tabtransformer_small | 23,136 | 0.034080 | 1.420850 | 0.461217 | -1.894813 |
| tabtransformer_medium | 112,448 | 0.026846 | 1.452114 | 0.435369 | -2.016589 |
| tabtransformer_large | 211,968 | 0.024884 | 1.470371 | 0.422383 | -2.004866 |

---

### Task: `maxpcon`

| Architecture | # Params | best_val_loss | cosine_gap_after | alignment_after | uniformity_after |
|---|---:|---:|---:|---:|---:|
| mlp_small | 11,968 | 0.013297 | 1.525265 | 0.356180 | -1.888953 |
| mlp_medium | 40,384 | 0.008124 | 1.548508 | 0.324975 | -1.988587 |
| mlp_large | 72,384 | 0.007448 | 1.551491 | 0.326161 | -1.974344 |
| mlp_deep | 138,176 | 0.006677 | 1.552575 | 0.312977 | -2.005322 |
| tabtransformer_small | 23,136 | 0.015268 | 1.516210 | 0.351580 | -1.913601 |
| tabtransformer_medium | 112,448 | 0.011876 | 1.522996 | 0.347682 | -1.971135 |
| tabtransformer_large | 211,968 | 0.010230 | 1.537267 | 0.333844 | -1.973744 |

---

### Task: `minpcon`

| Architecture | # Params | best_val_loss | cosine_gap_after | alignment_after | uniformity_after |
|---|---:|---:|---:|---:|---:|
| mlp_small | 11,968 | 0.022386 | 1.440404 | 0.329968 | -1.518020 |
| mlp_medium | 40,384 | 0.017703 | 1.454692 | 0.313179 | -1.543735 |
| mlp_large | 72,384 | 0.017240 | 1.457263 | 0.316080 | -1.571286 |
| mlp_deep | 138,176 | 0.016750 | 1.464352 | 0.314049 | -1.575527 |
| tabtransformer_small | 23,136 | 0.024653 | 1.452904 | 0.336118 | -1.557273 |
| tabtransformer_medium | 112,448 | 0.020526 | 1.414047 | 0.293990 | -1.509936 |
| tabtransformer_large | 211,968 | 0.021136 | 1.423097 | 0.313135 | -1.547493 |

-> mlp deep best all task
 -->
