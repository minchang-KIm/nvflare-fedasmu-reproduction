# Verification of appendix_lenet5_cifar10_fedbuff_s0

Run status: COMPLETED; 5000 logged updates. Result: **ALL CHECKS PASSED** (2/2).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| B1 | NVFLARE BuffModelManager makes a new global model after every K = 10 updates | FedBuff Alg. 1 | 5000 |  |  | PASS | 500 new global models from 5000 updates |
| B2 | model version increases by exactly 1 per aggregation | FedBuff Alg. 1 | 5000 |  |  | PASS |  |

## Example log rows

```
## FedBuff, first aggregation
 received   vtime  device algorithm   t  o  staleness  accepted  lam_before  sig_before  iota_before  prev_t  prev_tau  prev_xi  g_dot_prev  grad_lam  grad_sig  grad_iota  lam  sig  iota  xi  alpha  eq1_ratio  eq1_resid  slot_src  request_after  fresh_version  merged  gamma_before  upsilon_before  phi  beta  grad_dot  d_gamma  d_upsilon  gamma  upsilon  eq7_ratio  loss_before  loss_after  reward   b  q_prev_state  q_action  q_next_state  q_greedy  q_h_before  q_max_next  q_h_after  meta_feat  meta_probs  meta_decisions  meta_slot  meta_reward  meta_baseline_before  meta_adv  meta_lp_before  meta_lp_after  fb_generated  fb_model_version
       10 180.644      38   fedbuff NaN  1          0      True         NaN         NaN          NaN     NaN       NaN      NaN         NaN       NaN       NaN        NaN  NaN  NaN   NaN NaN    NaN        NaN        NaN       NaN            NaN            NaN   False           NaN             NaN  NaN   NaN       NaN      NaN        NaN    NaN      NaN        NaN          NaN         NaN     NaN NaN           NaN       NaN           NaN       NaN         NaN         NaN        NaN        NaN         NaN             NaN        NaN          NaN                   NaN       NaN             NaN            NaN          True                 2

```
