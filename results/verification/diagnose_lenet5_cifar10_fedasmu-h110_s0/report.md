# Verification of diagnose_lenet5_cifar10_fedasmu-h110_s0

Run status: COMPLETED; 5000 logged updates. Result: **ALL CHECKS PASSED** (28/28).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| S0 | at most m' = 10 devices train at the same time (triggered every T) | Algorithm 1 l.1-6 | 102 |  |  | PASS | devices in training at evaluation points: max 9, mean 8.9 |
| S1 | staleness tau = t - o + 1 | Algorithm 1 l.9 | 5000 | 0.0e+00 | 0e+00 | PASS |  |
| S2 | update discarded iff tau > 99 | Algorithm 1 l.9-10 | 5000 |  |  | PASS | discarded: 0 |
| S3 | xi = lam/(sqrt(t)(t-o+1)^sig) + iota | Eq. 2 | 5000 | 0.0e+00 | 1e-09 | PASS |  |
| S4 | alpha = mu*xi/(1+mu*xi) | Eq. 2 | 5000 | 0.0e+00 | 1e-09 | PASS |  |
| S5 | new global model equals (1-alpha) w_t + alpha w_i element by element (relative residual) | Eq. 1 | 5000 | 0.0e+00 | 1e-06 | PASS |  |
| S5b | mixing ratio measured on the tensors \|w_{t+1}-w_t\|/\|w_i-w_t\| = alpha, within float32 rounding bound 2^-24 \|w_{t+1}\|/\|w_i-w_t\| | Eq. 1 | 5000 |  |  | PASS | max \|ratio-alpha\| 8.3e-08, max bound 1.3e-05 |
| S6 | dF/dlam, dF/dsig, dF/diota (chain rule through alpha) | Eq. 3 + appendix A | 4900 | 6.9e-18 | 1e-09 | PASS |  |
| S7 | lam/sig/iota <- x - eta_x * dF/dx | Eq. 3 | 4900 | 0.0e+00 | 1e-09 | PASS | first upload of a device has no previous update: 100 rows, parameters unchanged (OK) |
| S8 | per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0 | Algorithm 1 (state per device) | 5000 | 0.0e+00 | 1e-12 | PASS |  |
| D1 | request slot l* in 1..L-1 (L = 5) | Algorithm 2 l.4 | 5000 |  |  | PASS |  |
| D2 | fresh model merged only if it is newer (g > o) | Algorithm 2 l.6-7 | 4125 |  |  | PASS | requests 5000, merged 4125, no newer model 875 |
| D3 | phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)) | Formula 8 | 4125 | 0.0e+00 | 1e-09 | PASS |  |
| D4 | beta = mu*phi/(1+mu*phi) | Formula 8 | 4125 | 0.0e+00 | 1e-09 | PASS |  |
| D5 | local model after merge equals (1-beta) w_local + beta w_g element by element (relative residual) | Formula 7 | 4125 | 0.0e+00 | 1e-06 | PASS |  |
| D5b | mixing ratio measured on the tensors = beta, within float32 rounding bound | Formula 7 | 4125 |  |  | PASS | max \|ratio-beta\| 7.6e-08, max bound 2.1e-06 |
| D6 | dF/dgamma, dF/dupsilon | Formula 9 + appendix A | 4125 | 0.0e+00 | 1e-09 | PASS |  |
| D7 | gamma/upsilon <- x - eta_x * dF/dx | Formula 9 | 4125 | 0.0e+00 | 1e-09 | PASS |  |
| D8 | per-device gamma: value before a merge == value after its previous merge; start at gamma0 | Algorithm 2 (device state) | 4125 | 0.0e+00 | 1e-12 | PASS |  |
| D9 | reward R = loss(before merge) - loss(after merge), same mini-batch | Algorithm 2 l.11 | 4125 | 0.0e+00 | 1e-09 | PASS | R > 0 (merge lowered the loss) in 29.5% of merges |
| D10 | device baseline b <- (1-rho) b + rho R (starts at 0) | Algorithm 2 l.13 | 4125 | 0.0e+00 | 1e-12 | PASS |  |
| Q1 | H(s,a) <- H + phi (R + psi max_a' H(s',a') - H) | Eq. 6 | 4042 | 0.0e+00 | 1e-09 | PASS |  |
| Q2 | action add/stay/minus moves l* by +1/0/-1; device requests at the new l* | Eq. 6 (actions) | 4042 |  |  | PASS |  |
| Q3 | epsilon-greedy: greedy choices >= 1 - epsilon = 0.90 (minus ties) | Eq. 6 (policy) | 4042 |  |  | PASS | greedy share 94.6% |
| M1 | LSTM decides the first request slot of each device once; the device uses it | Eq. 5 | 100 |  |  | PASS | 100 devices; later trainings use Q-learning (4900 rows) |
| M2 | sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1 | Eq. 5 | 100 |  |  | PASS |  |
| M3 | advantage = R - b (moving-average baseline) | Eq. 5 / REINFORCE | 83 | 0.0e+00 | 1e-09 | PASS |  |
| M4 | policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b | Eq. 5 / REINFORCE | 83 |  |  | PASS | direction correct in 100.0% of updates |

## Example log rows

```
## Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)
received=28 device=65 t=27 o=20 staleness=8
Eq.3  lam 1 -> 1.00002, sig 0.5 -> 0.499963, iota 0.01 -> 0.0241278   (g.dprev=-0.07335)
Eq.2  xi = 1.00002/(sqrt(27)*8^0.499963) + 0.0241278 = 0.0921755;  alpha = 10*xi/(1+10*xi) = 0.479642
Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = 0.479642  (alpha 0.479642)

## Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)
received=2 device=33 started from version o=0, request after epoch 3 (meta), received fresh version g=1
F8  phi = 1/sqrt(1)*(1-0.5/sqrt(2)) = 0.646447;  beta = 0.866032
F7  measured |w_after - w_local| / |w_g - w_local| = 0.866032  (beta 0.866032)
F9  gamma 1 -> 0.972224, upsilon 0.5 -> 0.500304
reward R = 1.2234 - 2.9716 = -1.7482

## Device Q-learning, one update (Eq. 6)
device=65 state l*=4 action=add -> l*=4 (greedy)
H 0.00000 -> -0.01823 = H + 0.1*(-0.1823 + 0.9*0.00000 - H)

## Server LSTM meta model, one REINFORCE update (Eq. 5)
device=33 features=[0.217584, 0.3035] P(l*)=[0.510364, 0.247867, 0.121786, 0.119983] decisions=[0, 0, 1] -> l*=3
R=-1.7482 b=-0.0181 advantage=-1.7301; log P(decisions) -2.10549 -> -2.10600

```
