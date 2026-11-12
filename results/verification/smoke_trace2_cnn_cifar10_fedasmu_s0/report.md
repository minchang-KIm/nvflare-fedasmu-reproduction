# Verification of smoke_trace2_cnn_cifar10_fedasmu_s0

Run status: COMPLETED; 60 logged updates. Result: **ALL CHECKS PASSED** (28/28).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| S0 | at most m' = 4 devices train at the same time (triggered every T) | Algorithm 1 l.1-6 | 9 |  |  | PASS | devices in training at evaluation points: max 3, mean 2.4 |
| S1 | staleness tau = t - o + 1 | Algorithm 1 l.9 | 60 | 0.0e+00 | 0e+00 | PASS |  |
| S2 | update discarded iff tau > 99 | Algorithm 1 l.9-10 | 60 |  |  | PASS | discarded: 0 |
| S3 | xi = lam/(sqrt(t)(t-o+1)^sig) + iota | Eq. 2 | 60 | 0.0e+00 | 1e-09 | PASS |  |
| S4 | alpha = mu*xi/(1+mu*xi) | Eq. 2 | 60 | 0.0e+00 | 1e-09 | PASS |  |
| S5 | new global model equals (1-alpha) w_t + alpha w_i element by element (relative residual) | Eq. 1 | 60 | 0.0e+00 | 1e-06 | PASS |  |
| S5b | mixing ratio measured on the tensors \|w_{t+1}-w_t\|/\|w_i-w_t\| = alpha, within float32 rounding bound 2^-24 \|w_{t+1}\|/\|w_i-w_t\| | Eq. 1 | 60 |  |  | PASS | max \|ratio-alpha\| 5.0e-08, max bound 1.0e-06 |
| S6 | dF/dlam, dF/dsig, dF/diota (chain rule through alpha) | Eq. 3 + appendix A | 52 | 5.6e-17 | 1e-09 | PASS |  |
| S7 | lam/sig/iota <- x - eta_x * dF/dx | Eq. 3 | 52 | 0.0e+00 | 1e-09 | PASS | first upload of a device has no previous update: 8 rows, parameters unchanged (OK) |
| S8 | per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0 | Algorithm 1 (state per device) | 60 | 0.0e+00 | 1e-12 | PASS |  |
| D1 | request slot l* in 1..L-1 (L = 3) | Algorithm 2 l.4 | 60 |  |  | PASS |  |
| D2 | fresh model merged only if it is newer (g > o) | Algorithm 2 l.6-7 | 46 |  |  | PASS | requests 60, merged 46, no newer model 14 |
| D3 | phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)) | Formula 8 | 46 | 0.0e+00 | 1e-09 | PASS |  |
| D4 | beta = mu*phi/(1+mu*phi) | Formula 8 | 46 | 0.0e+00 | 1e-09 | PASS |  |
| D5 | local model after merge equals (1-beta) w_local + beta w_g element by element (relative residual) | Formula 7 | 46 | 0.0e+00 | 1e-06 | PASS |  |
| D5b | mixing ratio measured on the tensors = beta, within float32 rounding bound | Formula 7 | 46 |  |  | PASS | max \|ratio-beta\| 2.9e-08, max bound 1.3e-06 |
| D6 | dF/dgamma, dF/dupsilon | Formula 9 + appendix A | 46 | 0.0e+00 | 1e-09 | PASS |  |
| D7 | gamma/upsilon <- x - eta_x * dF/dx | Formula 9 | 46 | 0.0e+00 | 1e-09 | PASS |  |
| D8 | per-device gamma: value before a merge == value after its previous merge; start at gamma0 | Algorithm 2 (device state) | 46 | 0.0e+00 | 1e-12 | PASS |  |
| D9 | reward R = loss(before merge) - loss(after merge), same mini-batch | Algorithm 2 l.11 | 46 | 0.0e+00 | 1e-09 | PASS | R > 0 (merge lowered the loss) in 10.9% of merges |
| D10 | device baseline b <- (1-rho) b + rho R (starts at 0) | Algorithm 2 l.13 | 46 | 0.0e+00 | 1e-12 | PASS |  |
| Q1 | H(s,a) <- H + phi (R + psi max_a' H(s',a') - H) | Eq. 6 | 41 | 0.0e+00 | 1e-09 | PASS |  |
| Q2 | action add/stay/minus moves l* by +1/0/-1; device requests at the new l* | Eq. 6 (actions) | 41 |  |  | PASS |  |
| Q3 | epsilon-greedy: greedy choices >= 1 - epsilon = 0.90 (minus ties) | Eq. 6 (policy) | 41 |  |  | PASS | greedy share 92.7% |
| M1 | LSTM decides the first request slot of each device once; the device uses it | Eq. 5 | 8 |  |  | PASS | 8 devices; later trainings use Q-learning (52 rows) |
| M2 | sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1 | Eq. 5 | 8 |  |  | PASS |  |
| M3 | advantage = R - b (moving-average baseline) | Eq. 5 / REINFORCE | 5 | 0.0e+00 | 1e-09 | PASS |  |
| M4 | policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b | Eq. 5 / REINFORCE | 5 |  |  | PASS | direction correct in 100.0% of updates |

## Example log rows

```
## Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)
received=8 device=5 t=7 o=6 staleness=2
Eq.3  lam 1 -> 1.00002, sig 0.5 -> 0.499977, iota 0.01 -> 0.0100073   (g.dprev=-0.08178)
Eq.2  xi = 1.00002/(sqrt(7)*2^0.499977) + 0.0100073 = 0.277277;  alpha = 10*xi/(1+10*xi) = 0.734943
Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = 0.734943  (alpha 0.734943)

## Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)
received=4 device=6 started from version o=0, request after epoch 2 (meta), received fresh version g=2
F8  phi = 1/sqrt(2)*(1-0.5/sqrt(3)) = 0.502983;  beta = 0.834158
F7  measured |w_after - w_local| / |w_g - w_local| = 0.834158  (beta 0.834158)
F9  gamma 1 -> 0.949219, upsilon 0.5 -> 0.500412
reward R = 1.4691 - 2.9291 = -1.4599

## Device Q-learning, one update (Eq. 6)
device=5 state l*=2 action=stay -> l*=2 (greedy)
H 0.00000 -> -0.04927 = H + 0.1*(-0.4927 + 0.9*0.00000 - H)

## Server LSTM meta model, one REINFORCE update (Eq. 5)
device=6 features=[0.950662, 0.25] P(l*)=[0.518139, 0.481861] decisions=[0] -> l*=2
R=-1.4599 b=-0.0218 advantage=-1.4381; log P(decisions) -0.73010 -> -0.73056

```
