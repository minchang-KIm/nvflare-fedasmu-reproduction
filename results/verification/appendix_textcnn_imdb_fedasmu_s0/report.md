# Verification of appendix_textcnn_imdb_fedasmu_s0

Run status: COMPLETED; 5000 logged updates. Result: **SOME CHECKS FAILED** (26/27).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| S0 | at most m' = 10 devices train at the same time (triggered every T) | Algorithm 1 l.1-6 | 102 |  |  | PASS | devices in training at evaluation points: max 9, mean 8.6 |
| S1 | staleness tau = t - o + 1 | Algorithm 1 l.9 | 5000 | 0.0e+00 | 0e+00 | PASS |  |
| S2 | update discarded iff tau > 99 | Algorithm 1 l.9-10 | 5000 |  |  | PASS | discarded: 0 |
| S3 | xi = lam/(sqrt(t)(t-o+1)^sig) + iota | Eq. 2 | 5000 | 0.0e+00 | 1e-09 | PASS |  |
| S4 | alpha = mu*xi/(1+mu*xi) | Eq. 2 | 5000 | 0.0e+00 | 1e-09 | PASS |  |
| S5 | new global model equals (1-alpha) w_t + alpha w_i element by element (relative residual) | Eq. 1 | 5000 | 0.0e+00 | 1e-06 | PASS |  |
| S5b | mixing ratio measured on the tensors \|w_{t+1}-w_t\|/\|w_i-w_t\| = alpha (float32 norms) | Eq. 1 | 5000 | 9.9e-04 | 1e-03 | PASS | run made before the rounding bound was logged; float32 measurement error grows with \|w\|/\|w_i-w_t\| |
| S6 | dF/dlam, dF/dsig, dF/diota (chain rule through alpha) | Eq. 3 + appendix A | 4900 | 1.5e-16 | 1e-09 | PASS |  |
| S7 | lam/sig/iota <- x - eta_x * dF/dx | Eq. 3 | 4900 | 0.0e+00 | 1e-09 | PASS | first upload of a device has no previous update: 100 rows, parameters unchanged (OK) |
| S8 | per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0 | Algorithm 1 (state per device) | 5000 | 0.0e+00 | 1e-12 | PASS |  |
| D1 | request slot l* in 1..L-1 (L = 5) | Algorithm 2 l.4 | 5000 |  |  | PASS |  |
| D2 | fresh model merged only if it is newer (g > o) | Algorithm 2 l.6-7 | 4652 |  |  | PASS | requests 5000, merged 4652, no newer model 348 |
| D3 | phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)) | Formula 8 | 4652 | 0.0e+00 | 1e-09 | PASS |  |
| D4 | beta = mu*phi/(1+mu*phi) | Formula 8 | 4652 | 0.0e+00 | 1e-09 | PASS |  |
| D5 | local model after merge = (1-beta) w_local + beta w_g, mixing ratio measured on the tensors (float32 norms) | Formula 7 | 4652 | 7.2e-03 | 1e-04 | FAIL | run made before the element-wise residual and rounding bound were logged |
| D6 | dF/dgamma, dF/dupsilon | Formula 9 + appendix A | 4652 | 8.7e-19 | 1e-09 | PASS |  |
| D7 | gamma/upsilon <- x - eta_x * dF/dx | Formula 9 | 4652 | 0.0e+00 | 1e-09 | PASS |  |
| D8 | per-device gamma: value before a merge == value after its previous merge; start at gamma0 | Algorithm 2 (device state) | 4652 | 0.0e+00 | 1e-12 | PASS |  |
| D9 | reward R = loss(before merge) - loss(after merge), same mini-batch | Algorithm 2 l.11 | 4652 | 0.0e+00 | 1e-09 | PASS | R > 0 (merge lowered the loss) in 17.5% of merges |
| D10 | device baseline b <- (1-rho) b + rho R (starts at 0) | Algorithm 2 l.13 | 4652 | 0.0e+00 | 1e-12 | PASS |  |
| Q1 | H(s,a) <- H + phi (R + psi max_a' H(s',a') - H) | Eq. 6 | 4565 | 0.0e+00 | 1e-09 | PASS |  |
| Q2 | action add/stay/minus moves l* by +1/0/-1; device requests at the new l* | Eq. 6 (actions) | 4565 |  |  | PASS |  |
| Q3 | epsilon-greedy: greedy choices >= 1 - epsilon = 0.90 (minus ties) | Eq. 6 (policy) | 4565 |  |  | PASS | greedy share 94.8% |
| M1 | LSTM decides the first request slot of each device once; the device uses it | Eq. 5 | 100 |  |  | PASS | 100 devices; later trainings use Q-learning (4900 rows) |
| M2 | sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1 | Eq. 5 | 100 |  |  | PASS |  |
| M3 | advantage = R - b (moving-average baseline) | Eq. 5 / REINFORCE | 87 | 0.0e+00 | 1e-09 | PASS |  |
| M4 | policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b | Eq. 5 / REINFORCE | 87 |  |  | PASS | direction correct in 98.9% of updates |

## Example log rows

```
## Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)
received=35 device=49 t=34 o=23 staleness=12
Eq.3  lam 1 -> 1, sig 0.5 -> 0.499995, iota 0.01 -> 0.0100227   (g.dprev=-0.08815)
Eq.2  xi = 1/(sqrt(34)*12^0.499995) + 0.0100227 = 0.0595308;  alpha = 10*xi/(1+10*xi) = 0.373162
Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = 0.373162  (alpha 0.373162)

## Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)
received=6 device=65 started from version o=0, request after epoch 4 (meta), received fresh version g=2
F8  phi = 1/sqrt(2)*(1-0.5/sqrt(3)) = 0.502983;  beta = 0.834158
F7  measured |w_after - w_local| / |w_g - w_local| = 0.834159  (beta 0.834158)
F9  gamma 1 -> 0.999997, upsilon 0.5 -> 0.500002
reward R = 0.2245 - 0.3349 = -0.1104

## Device Q-learning, one update (Eq. 6)
device=49 state l*=2 action=add -> l*=3 (greedy)
H 0.00000 -> -0.00227 = H + 0.1*(-0.0227 + 0.9*0.00000 - H)

## Server LSTM meta model, one REINFORCE update (Eq. 5)
device=65 features=[0.513514, 0.04] P(l*)=[0.507943, 0.2479, 0.122401, 0.121756] decisions=[0, 0, 0] -> l*=4
R=-0.1104 b=-0.0949 advantage=-0.0155; log P(decisions) -2.10573 -> -2.10578

```
