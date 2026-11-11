# Verification of appendix_lenet5_cifar10_fedasmu_s0

Run status: RUNNING; 3570 logged updates. Result: **ALL CHECKS PASSED** (24/24).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| S1 | staleness tau = t - o + 1 | Algorithm 1 l.9 | 3570 | 0.0e+00 | 0e+00 | PASS |  |
| S2 | update discarded iff tau > 99 | Algorithm 1 l.9-10 | 3570 |  |  | PASS | discarded: 0 |
| S3 | xi = lam/(sqrt(t)(t-o+1)^sig) + iota | Eq. 2 | 3570 | 0.0e+00 | 1e-09 | PASS |  |
| S4 | alpha = mu*xi/(1+mu*xi) | Eq. 2 | 3570 | 0.0e+00 | 1e-09 | PASS |  |
| S5 | global model w_{t+1} = (1-alpha) w_t + alpha w_i, measured on the tensors: |w_{t+1}-w_t|/|w_i-w_t| = alpha | Eq. 1 | 3570 | 1.3e-06 | 1e-04 | PASS | max relative residual of the whole parameter vector 0.0e+00 |
| S6 | dF/dlam, dF/dsig, dF/diota (chain rule through alpha) | Eq. 3 + appendix A | 3470 | 2.8e-17 | 1e-09 | PASS |  |
| S7 | lam/sig/iota <- x - eta_x * dF/dx | Eq. 3 | 3470 | 0.0e+00 | 1e-09 | PASS | first upload of a device has no previous update: 100 rows, parameters unchanged (OK) |
| S8 | per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0 | Algorithm 1 (state per device) | 3570 | 0.0e+00 | 1e-12 | PASS |  |
| D1 | request slot l* in 1..L-1 (L = 5) | Algorithm 2 l.4 | 3570 |  |  | PASS |  |
| D2 | fresh model merged only if it is newer (g > o) | Algorithm 2 l.6-7 | 3162 |  |  | PASS | requests 3570, merged 3162, no newer model 408 |
| D3 | phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)) | Formula 8 | 3162 | 0.0e+00 | 1e-09 | PASS |  |
| D4 | beta = mu*phi/(1+mu*phi) | Formula 8 | 3162 | 0.0e+00 | 1e-09 | PASS |  |
| D5 | local model after merge = (1-beta) w_local + beta w_g, measured on the tensors | Formula 7 | 3162 | 8.7e-08 | 1e-04 | PASS |  |
| D6 | dF/dgamma, dF/dupsilon | Formula 9 + appendix A | 3162 | 1.4e-17 | 1e-09 | PASS |  |
| D7 | gamma/upsilon <- x - eta_x * dF/dx | Formula 9 | 3162 | 0.0e+00 | 1e-09 | PASS |  |
| D8 | per-device gamma: value before a merge == value after its previous merge; start at gamma0 | Algorithm 2 (device state) | 3162 | 0.0e+00 | 1e-12 | PASS |  |
| D9 | reward R = loss(before merge) - loss(after merge), same mini-batch | Algorithm 2 l.11 | 3162 | 0.0e+00 | 1e-09 | PASS | R > 0 (merge lowered the loss) in 30.7% of merges |
| Q1 | H(s,a) <- H + phi (R + psi max_a' H(s',a') - H) | Eq. 6 | 3079 | 0.0e+00 | 1e-09 | PASS |  |
| Q2 | action add/stay/minus moves l* by +1/0/-1; device requests at the new l* | Eq. 6 (actions) | 3079 |  |  | PASS |  |
| Q3 | epsilon-greedy: greedy choices >= 1 - epsilon = 0.90 (minus ties) | Eq. 6 (policy) | 3079 |  |  | PASS | greedy share 94.3% |
| M1 | LSTM decides the first request slot of each device once; the device uses it | Eq. 5 | 100 |  |  | PASS | 100 devices; later trainings use Q-learning (3470 rows) |
| M2 | sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1 | Eq. 5 | 100 |  |  | PASS |  |
| M3 | advantage = R - b (moving-average baseline) | Eq. 5 / REINFORCE | 83 | 0.0e+00 | 1e-09 | PASS |  |
| M4 | policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b | Eq. 5 / REINFORCE | 83 |  |  | PASS | direction correct in 98.8% of updates |

## Example log rows

```
## Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)
received=28 device=13 t=27 o=22 staleness=6
Eq.3  lam 1 -> 1.00004, sig 0.5 -> 0.499933, iota 0.01 -> 0.0513557   (g.dprev=-0.1824)
Eq.2  xi = 1.00004/(sqrt(27)*6^0.499933) + 0.0513557 = 0.129936;  alpha = 10*xi/(1+10*xi) = 0.565096
Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = 0.565096  (alpha 0.565096)

## Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)
received=6 device=65 started from version o=0, request after epoch 4 (meta), received fresh version g=2
F8  phi = 1/sqrt(2)*(1-0.5/sqrt(3)) = 0.502983;  beta = 0.834158
F7  measured |w_after - w_local| / |w_g - w_local| = 0.834158  (beta 0.834158)
F9  gamma 1 -> 0.942238, upsilon 0.5 -> 0.500469
reward R = 1.5909 - 2.8474 = -1.2565

## Device Q-learning, one update (Eq. 6)
device=71 state l*=1 action=add -> l*=2 (greedy)
H 0.00000 -> -0.04342 = H + 0.1*(-0.4342 + 0.9*0.00000 - H)

## Server LSTM meta model, one REINFORCE update (Eq. 5)
device=65 features=[0.513514, 0.108] P(l*)=[0.510776, 0.247961, 0.12171, 0.119554] decisions=[0, 0, 0] -> l*=4
R=-1.2565 b=-0.0181 advantage=-1.2384; log P(decisions) -2.12399 -> -2.12772

```
