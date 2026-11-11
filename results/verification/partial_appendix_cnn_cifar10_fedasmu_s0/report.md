# Verification of appendix_cnn_cifar10_fedasmu_s0

Run status: RUNNING; 2810 logged updates. Result: **ALL CHECKS PASSED** (24/24).

| ID | Check | Paper | Rows | Max error | Tolerance | Result | Note |
|---|---|---|---|---|---|---|---|
| S1 | staleness tau = t - o + 1 | Algorithm 1 l.9 | 2810 | 0.0e+00 | 0e+00 | PASS |  |
| S2 | update discarded iff tau > 99 | Algorithm 1 l.9-10 | 2810 |  |  | PASS | discarded: 0 |
| S3 | xi = lam/(sqrt(t)(t-o+1)^sig) + iota | Eq. 2 | 2810 | 0.0e+00 | 1e-09 | PASS |  |
| S4 | alpha = mu*xi/(1+mu*xi) | Eq. 2 | 2810 | 0.0e+00 | 1e-09 | PASS |  |
| S5 | global model w_{t+1} = (1-alpha) w_t + alpha w_i, measured on the tensors: |w_{t+1}-w_t|/|w_i-w_t| = alpha | Eq. 1 | 2810 | 2.7e-06 | 1e-04 | PASS | max relative residual of the whole parameter vector 0.0e+00 |
| S6 | dF/dlam, dF/dsig, dF/diota (chain rule through alpha) | Eq. 3 + appendix A | 2710 | 1.5e-16 | 1e-09 | PASS |  |
| S7 | lam/sig/iota <- x - eta_x * dF/dx | Eq. 3 | 2710 | 0.0e+00 | 1e-09 | PASS | first upload of a device has no previous update: 100 rows, parameters unchanged (OK) |
| S8 | per-device lam/sig: value before an upload == value after its previous upload; start at lam0/sig0 | Algorithm 1 (state per device) | 2810 | 0.0e+00 | 1e-12 | PASS |  |
| D1 | request slot l* in 1..L-1 (L = 5) | Algorithm 2 l.4 | 2810 |  |  | PASS |  |
| D2 | fresh model merged only if it is newer (g > o) | Algorithm 2 l.6-7 | 2480 |  |  | PASS | requests 2810, merged 2480, no newer model 330 |
| D3 | phi = gamma/sqrt(g) * (1 - upsilon/sqrt(g-o+1)) | Formula 8 | 2480 | 0.0e+00 | 1e-09 | PASS |  |
| D4 | beta = mu*phi/(1+mu*phi) | Formula 8 | 2480 | 0.0e+00 | 1e-09 | PASS |  |
| D5 | local model after merge = (1-beta) w_local + beta w_g, measured on the tensors | Formula 7 | 2480 | 7.2e-08 | 1e-04 | PASS |  |
| D6 | dF/dgamma, dF/dupsilon | Formula 9 + appendix A | 2480 | 2.8e-17 | 1e-09 | PASS |  |
| D7 | gamma/upsilon <- x - eta_x * dF/dx | Formula 9 | 2480 | 0.0e+00 | 1e-09 | PASS |  |
| D8 | per-device gamma: value before a merge == value after its previous merge; start at gamma0 | Algorithm 2 (device state) | 2480 | 0.0e+00 | 1e-12 | PASS |  |
| D9 | reward R = loss(before merge) - loss(after merge), same mini-batch | Algorithm 2 l.11 | 2480 | 0.0e+00 | 1e-09 | PASS | R > 0 (merge lowered the loss) in 27.9% of merges |
| Q1 | H(s,a) <- H + phi (R + psi max_a' H(s',a') - H) | Eq. 6 | 2398 | 0.0e+00 | 1e-09 | PASS |  |
| Q2 | action add/stay/minus moves l* by +1/0/-1; device requests at the new l* | Eq. 6 (actions) | 2398 |  |  | PASS |  |
| Q3 | epsilon-greedy: greedy choices >= 1 - epsilon = 0.90 (minus ties) | Eq. 6 (policy) | 2398 |  |  | PASS | greedy share 94.5% |
| M1 | LSTM decides the first request slot of each device once; the device uses it | Eq. 5 | 100 |  |  | PASS | 100 devices; later trainings use Q-learning (2710 rows) |
| M2 | sampled decisions s_l: first s_l = 1 gives l*; P(l*) sums to 1 | Eq. 5 | 100 |  |  | PASS |  |
| M3 | advantage = R - b (moving-average baseline) | Eq. 5 / REINFORCE | 82 | 0.0e+00 | 1e-09 | PASS |  |
| M4 | policy-gradient step raises log P(chosen slots) when R > b and lowers it when R < b | Eq. 5 / REINFORCE | 82 |  |  | PASS | direction correct in 100.0% of updates |

## Example log rows

```
## Server, one received update (Eq. 3 -> Eq. 2 -> Eq. 1)
received=27 device=13 t=26 o=22 staleness=5
Eq.3  lam 1 -> 1.00013, sig 0.5 -> 0.499795, iota 0.01 -> 0.0101271   (g.dprev=-0.5604)
Eq.2  xi = 1.00013/(sqrt(26)*5^0.499795) + 0.0101271 = 0.0978729;  alpha = 10*xi/(1+10*xi) = 0.494625
Eq.1  measured |w_t+1 - w_t| / |w_i - w_t| = 0.494625  (alpha 0.494625)

## Device, one fresh-model merge (Formula 8 -> 7 -> 9, reward)
received=6 device=65 started from version o=0, request after epoch 4 (meta), received fresh version g=2
F8  phi = 1/sqrt(2)*(1-0.5/sqrt(3)) = 0.502983;  beta = 0.834158
F7  measured |w_after - w_local| / |w_g - w_local| = 0.834158  (beta 0.834158)
F9  gamma 1 -> 0.918371, upsilon 0.5 -> 0.500663
reward R = 0.7705 - 3.6345 = -2.8640

## Device Q-learning, one update (Eq. 6)
device=48 state l*=3 action=stay -> l*=3 (greedy)
H 0.00000 -> 0.00711 = H + 0.1*(0.0711 + 0.9*0.00000 - H)

## Server LSTM meta model, one REINFORCE update (Eq. 5)
device=65 features=[0.513514, 0.108] P(l*)=[0.517807, 0.247991, 0.119941, 0.114261] decisions=[0, 0, 0] -> l*=4
R=-2.8640 b=-0.0218 advantage=-2.8422; log P(decisions) -2.16927 -> -2.17808

```
