# FedASMU reproduction conditions

Sources:
- **[AAAI]**: Liu et al., *FedASMU*, AAAI-24, pp. 13900–13908 (https://ojs.aaai.org/index.php/AAAI/article/view/29297).
- **[arXiv]**: the extended version arXiv:2312.05770 (appendix: derivatives, experiment details, Tables 3–6).
- **[FedBuff]**: Nguyen et al., AISTATS 2022 (https://proceedings.mlr.press/v151/nguyen22b.html).

**Official code.** No public implementation by the FedASMU authors could be found (web search, 2026-11-05). Everything below is therefore based on the paper. No unofficial implementation was used as a reference.

Status values:
- **Paper**: stated in the paper (AAAI or arXiv).
- **Official code**: confirmed in the authors' code (none exists).
- **Assumption**: not published; chosen for this reproduction.
- **Unknown**: cannot be determined from the available material.

## 1. System and data

| Item | Value used | Status | Source / rationale |
|---|---|---|---|
| Devices m | 100 | Paper | [AAAI] Experimental Setup; [arXiv] Tables 5–6 |
| Devices triggered m′ / max parallelism | 10 (10 % of m) | Paper | [arXiv] "maximum parallelism constraint, i.e., 10% of the total device number" |
| Trigger period 𝒯 | 10 virtual seconds; idle devices are filled up to m′ | Paper (value) / Assumption (fill rule) | Tables 5–6 give 𝒯 = 10. The text also says "one idle task every 5 seconds". |
| Staleness bound τ | 99 | Paper | Tables 5–6 |
| Global budget | 5000 received local updates for async methods, 500 rounds for FedAvg | Paper (number) / Assumption (unit) | [arXiv]: "500 as the maximum number of epochs for synchronous approaches and 5000 … asynchronous". We read "epochs" as global rounds/updates. |
| Non-IID split | Label-skew Dirichlet, NIID-Bench procedure | Paper (method) | [AAAI] cites Li et al. 2021 |
| Dirichlet α | 0.5 | Assumption | Not given; 0.5 is a common NIID-Bench setting |
| Train/test split | Official CIFAR train 50 000 / test 10 000; test set used only for reporting | Assumption | Not given |
| Resolution / normalisation | 32×32 RGB, per-channel mean/std normalisation, no augmentation | Assumption | Not given |
| System heterogeneity | Per-device epoch time = base × U[1, 5] | Paper | [arXiv]: "the slowest device is five times longer than the fastest … randomly sampled within this range" |
| Base epoch time | 10 s (CNN, LeNet), 30 s (VGG, ResNet, AlexNet) | Assumption | Time unit/hardware not given; only the 5× ratio is specified |
| Communication delay | Model size (MB) / bw_i, bw_i ~ U[1, 5] MB/s, symmetric up/down; fresh-model request latency 0.1 s | Assumption | The paper says "normal bandwidth" without a number |
| Seeds | 0 (pilot); 0, 1, 2 (paper profile) | Assumption | Not given |
| Time measurement | Virtual seconds of the simulated system; real wall-clock logged separately | Assumption | The paper reports "Time (s)" from a 44-GPU simulation, method unknown |
| Accuracy measure | Global model test accuracy. "Final" = mean of the last 5 evaluations; "best" = maximum | Assumption | Paper: "convergence accuracy of the global model" |
| Target accuracies | CNN-C10 0.40, CNN-C100 0.15, VGG-C10 0.45, VGG-C100 0.12, LeNet-C10 0.30, LeNet-C100 0.13, ResNet-C100 0.25, AlexNet-C10 0.40, AlexNet-C100 0.12 | Paper | Captions of Tables 1–2 |

## 2. Models

| Model | Definition used | Status |
|---|---|---|
| CNN | conv(64, k=2) → act → conv(32, k=2) → act → flatten → dense(C). Activation = ReLU; stride 1; no padding | Paper ([arXiv] Table 3) / Assumption (ReLU, padding) |
| VGG-11 | Configuration A [64,M,128,M,256,256,M,512,512,M,512,512,M]; 3×3 convolutions with padding 1; no BatchNorm; classifier 512-512-C with dropout 0.5; Kaiming init | Paper (name, Simonyan & Zisserman) / Assumption (CIFAR head, no BN) |
| LeNet-5 | conv6-5 → pool → conv16-5 → pool → 120 → 84 → C | Assumption (standard) |
| ResNet-20 | He et al. 2016 CIFAR version (3×3 stages 16/32/64, BN) | Assumption (standard) |
| AlexNet | CIFAR-sized variant: 5 conv + 3 FC | Assumption |
| TextCNN / IMDb, ResNet-20 / Tiny-ImageNet, LeNet / FMNIST | Not prepared | Not run |

## 3. Local training

| Item | Value | Status |
|---|---|---|
| Optimizer | Plain mini-batch SGD (Eq. 4), no momentum or weight decay | Paper (SGD) |
| Learning rate η_i | CNN-C10 0.028, CNN-C100 0.013, LeNet-C10/C100 0.03, VGG/AlexNet/ResNet 0.03 | Paper (Tables 5–6) |
| LR schedule | η · 0.9998^(#received updates); same for all algorithms | Paper ("learning rate decay") / Assumption (form) |
| Local epochs L | 5 full passes over the local data. Request slot l* ∈ {1..L−1}: request after l* epochs | Assumption |
| Batch size | 50 | Assumption |

The paper's Formula 4 writes one SGD mini-batch step per "epoch" l. We use full local epochs, and the pseudo-gradient normalisation uses the true number of SGD steps (η · steps). This keeps it consistent with the appendix's η·L under either reading.

## 4. FedASMU server (Algorithm 1, Eq. 1–3)

| Item | Value / implementation | Status |
|---|---|---|
| Staleness | τ_i = t − o + 1; discard if τ_i > τ (= 99) | Paper |
| Eq. 2 | ξ = λ / (√t · (t−o+1)^σ) + ι; α = µ_α ξ / (1 + µ_α ξ) | Paper |
| Eq. 1 | w_{t+1} = (1−α) w_t + α w_o^i | Paper |
| Eq. 3 | λ, σ, ι ← x − η_x ∂F/∂x, where ∂F/∂x ≈ g^T (w_{o′}^i − w_{o−1}) · µ_α/(1+µ_α ξ_prev)² · ∂ξ/∂x | Paper ([arXiv] appendix) |
| η_λ, η_σ, η_ι | Per Tables 5–6 (e.g. CNN-C10: 1e-3, 1e-3, 1e-4) | Paper |
| µ_α, λ₀, σ₀, ι₀ | 10, 1.0, 0.5, 0.01 | Assumption. At t=1, τ=1: α≈0.91; late in training α floor ≈0.09. σ₀ = 0.5 mirrors FedAsync's exponent. |
| Pseudo-gradient sign | g = (w_o − w_o^i)/(η·steps) (the true SGD gradient direction) | Assumption (interpretation). The appendix prints (w_o^i − w_o)/(ηL), i.e. the negative gradient. `paper_sign: true` reproduces the printed sign. |
| ∂ξ/∂σ | Exact: −λ ln(τ_prev) / (√t_prev · τ_prev^σ) | Interpretation. The appendix prints ln(σ) in place of λ·ln(τ), which we treat as a typo. With ln(σ), a stationary point would not correspond to Eq. 2. |
| Previous-upload information | Per device: last accepted direction (w_{o′}^i − w_{t_prev}), t_prev, τ_prev, ξ_prev. Applied at the device's next upload against the model w_o it started from. | Interpretation. The appendix assumes the device restarts from exactly the version its previous upload produced, which is not true in general under asynchrony. |
| Edge cases | √t uses max(t,1); ξ clipped ≥ 0; λ, σ projected ≥ 0 | Assumption |

## 5. FedASMU device side (Algorithm 2, Eq. 5–9)

| Item | Value / implementation | Status |
|---|---|---|
| Request | Once per local training (only one fresh model), after l* epochs, via an NVFLARE aux message; the server answers at the virtual request time and sends w_g only if g > o | Paper |
| Eq. 7–8 | w^a = (1−β) w^b + β w_g; φ = γ/√g · (1 − υ/√(g−o+1)); β = µ_β φ/(1+µ_β φ) | Paper |
| Eq. 9 | γ, υ updated with the gradient at the merged model on mini-batch ζ (appendix chain rule) | Paper |
| µ_β, γ₀, υ₀ | 10, 1.0, 0.5 | Assumption |
| η_γ, η_υ | Tables 5–6 | Paper |
| Reward | R = loss(before merge, ζ) − loss(after merge, ζ), same mini-batch | Paper |
| Baseline b | b ← (1−ρ) b + ρ R, ρ = 0.1 | Paper (form) / Assumption (ρ) |
| Meta model | Server-side LSTMCell(hidden 32) + FC. It emits a request decision for each epoch l = 1..L−2 (stop at the first 1); l* = L−1 if none. Inputs: previous decision + device features (normalised slowdown, normalised data size). Used for a device's first training only. | Paper (LSTM+FC, first training) / Assumption (inputs, size) |
| Meta update (Eq. 5) | REINFORCE with η_RL = 0.001 and an EMA baseline over first-training rewards; applied on the server when the device returns R | Paper (Eq. 5, η_RL) / Assumption (baseline location) |
| Meta pretraining | "Pre-trained with some historical data": the (features, decisions, R) history of the pilot FedASMU run of the same model/dataset is used to pretrain later runs. Pilot runs have no history, so no pretraining. | Paper (statement) / Assumption (data source) |
| Q-learning (Eq. 6) | State l*; actions add/stay/minus (the paper's "stay = l*+1" is treated as a typo); ε-greedy ε=0.1; φ=0.1, ψ=0.9; Q initialised to 0 at state l* from the meta model | Paper (form) / Assumption (ε, φ, ψ) |
| Fixed-slot heuristics | H1 = after epoch 1, H2 = middle, H3 = L−1 (`fedasmu.fixed_slot`), ablation only | Paper ([arXiv] "Ablation Study for Request Time Slot Selection") |

## 6. FedBuff

| Item | Value | Status |
|---|---|---|
| K | 10 | [FedBuff] ("K = 10 is a good choice and does not require tuning"). Not given in the FedASMU paper. |
| Client message | Model difference w_local − w_received | [FedBuff] Algorithm 2 |
| Server update | w ← w + η_g · (Σ s(τ) Δ_i)/K, η_g = 1.0 | [FedBuff] Algorithm 1 / Assumption (η_g; the paper tunes it) |
| Staleness weight | Official NVFLARE: 1/(1+√(τ+1)). Paper variant: 1/√(1+τ) | See ARCHITECTURE.md §FedBuff comparison |
| Server momentum | None | Assumption (the FedBuff paper tunes momentum per task) |

## 7. Ablations (paper Fig. 2)

| Name | Definition here |
|---|---|
| FedASMU-DA | FedASMU without dynamic server aggregation: λ, σ, ι frozen at their initial values (static staleness polynomial) |
| FedASMU-FA | FedASMU without fresh-global-model aggregation on devices (no request) |
| FedASMU-0 | Neither of the two, i.e. a static polynomial α with the staleness bound τ = 99. This matches the paper's statement "equivalent to FedAsync with staleness bound". |
| FedAvg | Synchronous rounds of m′ = 10 random devices, sample-size weighted average, 500 rounds |

## 8. Fidelity notes

- The paper's own hyper-parameters were tuned per method ("we fine-tune the hyper-parameters … and report the best one"). We did no tuning: no test-set tuning, and no validation tuning either because of the compute budget. Fixed assumptions are used for both algorithms.
- 100 logical devices are multiplexed onto 10 NVFLARE site processes, one per concurrently training device. Each logical device keeps its own data shard, RL state and RNG stream in its own state file. See ARCHITECTURE.md.
