"""Server-side LSTM meta model for the initial request time slot (Eq. 5).

An LSTM cell + fully-connected layer emits, for local epochs l = 1..L-1, the
probability of sending the request after epoch l (decision s_l in {0,1});
decoding stops at the first 1 (only one fresh model per local training).  If no
1 is sampled before L-1, the request is forced after epoch L-1 (that forced step
contributes no log-probability).  Training follows REINFORCE with baseline
(Zoph & Le 2017, as cited):  theta += eta_RL * sum_l grad log P(s_l|s_<l) (R - b).
Inputs per step: [previous decision one-hot (2), device features (2)] where the
device features are the normalised compute slowdown and local data size
(assumption: the paper does not list the LSTM inputs).
"""
import torch
import torch.nn as nn


class MetaTimeSlotLSTM(nn.Module):
    def __init__(self, max_slot: int, hidden: int = 32, n_feat: int = 2):
        super().__init__()
        self.max_slot = max_slot
        self.cell = nn.LSTMCell(2 + n_feat, hidden)
        self.fc = nn.Linear(hidden, 1)
        self.hidden = hidden

    def _logits_iter(self, feat: torch.Tensor, decisions):
        h = torch.zeros(1, self.hidden)
        c = torch.zeros(1, self.hidden)
        prev = torch.tensor([[1.0, 0.0]])
        for d in decisions:
            h, c = self.cell(torch.cat([prev, feat.view(1, -1)], 1), (h, c))
            yield self.fc(h).view(())
            prev = torch.tensor([[0.0, 1.0]]) if d else torch.tensor([[1.0, 0.0]])

    @torch.no_grad()
    def sample(self, feat, gen: torch.Generator):
        """Return (l_star, decisions list) sampled from the policy."""
        feat = torch.as_tensor(feat, dtype=torch.float32)
        h = torch.zeros(1, self.hidden)
        c = torch.zeros(1, self.hidden)
        prev = torch.tensor([[1.0, 0.0]])
        decisions = []
        for l in range(1, self.max_slot):
            h, c = self.cell(torch.cat([prev, feat.view(1, -1)], 1), (h, c))
            p = torch.sigmoid(self.fc(h)).item()
            d = int(torch.rand((), generator=gen).item() < p)
            decisions.append(d)
            if d:
                return l, decisions
            prev = torch.tensor([[1.0, 0.0]])
        return self.max_slot, decisions  # forced request after epoch L-1

    def log_prob(self, feat, decisions) -> torch.Tensor:
        feat = torch.as_tensor(feat, dtype=torch.float32)
        lp = torch.zeros(())
        for logit, d in zip(self._logits_iter(feat, decisions), decisions):
            lp = lp + (nn.functional.logsigmoid(logit) if d else nn.functional.logsigmoid(-logit))
        return lp

    def probs(self, feat):
        """Distribution over l* (for logging / tests)."""
        with torch.no_grad():
            out, surv = [], 1.0
            feat = torch.as_tensor(feat, dtype=torch.float32)
            for l, logit in enumerate(self._logits_iter(feat, [0] * (self.max_slot - 1)), start=1):
                p = torch.sigmoid(logit).item()
                out.append(surv * p)
                surv *= 1 - p
            out.append(surv)
            return out


class MetaController:
    """REINFORCE trainer around the LSTM (Eq. 5) with a moving-average baseline."""

    def __init__(self, max_slot: int, lr: float, rho: float, seed: int, hidden: int = 32):
        torch.manual_seed(seed)
        self.model = MetaTimeSlotLSTM(max_slot, hidden)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.baseline = 0.0
        self.rho = rho
        self.gen = torch.Generator().manual_seed(seed + 1)
        self.n_updates = 0

    def decide(self, feat):
        return self.model.sample(feat, self.gen)

    def update(self, feat, decisions, reward: float):
        if not decisions:
            return 0.0
        adv = reward - self.baseline
        self.opt.zero_grad()
        loss = -adv * self.model.log_prob(feat, decisions)
        loss.backward()
        self.opt.step()  # theta <- theta + eta_RL * grad log P * (R - b)
        self.baseline = (1 - self.rho) * self.baseline + self.rho * reward
        self.n_updates += 1
        return adv

    def pretrain(self, history, epochs: int = 3):
        """Pre-train on historical (feat, decisions, reward) records (paper: 'pre-trained
        with some historical data').  Returns the number of records used."""
        for _ in range(epochs):
            for rec in history:
                self.update(rec["feat"], rec["decisions"], rec["reward"])
        return len(history)

    def state_dict(self):
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "baseline": self.baseline,
                "gen": self.gen.get_state(), "n": self.n_updates}

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.baseline, self.n_updates = d["baseline"], d["n"]
        self.gen.set_state(d["gen"])
