"""Per-device Q-learning time-slot selector (Eq. 6).

State  : the current request epoch l* in {1, ..., L-1} (request *after* l* local
         epochs; at least one epoch remains after the merge).
Actions: {add, stay, minus} -> l* + 1, l*, l* - 1 (clipped to the valid range).
         (The paper's text defines "stay" as l*+1, an evident typo; stay keeps l*.)
Update : H(l_prev, a_prev) += phi * (R + psi * max_a H(l_cur, a) - H(l_prev, a_prev))
Policy : epsilon-greedy over H(l_cur, .).
"""
import numpy as np

ACTIONS = (+1, 0, -1)  # add, stay, minus
ACTION_NAMES = ("add", "stay", "minus")


class QTimeSlotSelector:
    def __init__(self, max_slot: int, phi: float, psi: float, epsilon: float, seed: int):
        self.max_slot = max_slot  # valid l* in 1..max_slot
        self.phi, self.psi, self.eps = phi, psi, epsilon
        self.H = np.zeros((max_slot + 1, len(ACTIONS)))  # row 0 unused
        self.rng = np.random.RandomState(seed)
        self.cur = None  # current state l*
        self.last = None  # (l_prev, a_idx) that produced cur
        self.n_updates = 0

    def init_state(self, l_star: int):
        self.cur = int(np.clip(l_star, 1, self.max_slot))
        self.last = None

    def select(self) -> int:
        """Choose an action at state ``cur`` and move to the next l*."""
        if self.rng.rand() < self.eps:
            a = self.rng.randint(len(ACTIONS))
        else:
            row = self.H[self.cur]
            best = np.flatnonzero(row == row.max())
            a = int(self.rng.choice(best))
        nxt = int(np.clip(self.cur + ACTIONS[a], 1, self.max_slot))
        self.last = (self.cur, a)
        self.cur = nxt
        return nxt

    def update(self, reward: float):
        """Apply Eq. 6 for the transition (last state, last action) -> cur."""
        if self.last is None:
            return None
        l_prev, a = self.last
        target = reward + self.psi * self.H[self.cur].max()
        self.H[l_prev, a] += self.phi * (target - self.H[l_prev, a])
        self.n_updates += 1
        return float(self.H[l_prev, a])

    def state_dict(self):
        return {"H": self.H.tolist(), "cur": self.cur, "last": self.last, "n": self.n_updates,
                "rng": self.rng.get_state()}

    def load_state_dict(self, d):
        self.H = np.array(d["H"])
        self.cur, self.last, self.n_updates = d["cur"], (tuple(d["last"]) if d["last"] else None), d["n"]
        self.rng.set_state(d["rng"])
