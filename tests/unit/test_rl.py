"""Q-learning (Eq. 6) and LSTM meta model (Eq. 5)."""
import numpy as np
import pytest
import torch

from src.algorithms.fedasmu.meta_model import MetaController, MetaTimeSlotLSTM
from src.algorithms.fedasmu.q_learning import ACTIONS, QTimeSlotSelector


def test_q_update_hand_computed():
    q = QTimeSlotSelector(max_slot=4, phi=0.5, psi=0.9, epsilon=0.0, seed=0)
    q.init_state(2)
    q.H[3] = [0.0, 2.0, 0.0]
    q.H[2] = [1.0, 0.0, 0.0]  # greedy -> action 'add' (+1) -> state 3
    nxt = q.select()
    assert nxt == 3 and q.last == (2, 0)
    q.update(reward=1.0)
    # H(2,add) = 1 + 0.5 * (1 + 0.9*2 - 1) = 1.9
    assert q.H[2, 0] == pytest.approx(1.9)


def test_q_actions_clip_and_stay():
    q = QTimeSlotSelector(max_slot=3, phi=0.5, psi=0.9, epsilon=0.0, seed=0)
    q.init_state(3)
    q.H[3] = [5.0, 0.0, 0.0]  # add at the upper bound is clipped
    assert q.select() == 3
    q.H[3] = [0.0, 5.0, 0.0]  # stay keeps l*
    assert q.select() == 3
    q.H[3] = [0.0, 0.0, 5.0]
    assert q.select() == 2
    assert ACTIONS == (1, 0, -1)


def test_q_epsilon_explores_and_state_roundtrip():
    q = QTimeSlotSelector(max_slot=4, phi=0.5, psi=0.9, epsilon=1.0, seed=3)
    q.init_state(2)
    seen = {q.select() for _ in range(50)}
    assert len(seen) > 1
    sd = q.state_dict()
    q2 = QTimeSlotSelector(max_slot=4, phi=0.5, psi=0.9, epsilon=1.0, seed=99)
    q2.load_state_dict(sd)
    assert [q.select() for _ in range(5)] == [q2.select() for _ in range(5)]


def test_meta_probs_and_sampling():
    m = MetaTimeSlotLSTM(max_slot=4)
    p = m.probs([0.5, 0.2])
    assert len(p) == 4 and sum(p) == pytest.approx(1.0)
    slot, dec = m.sample([0.5, 0.2], torch.Generator().manual_seed(0))
    assert 1 <= slot <= 4 and (dec[-1] == 1 or slot == 4)


def test_reinforce_increases_prob_of_rewarded_slot():
    mc = MetaController(max_slot=4, lr=0.5, rho=0.1, seed=0)
    feat = [0.3, 0.7]
    before = mc.model.probs(feat)[0]
    for _ in range(30):
        mc.update(feat, [1], reward=1.0)  # request after epoch 1 is rewarded
    assert mc.model.probs(feat)[0] > before
    sd = mc.state_dict()
    mc2 = MetaController(max_slot=4, lr=0.5, rho=0.1, seed=5)
    mc2.load_state_dict(sd)
    assert np.allclose(mc2.model.probs(feat), mc.model.probs(feat))
