# src/tests/test_utils.py
import numpy as np
from utils import ReplayBuffer

def test_replaybuffer_push_sample():
    buf = ReplayBuffer(10)
    for i in range(5):
        buf.push(i, i%2, float(i), i+1, False)
    assert len(buf) == 5
    batch = buf.sample(3)
    assert len(batch.s) == 3
