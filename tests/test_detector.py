"""Readout and alarm, on synthetic maps and masks (paper sections 2.2-2.3)."""

import numpy as np
import pytest

import common
import detector

CFG = {"top_k": 8, "coverage": 0.1, "weak": 0.4, "share": 0.5, "alarm_count": 5}


def patches(*cells):
    """A 16 x 16 attention map: equal weight on the given (row, col) patches, a little everywhere else."""
    m = np.full((16, 16), 1e-4, np.float32)
    for r, c in cells:
        m[r, c] = 1.0
    return m


def mask(r0, r1, c0, c1):
    """A 224 x 224 mask covering patches rows r0..r1-1, cols c0..c1-1 entirely."""
    m = np.zeros((224, 224), bool)
    m[r0 * 14:r1 * 14, c0 * 14:c1 * 14] = True
    return m


def readout(attention, masks, task):
    return detector.readout(attention, common.patch_coverage(np.array(masks)), task, CFG)


A, B, C = mask(0, 4, 0, 4), mask(8, 12, 8, 12), mask(0, 4, 12, 16)
ON_A = [(r, c) for r in range(2) for c in range(4)]          # 8 patches inside A
ON_B = [(8 + r, 8 + c) for r in range(2) for c in range(4)]  # 8 patches inside B


def test_on_target():
    verdict, other, shares, mass = readout(patches(*ON_A), [A, B], [0])
    assert verdict == detector.ON and other is None
    assert shares[0] == pytest.approx(1.0, abs=1e-3) and mass > 0.9


def test_off_target_names_the_candidate():
    verdict, other, *_ = readout(patches(*ON_B), [A, B, C], [0, 2])
    assert verdict == detector.OFF and other == 1


def test_weak_attention_abstains():
    verdict, *_ = readout(np.ones((16, 16), np.float32), [A, B], [0])   # top 8 of 256 equal patches hold 3 %
    assert verdict == detector.ABSTAIN


def test_attention_off_every_candidate_abstains():
    verdict, other, shares, _ = readout(patches(*[(14, c) for c in range(8)]), [A, B], [0])
    assert verdict == detector.ABSTAIN and shares.sum() == 0


def test_patch_goes_whole_to_the_candidate_covering_most():
    # patch (0, 0): candidate 0 covers 3/14 rows (0.21), candidate 1 the other 11/14 (0.79)
    a = np.zeros((224, 224), bool)
    a[:3, :14] = True
    b = np.zeros((224, 224), bool)
    b[3:14, :14] = True
    verdict, other, shares, _ = readout(patches((0, 0)), [a, b], [0])
    assert shares[1] > 0.99 and shares[0] == 0 and verdict == detector.OFF and other == 1


def test_coverage_below_threshold_is_outside():
    a = np.zeros((224, 224), bool)
    a[:1, :14] = True                                  # 1/14 = 7 % of the patch, below 10 %
    _, _, shares, _ = readout(patches((0, 0)), [a], [0])
    assert shares[0] == 0


def test_split_attention_abstains():
    # 4 patches on the task object, 4 on another: 0.5 each -> task share >= 0.5 and >= other -> on-target;
    # 3 / 3 / 2 off-candidate -> neither reaches 0.5 -> abstain
    on = readout(patches(*ON_A[:4], *ON_B[:4]), [A, B], [0])[0]
    split = readout(patches(*ON_A[:3], *ON_B[:3], (14, 0), (14, 1)), [A, B], [0])[0]
    assert on == detector.ON and split == detector.ABSTAIN


def test_alarm_counts_any_other_object_and_keeps_through_abstain():
    v = [detector.OFF, detector.OFF, detector.ABSTAIN, detector.OFF, detector.OFF, detector.OFF]
    fired, counts = detector.alarm(v, 5)
    assert fired == 5 and counts == [1, 2, 2, 3, 4, 5]


def test_on_target_resets_the_count():
    v = [detector.OFF] * 4 + [detector.ON] + [detector.OFF] * 4
    fired, counts = detector.alarm(v, 5)
    assert fired is None and counts[4] == 0 and counts[-1] == 4


def test_alarm_fires_once_at_the_first_crossing():
    v = [detector.OFF] * 5 + [detector.ON] + [detector.OFF] * 5
    fired, _ = detector.alarm(v, 5)
    assert fired == 4


def test_run_over_calls():
    attention = np.stack([patches(*ON_B)] * 6)
    masks = np.stack([np.array([A, B])] * 6)
    r = detector.run(attention, masks, [0], CFG)
    assert r["verdicts"] == [detector.OFF] * 6 and r["alarm_call"] == 4 and r["off_target"][0] == 1
