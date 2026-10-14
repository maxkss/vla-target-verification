"""First handled object: stable contact plus a real change (scoring labels)."""

import numpy as np

import label

CFG = {"dwell_steps": 10, "min_change": 0.02, "settle_steps": 10}


def run(contacts, change):
    return label.first_handled(np.array(contacts, bool), np.array(change, float), 10, 0.02, 10)


def contact(n_steps, n_objects, spans):
    c = np.zeros((n_steps, n_objects), bool)
    for obj, start, end in spans:
        c[start:end, obj] = True
    return c


def moved(n_steps, n_objects, moves):
    """Change since the start: object `obj` moves by `amount` from step `at` on."""
    ch = np.zeros((n_steps, n_objects))
    for obj, at, amount in moves:
        ch[at:, obj] = amount
    return ch


def test_held_and_moved_is_handled_with_its_onset():
    index, onset, reason, _ = run(contact(60, 2, [(1, 20, 40)]), moved(60, 2, [(1, 30, 0.1)]))
    assert (index, onset, reason) == (1, 20, "handled")


def test_brushing_is_not_contact():
    index, _, reason, _ = run(contact(60, 2, [(0, 5, 9), (1, 20, 40)]), moved(60, 2, [(0, 6, 0.1), (1, 30, 0.1)]))
    assert index == 1   # object 0 touched for 4 steps only


def test_contact_without_change():
    index, _, reason, _ = run(contact(60, 1, [(0, 10, 40)]), moved(60, 1, [(0, 30, 0.015)]))
    assert index is None and reason == "contact without change"


def test_first_by_onset():
    index, onset, _, _ = run(contact(80, 2, [(0, 30, 60), (1, 10, 25)]), moved(80, 2, [(0, 40, 0.1), (1, 15, 0.1)]))
    assert (index, onset) == (1, 10)


def test_two_objects_at_once_leave_no_label():
    index, _, reason, _ = run(contact(60, 2, [(0, 10, 30), (1, 10, 30)]), moved(60, 2, [(0, 15, 0.1), (1, 15, 0.1)]))
    assert index is None and reason == "two objects at once"


def test_object_change_takes_joints_into_account():
    positions = np.zeros((5, 2, 3))
    joints = np.zeros((5, 2))
    joints[3:, 1] = 0.05            # the drawer slide of object 1
    change = label.object_change(positions, joints, np.array([0, 1]))
    assert change[4, 1] == 0.05 and change[4, 0] == 0


def test_label_record():
    n = 40
    traj = {"names": np.array(["bowl", "plate"]), "contacts": contact(n, 2, [(0, 5, 30)]),
            "positions": np.zeros((n, 2, 3)), "joints": np.zeros((n, 0)), "joint_owner": np.zeros(0, int)}
    traj["positions"][10:, 0, 2] = 0.1
    rec = label.label(traj, CFG)
    assert rec["object"] == "bowl" and rec["onset_step"] == 5
