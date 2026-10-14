"""Attention readout and failure alarm (paper sections 2.2 and 2.3). numpy only; never sees a label.

Per policy call, from the readout head's attention to the agent-view image (16 x 16 patches), the tracked
candidate masks at that call and the task objects (candidate indices chosen by the VLM on the first frame):

  1. P = the `top_k` patches with the largest attention, weights A(p) = attention renormalised over the view
  2. each patch of P goes whole to the candidate covering most of it, if that candidate covers at least
     `coverage` of the patch; otherwise to no candidate (table, gaps)
  3. s_k = sum_{p in P_k} A(p) / sum_{p in P} A(p)                                           (eq. 1)
  4. verdict
       abstain     P holds less than `weak` of the map (attention not concentrated)
       on-target   the task objects' shares sum to >= `share` and to at least any other candidate's share
       off-target  one other candidate's share is >= `share` and above the task objects' sum
       abstain     otherwise (e.g. attention on the table)

Alarm: off-target calls are counted whichever other candidate they are on; an on-target call resets the
count, an abstained call keeps it. The alarm goes off at the call where the count first reaches `alarm_count`.
"""

import numpy as np

import common

ON, OFF, ABSTAIN = "on-target", "off-target", "abstain"


def readout(attention, coverage, task_objects, cfg=None):
    """One call. attention (16, 16), coverage (candidates, 256) from common.patch_coverage, task_objects
    [candidate index]. Returns (verdict, off-target candidate or None, shares (candidates,), top-k mass)."""
    cfg = cfg or common.CONFIG["detector"]
    h = np.asarray(attention, np.float32).reshape(-1)
    h = h / h.sum()
    top = np.argsort(-h)[: cfg["top_k"]]
    w = np.zeros_like(h)
    w[top] = h[top]
    mass = w.sum()
    w = w / mass
    coverage = np.asarray(coverage, np.float32)
    owner = np.where(coverage.max(0) >= cfg["coverage"], coverage.argmax(0), -1)
    shares = np.array([(w * (owner == k)).sum() for k in range(len(coverage))], np.float32)
    if mass < cfg["weak"]:
        return ABSTAIN, None, shares, float(mass)
    task = np.isin(np.arange(len(coverage)), task_objects)
    s_task = shares[task].sum()
    others = np.flatnonzero(~task)
    j = int(np.argmax(shares[others])) if len(others) else None
    s_other = shares[others[j]] if j is not None else 0.0
    if s_task >= cfg["share"] and s_task >= s_other:
        return ON, None, shares, float(mass)
    if s_other >= cfg["share"] and s_other > s_task:
        return OFF, int(others[j]), shares, float(mass)
    return ABSTAIN, None, shares, float(mass)


def alarm(verdicts, alarm_count=None):
    """(index of the call that raises the alarm or None, off-target count after every call)."""
    alarm_count = alarm_count or common.CONFIG["detector"]["alarm_count"]
    count, counts, fired = 0, [], None
    for i, verdict in enumerate(verdicts):
        if verdict == ON:
            count = 0
        elif verdict == OFF:
            count += 1
            if count == alarm_count and fired is None:
                fired = i
        counts.append(count)
    return fired, counts


def run(attention, masks, task_objects, cfg=None):
    """A whole rollout. attention (calls, 16, 16); masks (calls, candidates, H, W) tracked candidate masks.
    Returns per-call verdicts, off-target candidates, shares, top-k mass, the counts and the alarm call."""
    coverage = common.patch_coverage(masks)
    calls = [readout(a, c, task_objects, cfg) for a, c in zip(attention, coverage)]
    verdicts = [c[0] for c in calls]
    fired, counts = alarm(verdicts, (cfg or common.CONFIG["detector"])["alarm_count"])
    return {"verdicts": verdicts, "off_target": [c[1] for c in calls],
            "shares": np.array([c[2] for c in calls]), "top_k_mass": [c[3] for c in calls],
            "counts": counts, "alarm_call": fired}
