"""Plans, noise, patch coverage, VLM reply parsing, task-object regions and failure types."""

import os

import numpy as np
import pytest

import common
import evaluate
import task_objects


def test_noise_is_fixed_by_seed_task_init_call():
    a = common.noise(7, "turn_on_the_stove", 3, 0)
    assert a.shape == (10, 32) and a.dtype == np.float32
    assert np.array_equal(a, common.noise(7, "turn_on_the_stove", 3, 0))
    assert not np.array_equal(a, common.noise(7, "turn_on_the_stove", 3, 1))
    assert not np.array_equal(a, common.noise(8, "turn_on_the_stove", 3, 0))


def test_patch_coverage():
    m = np.zeros((224, 224), bool)
    m[:14, :7] = True
    c = common.patch_coverage(m)
    assert c.shape == (256,) and c[0] == 0.5 and c[1:].sum() == 0


LISTED = {"name": "t", "seed": 7, "sentences": {"source": "listed", "tasks": {
    "turn_on_the_stove": {"init": 12, "sentences": [{"id": "a", "text": "switch the stove on"},
                                                    {"id": "b", "text": "turn the cooker on"}]}}}}


def test_listed_plan_adds_the_originals_it_needs():
    plan = common.make_plan(LISTED)
    originals = [p for p in plan if p["sentence"] == common.ORIGINAL]
    assert {(p["task"], p["init"]) for p in originals} == {(t, 0) for t in common.TASKS} | {("turn_on_the_stove", 12)}
    assert [p["sentence"] for p in plan if p["sentence"] != common.ORIGINAL] == ["a", "b"]


def test_only_keeps_matching_sentences():
    plan = common.make_plan(LISTED, only=["turn_on_the_stove/b/*"])
    assert [p["sentence"] for p in plan if p["sentence"] != common.ORIGINAL] == ["b"]


@pytest.mark.skipif(not os.environ.get("LIBERO_PARA_ROOT"), reason="needs LIBERO-Para")
def test_libero_para_plan_round_robin_over_inits():
    plan = common.make_plan({"name": "p", "seed": 7, "sentences": {"source": "libero_para", "inits": [1, 49]}})
    sentences = [p for p in plan if p["sentence"] != common.ORIGINAL]
    originals = [p for p in plan if p["sentence"] == common.ORIGINAL]
    assert len(sentences) == 4092 and len(originals) == 500
    task = [p for p in sentences if p["task"] == common.TASKS[0]]
    assert [p["init"] for p in task[:3]] == [1, 2, 3] and task[49]["init"] == 1
    assert common.make_plan({"name": "p", "seed": 7, "sentences": {"source": "libero_para", "inits": [1, 49]}}) == plan


def test_parse_reply():
    assert task_objects.parse('```json\n{"objects": "bowl, stove", "contact": [4, 2], "unclear": false}\n```', 5)[1:] == ("ok", [1, 3])
    assert task_objects.parse('{"contact": [9], "unclear": false}', 5)[1] == "invalid"
    assert task_objects.parse('{"contact": [1], "unclear": true}', 5)[1] == "unclear"
    assert task_objects.parse("no json", 5)[1] == "parse_error"


def test_region_adds_parts_and_wholes():
    big = np.zeros((10, 10), bool)
    big[:, :6] = True
    part = np.zeros((10, 10), bool)
    part[2:4, 2:4] = True             # lies inside the chosen candidate
    other = np.zeros((10, 10), bool)
    other[:, 7:] = True
    assert task_objects.region(np.array([big, part, other]), [0], 0.5) == [0, 1]
    assert task_objects.region(np.array([big, part, other]), [1], 0.5) == [0, 1]   # covers most of the chosen part
    assert task_objects.region(np.array([big, part, other]), [], 0.5) == []


@pytest.mark.skipif(not os.environ.get("LIBERO_PARA_ROOT"), reason="needs LIBERO-Para")
def test_failure_types_against_the_scene_original():
    names = ["akita_black_bowl_1", "cream_cheese_1", "flat_stove_1", "plate_1", "wine_bottle_1", "wine_rack_1",
             "wooden_cabinet_1"]
    base = {"task": "put_the_bowl_on_the_stove", "init": 3, "names": names, "onset": 20}
    rows = [{**base, "original": True, "success": True, "handled": 0},
            {**base, "original": False, "success": False, "handled": 1},     # cream cheese: not a task object
            {**base, "original": False, "success": False, "handled": 2},     # the stove: destination
            {**base, "original": False, "success": False, "handled": 0},
            {**base, "original": False, "success": False, "handled": None},
            {**base, "original": False, "success": True, "handled": 0},
            {**base, "init": 4, "original": True, "success": False, "handled": 0},
            {**base, "init": 4, "original": False, "success": False, "handled": 1}]
    types = [r["failure"] for r in evaluate.classify(rows)]
    assert types == [None, evaluate.TYPES[0], evaluate.TYPES[1], evaluate.TYPES[2], evaluate.TYPES[3], None, None,
                     evaluate.TYPES[4]]
