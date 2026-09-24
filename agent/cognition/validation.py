"""Strict proposal parsing, legal action grounding and three-valued predicates."""
from __future__ import annotations

import json
from .state import Action, Predicate, Proposal, Memory


def parse_json(text: str) -> dict:
    if len(text) > 64000:
        raise ValueError("model response exceeds limit")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def validate_action(raw: dict | Action, obs: dict, *, reset: bool = False) -> Action:
    value = raw.model_dump() if isinstance(raw, Action) else dict(raw)
    name = value.get("action")
    if type(name) is int:
        name = str(name)
    if not isinstance(name, str):
        raise ValueError("Invalid model action")
    name = name.strip().upper()
    if name.isdecimal():
        name = "RESET" if int(name) == 0 else f"ACTION{int(name)}"
    allowed = set(obs.get("available_actions", [])) - {"RESET"}
    if reset and obs.get("state") in ("NOT_PLAYED", "GAME_OVER"):
        allowed.add("RESET")
    if name not in allowed:
        raise ValueError(f"Invalid model action: {name}; allowed: {sorted(allowed)}")
    value["action"] = name
    action = Action.model_validate(value)
    if name == "ACTION6":
        if action.x is None or action.y is None:
            raise ValueError("ACTION6 requires integer x and y")
        if action.x >= obs.get("width", 64) or action.y >= obs.get("height", 64):
            raise ValueError("click lies outside observation")
    elif action.x is not None or action.y is not None:
        raise ValueError("coordinates are only valid for ACTION6")
    return action


def evaluate(p: Predicate, obs: dict, memory: Memory, baseline_hash: str = "") -> bool | None:
    if p.kind == "state":
        return obs.get("state") == p.value
    if p.kind == "levels_min":
        return obs.get("levels_completed", 0) >= p.value
    if p.kind == "frame_changed":
        if not baseline_hash or not obs.get("frame_hash"):
            return None
        return (baseline_hash != obs["frame_hash"]) == p.value
    if p.kind == "cell":
        grid = obs.get("grid")
        if grid is None or p.y >= len(grid) or p.x >= len(grid[p.y]):
            return None
        return grid[p.y][p.x] == p.value
    fact = memory.facts.get(p.key)
    if not fact or not fact.visible or fact.evidence != obs["observation_id"]:
        return None
    return fact.value == p.value


def all_true(predicates: list[Predicate], obs: dict, memory: Memory, baseline: str = "") -> bool:
    return all(evaluate(p, obs, memory, baseline) is True for p in predicates)


def validate_proposal(p: Proposal, obs: dict, memory: Memory) -> None:
    if p.observation_id != obs["observation_id"] or p.memory_revision != memory.revision:
        raise ValueError("stale proposal")
    known_evidence = set(memory.evidence_ids)
    if not set(p.evidence_refs) <= known_evidence:
        raise ValueError("unknown evidence reference")
    if not set(p.invalidated_hypotheses) <= set(memory.hypotheses):
        raise ValueError("unknown hypothesis reference")
    if p.invalidated_hypotheses and not p.evidence_refs:
        raise ValueError("model revision requires evidence")
    ids = {n.id for n in p.plan}
    if len(ids) != len(p.plan):
        raise ValueError("duplicate plan node")
    visited, visiting = set(), set()
    by_id = {n.id: n for n in p.plan}
    def visit(key):
        if key in visiting:
            raise ValueError("cyclic plan")
        if key in visited:
            return
        if key not in by_id:
            raise ValueError("unknown plan dependency")
        visiting.add(key)
        for dep in by_id[key].depends_on:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for n in p.plan:
        visit(n.id)
        if n.status != "todo" or n.started_hash or n.started_step is not None:
            raise ValueError("model cannot assert plan completion")
        if not set(n.hypothesis_ids) <= set(memory.hypotheses):
            raise ValueError("unknown plan hypothesis")
        if any(memory.hypotheses[hid].status in ("refuted", "suspended") for hid in n.hypothesis_ids):
            raise ValueError("plan depends on invalid hypothesis")
        if n.action:
            validate_action(n.action, obs)
        if n.earliest_step is not None and n.latest_step is not None and n.earliest_step > n.latest_step:
            raise ValueError("inverted temporal constraint")
    if p.status != "ok":
        return
    if p.purpose == "probe":
        if not p.experiment or not p.action:
            raise ValueError("probe requires action and discriminating experiment")
        predictions = [json.dumps([v.model_dump() for v in ps], sort_keys=True)
                       for ps in p.experiment.alternatives.values()]
        if any(not ps for ps in p.experiment.alternatives.values()) or len(set(predictions)) < 2:
            raise ValueError("experiment cannot distinguish alternatives")
    if p.purpose == "plan" and not p.plan:
        raise ValueError("planning requires a subgoal graph")
    if p.node_id is not None and p.node_id not in ids:
        raise ValueError("unknown selected plan node")
    if p.action:
        validate_action(p.action, obs)
