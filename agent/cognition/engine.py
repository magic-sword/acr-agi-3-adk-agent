"""Pure transition logic, independent of ADK transport and of any game's rules."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
import uuid

from agent.observation import VISUAL_PAYLOAD_KEYS

from .state import Action, Memory, Pending, Perception, Proposal
from .validation import all_true, evaluate, validate_action, validate_proposal


class CognitiveTurn:
    """An isolated transaction; only COMMIT publishes its memory to the runtime."""

    def __init__(self, memory: Memory, observation: dict, *, max_resets: int = 2,
                 max_calls: int = 3, deadline: float | None = None):
        self.started_at = time.monotonic()
        self.memory = memory.model_copy(deep=True)
        self.obs = deepcopy(observation)
        self.max_resets = max_resets
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.trace: list[str] = []
        self.errors: list[str] = []
        self.verification: list[dict] = []
        self.proposal: Proposal | None = None
        self.selected: Action | None = None
        self.node_id: str | None = None
        self.effects = []
        self.invariants = []
        self.delay = 1
        self.experiment = None
        self.revise = False
        self.boundary = False
        self.duplicate = False
        self.result: dict | None = None

    def enter(self, state: str):
        self.trace.append(state)
        if len(self.trace) > 24:
            self.stop("internal_transition_budget")
            raise ValueError("internal transition budget exhausted")

    def stop(self, reason: str):
        self.memory.lifecycle = "DONE" if self.obs.get("state") == "WIN" else "STOPPED"
        self.memory.stop_reason = reason
        self.selected = None

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else max(0.0, self.deadline - time.monotonic())

    def observe(self):
        self.enter("OBSERVE")
        o, m = self.obs, self.memory
        if o.get("game_id") != m.game_id:
            raise ValueError("observation belongs to a different game")
        if type(o.get("step")) is not int or o["step"] < 0:
            raise ValueError("invalid driver step")
        if o.get("state") not in {"NOT_PLAYED", "NOT_FINISHED", "IN_PROGRESS", "GAME_OVER", "WIN"}:
            raise ValueError("invalid game state")
        grid = o.get("grid")
        if grid is not None:
            if (not grid or len(grid) > 64 or not isinstance(grid[0], list) or not grid[0]
                    or len(grid[0]) > 64 or any(len(row) != len(grid[0]) for row in grid)
                    or any(type(c) is not int or not 0 <= c <= 15 for row in grid for c in row)):
                raise ValueError("invalid color-ID grid")
            o["width"], o["height"] = len(grid[0]), len(grid)
        if not o.get("frame_hash"):
            raw = json.dumps(grid, separators=(",", ":")) if grid is not None else o.get("image_png_base64", "")
            o["frame_hash"] = hashlib.sha256(raw.encode()).hexdigest() if raw else ""
        identity = {k: o.get(k) for k in ("step", "frame_hash", "state", "levels_completed", "available_actions")}
        o["observation_id"] = f'{m.run_id}:{o["step"]}:' + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        previous = m.last_observation
        if previous and o["step"] <= previous["step"]:
            if o["observation_id"] == previous["observation_id"]:
                if m.last_result.get("status") == "action" and (o.get("remaining_actions", 1) <= 0 or self.time_left() <= 0):
                    self.stop("budget_exhausted")
                    return
                self.duplicate = True
                self.result = deepcopy(m.last_result)
                return
            self.stop("out_of_order_observation")
            return
        if m.lifecycle in ("DONE", "STOPPED"):
            return
        if previous and m.pending and o["step"] != m.pending.step + 1:
            self.stop("unacknowledged_action_or_missing_observation")
            return
        m.lifecycle = "ACTIVE"
        reset_ack = bool(m.pending and m.pending.action.action == "RESET")
        level = o.get("levels_completed", 0)
        self.boundary = bool(previous and (level != m.level or reset_ack or (o.get("full_reset", False) and not previous.get("full_reset", False))))
        if self.boundary:
            # Do not judge effects on the old board against a different board.
            for pending in ([m.pending] if m.pending else []) + m.deferred:
                self.verification.append({"decision_id": pending.decision_id, "result": "boundary",
                                          "evidence": o["observation_id"]})
            if level > m.level:
                self.consolidate("level_transition")
            m.pending = None
            m.deferred = []
            m.facts = {}
            m.plan = []
            m.goal = ""
            m.unknowns = []
            for h in m.hypotheses.values():
                h.status = "suspended" if h.scope == "level" else "candidate"
            m.model_revision += 1
            if reset_ack:
                m.attempt += 1
        m.level = level
        m.evidence_ids = (m.evidence_ids + [o["observation_id"]])[-128:]
        for fact in m.facts.values():
            fact.visible = False
        old_grid = previous.get("grid")
        self.obs["changed_cells"] = []
        if not self.boundary and grid and old_grid and len(grid) == len(old_grid) and len(grid[0]) == len(old_grid[0]):
            changes = [{"x": x, "y": y, "before": old_grid[y][x], "after": c}
                       for y, row in enumerate(grid) for x, c in enumerate(row) if c != old_grid[y][x]]
            self.obs["changed_cell_count"] = len(changes)
            self.obs["changed_cells"] = changes[:96]

    def apply_perception(self, p: Perception):
        m, o = self.memory, self.obs
        if p.observation_id != o["observation_id"] or p.memory_revision != m.revision:
            raise ValueError("stale perception")
        if any(f.evidence != o["observation_id"] or not f.visible for f in p.facts):
            raise ValueError("perception facts must be current visible evidence")
        for h in p.hypotheses:
            if not set(h.evidence_refs) <= set(m.evidence_ids):
                raise ValueError("unknown hypothesis evidence")
            if h.status != "candidate" and not h.evidence_refs:
                raise ValueError("hypothesis status requires evidence")
        if not set(p.change_evidence) <= set(m.evidence_ids):
            raise ValueError("unknown change evidence")
        if p.change in ("mode", "dynamics", "goal") and not p.change_evidence:
            raise ValueError("revision requires evidence")
        for f in p.facts:
            m.facts[f.key] = f
        # Recently visible facts replace least-recent hidden facts only when bounded memory fills.
        order = {eid: index for index, eid in enumerate(m.evidence_ids)}
        m.facts = dict(sorted(m.facts.items(), key=lambda kv: order.get(kv[1].evidence, -1))[-64:])
        affected = set()
        for h in p.hypotheses:
            old = m.hypotheses.get(h.id)
            if old and (old.claim != h.claim or old.status != h.status):
                affected.add(h.id)
            m.hypotheses[h.id] = h
        m.hypotheses = dict(list(m.hypotheses.items())[-32:])
        self.invalidate(affected)
        if p.goal and p.goal != m.goal:
            if m.goal:
                self.revise = True
                self.invalidate_all()
            m.goal = p.goal
        m.unknowns = p.unknowns
        if p.change in ("mode", "dynamics", "goal"):
            self.revise = True
            self.invalidate_all()
        if affected or self.revise:
            m.model_revision += 1

    def verify(self):
        self.enter("VERIFY")
        if self.duplicate:
            return
        m, o = self.memory, self.obs
        if (m.last_observation and o["step"] <= m.last_observation["step"]) or m.stop_reason in (
                "invalid_observation", "out_of_order_observation", "unacknowledged_action_or_missing_observation"):
            return
        candidates = ([m.pending] if m.pending else []) + m.deferred
        m.pending, m.deferred = None, []
        for pending in candidates:
            effects = [evaluate(p, o, m, pending.baseline_hash) for p in pending.effects]
            invariants = [evaluate(p, o, m, pending.baseline_hash) for p in pending.invariants]
            late = o["step"] >= pending.deadline_step
            # A later intervention prevents attribution, even when a visible effect matches.
            confounded = bool(pending.intervening_actions)
            if False in invariants:
                outcome = "contradicted"
            elif effects and all(v is True for v in effects):
                outcome = "unknown" if confounded else "supported"
            elif late and False in effects:
                outcome = "unknown" if confounded else "contradicted"
            else:
                outcome = "unknown"
            item = {"decision_id": pending.decision_id, "result": outcome,
                    "effects": effects, "invariants": invariants,
                    "evidence": o["observation_id"], "confounded": confounded}
            if pending.experiment:
                item["alternatives"] = {
                    label: [evaluate(v, o, m, pending.baseline_hash) for v in predicates]
                    for label, predicates in pending.experiment.alternatives.items()}
            self.verification.append(item)
            if outcome == "unknown" and not late:
                m.deferred.append(pending)
            if outcome == "contradicted":
                self.revise = True
                self.invalidate_node(pending.node_id)
        m.deferred = m.deferred[-8:]

    def update(self):
        self.enter("UPDATE")
        if self.duplicate:
            return
        m, o = self.memory, self.obs
        if o["state"] == "WIN":
            self.consolidate("environment_win")
            self.stop("environment_win")
            return
        if o.get("evaluation_stop_reason") == "level_limit":
            self.stop("level_limit")
            return
        if o.get("remaining_actions", 1) <= 0 or self.time_left() <= 0:
            self.stop("budget_exhausted")
            return
        if m.lifecycle in ("DONE", "STOPPED"):
            return
        for node in m.plan:
            if node.status not in ("done", "invalid"):
                if all_true(node.completion, o, m, node.started_hash):
                    node.status = "done"
                    self.consolidate("subgoal_verified", node.id)
                elif node.latest_step is not None and o["step"] > node.latest_step:
                    self.invalidate_node(node.id)
                    self.revise = True
                elif any(evaluate(v, o, m) is False for v in node.invariants):
                    self.invalidate_node(node.id)
                    self.revise = True

    def invalidate_all(self):
        for n in self.memory.plan:
            if n.status != "done":
                n.status = "invalid"

    def invalidate(self, hypotheses: set[str]):
        for n in self.memory.plan:
            if hypotheses.intersection(n.hypothesis_ids):
                self.invalidate_node(n.id)

    def invalidate_node(self, node_id: str | None):
        if node_id is None:
            return
        affected = {node_id}
        for _ in self.memory.plan:
            for n in self.memory.plan:
                if n.id in affected or affected.intersection(n.depends_on):
                    n.status = "invalid"
                    affected.add(n.id)

    def ready_node(self):
        m, o = self.memory, self.obs
        done = {n.id for n in m.plan if n.status == "done"}
        for n in m.plan:
            if (n.status == "todo" and set(n.depends_on) <= done and n.action
                    and (n.earliest_step is None or o["step"] >= n.earliest_step)
                    and (n.latest_step is None or o["step"] <= n.latest_step)
                    and all(hid in m.hypotheses and m.hypotheses[hid].status not in ("refuted", "suspended")
                            for hid in n.hypothesis_ids)
                    and all_true(n.preconditions + n.invariants, o, m)):
                return n
        return None

    def route(self) -> str:
        if self.duplicate or self.memory.lifecycle in ("DONE", "STOPPED"):
            return "COMMIT"
        if self.obs["state"] in ("NOT_PLAYED", "GAME_OVER"):
            return "RECOVER"
        if self.revise:
            return "REVISE"
        if self.ready_node() is not None:
            return "ACT"
        return "PROBE" if not self.memory.goal or self.memory.unknowns else "PLAN"

    def revise_model(self):
        self.enter("REVISE")
        # Preserve completed/independent nodes for the planner; no ungrounded reset.
        self.memory.model_revision += 1

    def accept(self, p: Proposal):
        validate_proposal(p, self.obs, self.memory)
        if p.status != "ok":
            raise ValueError(f"model could not propose a grounded action: {p.status}")
        for hid in p.invalidated_hypotheses:
            self.memory.hypotheses[hid].status = "suspended"
        self.invalidate(set(p.invalidated_hypotheses))
        if p.plan:
            self.memory.plan = p.plan
        self.proposal = p

    def act(self):
        self.enter("ACT")
        if self.memory.lifecycle in ("DONE", "STOPPED"):
            return
        p = self.proposal
        if p and p.purpose == "probe":
            self.selected = validate_action(p.action, self.obs)
            self.effects, self.invariants = p.effects, p.invariants
            self.delay, self.experiment = p.delay_steps, p.experiment
        else:
            node = self.ready_node()
            if not node:
                raise ValueError("no plan action with satisfied dependencies and preconditions")
            if p and p.node_id and p.node_id != node.id:
                # Explicit selection is allowed only if it is independently ready.
                other = next(n for n in self.memory.plan if n.id == p.node_id)
                done = {n.id for n in self.memory.plan if n.status == "done"}
                if (other.status != "todo" or not set(other.depends_on) <= done
                        or not all_true(other.preconditions + other.invariants, self.obs, self.memory)
                        or (other.earliest_step is not None and self.obs["step"] < other.earliest_step)
                        or (other.latest_step is not None and self.obs["step"] > other.latest_step)):
                    raise ValueError("selected plan node is not ready")
                node = other
            if not node.effects:
                raise ValueError("plan action requires predicted effects")
            self.selected = validate_action(node.action, self.obs)
            self.effects, self.invariants = node.effects, node.invariants
            self.delay, self.node_id = node.delay_steps, node.id
            node.status = "active"
            node.started_hash = self.obs.get("frame_hash", "")
            node.started_step = self.obs["step"]
        if not all_true(self.invariants, self.obs, self.memory):
            raise ValueError("action invariants not currently verified")
        if self.time_left() <= 0:
            self.stop("budget_exhausted")

    def recover(self):
        self.enter("RECOVER")
        m, o = self.memory, self.obs
        if m.lifecycle in ("DONE", "STOPPED"):
            return
        if (o["state"] == "NOT_PLAYED" and m.attempt == 0) or (o["state"] == "GAME_OVER" and m.resets < self.max_resets):
            self.selected = validate_action({"action": "RESET", "reason": "start" if o["state"] == "NOT_PLAYED" else "bounded retry"}, o, reset=True)
            self.effects = []
        else:
            self.stop("reset_budget" if o["state"] == "GAME_OVER" else "start_failed" if o["state"] == "NOT_PLAYED" else "proposal_unavailable")

    def consolidate(self, cause: str, node_id: str | None = None):
        self.enter("CONSOLIDATE")
        m = self.memory
        # A witnessed template, never an unconditionally reusable rule.
        steps = [h for h in m.history if h.get("node_id") == node_id] if node_id else m.history[-8:]
        if steps:
            m.procedures.append({"status": "candidate", "scope": "game", "cause": cause,
                                 "source_level": m.level, "node_id": node_id,
                                 "goal": m.goal, "applicability": "revalidate controls and preconditions",
                                 "evidence": self.obs["observation_id"],
                                 "steps": [{k: h[k] for k in ("action", "effects", "invariants") if k in h} for h in steps[-8:]]})
            m.procedures = m.procedures[-8:]

    def commit(self) -> dict:
        self.enter("COMMIT")
        if self.duplicate:
            return self.result
        m, o = self.memory, self.obs
        if self.selected is not None and (self.time_left() <= 0 or o.get("remaining_actions", 1) <= 0):
            self.stop("budget_exhausted")
        m.revision += 1
        m.model_calls += self.calls
        if self.selected is not None:
            decision_id = f"{m.run_id}:{m.revision}"
            pending = Pending(decision_id=decision_id, observation_id=o["observation_id"], step=o["step"],
                              action=self.selected, effects=self.effects, invariants=self.invariants,
                              deadline_step=o["step"] + self.delay, baseline_hash=o.get("frame_hash", ""),
                              node_id=self.node_id, experiment=self.experiment)
            for delayed in m.deferred:
                delayed.intervening_actions.append(decision_id)
            m.pending = pending
            m.lifecycle = "AWAIT_FRAME"
            if self.selected.action == "RESET" and o["state"] == "GAME_OVER":
                m.resets += 1
            self.result = {"status": "action", **self.selected.model_dump(exclude_none=True), "decision_id": decision_id}
        else:
            if m.lifecycle not in ("DONE", "STOPPED"):
                self.stop("no_action")
            self.result = {"status": "done" if m.lifecycle == "DONE" else "stopped", "reason": m.stop_reason}
        entry = {"observation_id": o.get("observation_id"), "step": o.get("step"),
                 "trace": self.trace[:], "verification": self.verification,
                 "node_id": self.node_id, "action": self.result,
                 "effects": [p.model_dump() for p in self.effects],
                 "invariants": [p.model_dump() for p in self.invariants],
                 "errors": self.errors, "model_calls": self.calls,
                 "decision_seconds": time.monotonic() - self.started_at,
                 "frame_hash": o.get("frame_hash"), "changed_cell_count": o.get("changed_cell_count"),
                 "levels_completed": o.get("levels_completed"), "game_state": o.get("state")}
        m.history = (m.history + [entry])[-64:]
        m.last_observation = {k: v for k, v in o.items() if k not in VISUAL_PAYLOAD_KEYS | {"recent_actions"}}
        m.last_result = deepcopy(self.result)
        return self.result


def new_memory(game_id: str) -> Memory:
    return Memory(run_id=uuid.uuid4().hex, game_id=game_id)
