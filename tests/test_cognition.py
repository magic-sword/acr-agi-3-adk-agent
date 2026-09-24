"""Behavioral contracts for changing goals, partial observation and external execution."""
from __future__ import annotations

import json
import unittest

from agent.cognition.engine import CognitiveTurn, new_memory
from agent.cognition.state import Fact, Hypothesis, Pending, Action, Predicate, Interpretation, PlanNode, Proposal
from agent.cognition.validation import evaluate, parse_json, validate_action, validate_proposal
from agent.cognition.workflow import CognitiveRuntime


def observation(step=0, **changes):
    value = dict(game_id="unknown", state="NOT_FINISHED", step=step, levels_completed=0,
                 available_actions=["ACTION1", "ACTION6"], grid=[[0, 0], [0, 0]], remaining_actions=20)
    value.update(changes)
    return value


def turn(memory=None, step=0, **changes):
    t = CognitiveTurn(memory or new_memory("unknown"), observation(step, **changes))
    t.observe()
    return t


def pending(t, *, effects=None, invariants=None, delay=1):
    return Pending(decision_id="d", observation_id="before", step=t.obs["step"] - 1,
                   action=Action(action="ACTION1"), deadline_step=t.obs["step"] - 1 + delay,
                   baseline_hash="old", effects=effects or [], invariants=invariants or [])


class CognitiveContracts(unittest.TestCase):
    def test_invisible_fact_is_remembered_but_not_current_evidence(self):
        m = new_memory("unknown")
        m.facts["gem"] = Fact(key="gem", value="2,3", evidence="old")
        t = turn(m)
        self.assertEqual(t.memory.facts["gem"].value, "2,3")
        self.assertIsNone(evaluate(Predicate(kind="fact", key="gem", value="2,3"), t.obs, t.memory))

    def test_delayed_effect_not_prematurely_refuted(self):
        t = turn(step=1)
        pred = Predicate(kind="cell", x=0, y=0, value=9)
        t.memory.pending = pending(t, effects=[pred], delay=3)
        t.verify()
        self.assertEqual(t.verification[-1]["result"], "unknown")
        self.assertEqual(len(t.memory.deferred), 1)
        self.assertIsNone(t.memory.pending)
        t.obs["step"] = 3
        t.obs["grid"][0][0] = 9
        t.verify()
        self.assertEqual(t.verification[-1]["result"], "supported")
        self.assertFalse(t.memory.deferred)

    def test_intervention_prevents_false_causal_attribution(self):
        t = turn(step=2)
        t.memory.pending = pending(t, effects=[Predicate(kind="cell", x=0, y=0, value=0)])
        t.memory.pending.intervening_actions = ["another_action"]
        t.verify()
        self.assertEqual(t.verification[-1]["result"], "unknown")

    def test_invariant_break_revises_affected_dependencies(self):
        t = turn(step=1)
        cell = Predicate(kind="cell", x=0, y=0, value=9)
        t.memory.plan = [PlanNode(id="a", subgoal="place", completion=[cell]),
                         PlanNode(id="b", subgoal="then move", depends_on=["a"], completion=[cell]),
                         PlanNode(id="c", subgoal="independent", completion=[cell])]
        t.memory.pending = pending(t, invariants=[cell])
        t.memory.pending.node_id = "a"
        t.verify()
        self.assertTrue(t.revise)
        self.assertEqual([n.status for n in t.memory.plan], ["invalid", "invalid", "todo"])

    def test_visual_goal_does_not_claim_environment_win(self):
        t = turn()
        t.memory.goal = "looks complete"
        t.verify(); t.update()
        self.assertNotEqual(t.memory.lifecycle, "DONE")
        t.obs["state"] = "WIN"
        t.update()
        self.assertEqual(t.commit()["status"], "done")

    def test_boundary_preserves_scoped_candidates_not_positions(self):
        m = new_memory("unknown")
        m.last_observation = {"step": 0, "observation_id": "old"}
        m.facts["position"] = Fact(key="position", value="0,0", evidence="old")
        m.hypotheses["h"] = Hypothesis(id="h", kind="control", claim="linked movement", scope="game", status="supported", evidence_refs=["old"])
        t = turn(m, step=1, levels_completed=1)
        self.assertFalse(t.memory.facts)
        self.assertEqual(t.memory.hypotheses["h"].status, "candidate")
        self.assertTrue(t.boundary)

    def test_unavailable_reset_and_unknown_actions_are_rejected(self):
        for data in ({"action": "RESET"}, {"action": "ACTION2"}, {"action": "ACTION6", "x": True, "y": 1}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_action(data, observation())
        with self.assertRaises(ValueError):
            validate_action({"action": "ACTION1"}, {"available_actions": []})

    def test_duplicate_and_nested_json_contracts(self):
        with self.assertRaises(ValueError):
            parse_json('{"x":1,"x":2}')
        self.assertEqual(parse_json('```json\n{"plan":{"nodes":[]}}\n```'), {"plan": {"nodes": []}})
        with self.assertRaises(ValueError):
            parse_json('preface {"action":"ACTION1"}')

    def test_stale_and_cyclic_plans_rejected(self):
        t = turn()
        p = Proposal(observation_id=t.obs["observation_id"], memory_revision=9, purpose="plan")
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_proposal(p, t.obs, t.memory)
        p.memory_revision = 0
        pred = Predicate(kind="cell", x=0, y=0, value=1)
        p.plan = [PlanNode(id="a", subgoal="a", depends_on=["b"], completion=[pred]),
                  PlanNode(id="b", subgoal="b", depends_on=["a"], completion=[pred])]
        with self.assertRaisesRegex(ValueError, "cyclic"):
            validate_proposal(p, t.obs, t.memory)

    def test_unobserved_preconditions_block_plan_execution(self):
        t = turn()
        t.memory.plan = [PlanNode(id="a", subgoal="move", action=Action(action="ACTION1"),
                                 preconditions=[Predicate(kind="fact", key="path_open", value=True)],
                                 completion=[Predicate(kind="state", value="WIN")])]
        self.assertIsNone(t.ready_node())

    def test_multi_step_plan_continues_only_after_completion(self):
        t = turn()
        pred = Predicate(kind="cell", x=0, y=0, value=1)
        p = Proposal(observation_id=t.obs["observation_id"], memory_revision=0, purpose="plan", plan=[
            PlanNode(id="a", subgoal="stage", action=Action(action="ACTION1"), completion=[pred], effects=[pred]),
            PlanNode(id="b", subgoal="finish", depends_on=["a"], preconditions=[pred],
                     action=Action(action="ACTION1"), completion=[Predicate(kind="state", value="WIN")],
                     effects=[Predicate(kind="state", value="WIN")])])
        t.accept(p); t.act(); t.commit()
        next_turn = turn(t.memory, step=1, grid=[[1, 0], [0, 0]])
        next_turn.verify(); next_turn.update()
        self.assertEqual(next_turn.ready_node().id, "b")
        self.assertEqual(next_turn.route(), "ACT")
        self.assertEqual(next_turn.memory.procedures[0]["status"], "candidate")

    def test_temporal_deadline_invalidates_plan(self):
        t = turn(step=4)
        t.memory.plan = [PlanNode(id="a", subgoal="sync", latest_step=3,
                                  completion=[Predicate(kind="state", value="WIN")])]
        t.update()
        self.assertEqual(t.memory.plan[0].status, "invalid")

    def test_goal_change_keeps_causal_claim_and_invalidates_plan(self):
        t = turn()
        t.memory.goal = "avoid yellow"
        t.memory.hypotheses["effect"] = Hypothesis(id="effect", kind="dynamics", claim="orange becomes yellow")
        t.memory.plan = [PlanNode(id="a", subgoal="avoid", completion=[Predicate(kind="state", value="WIN")])]
        t.apply_interpretation(Interpretation(observation_id=t.obs["observation_id"], memory_revision=0,
                                     goal="produce yellow", evidence_refs=[t.obs["observation_id"]]), "PLAN")
        self.assertEqual(t.memory.hypotheses["effect"].claim, "orange becomes yellow")
        self.assertEqual(t.memory.plan[0].status, "invalid")


class WorkflowContracts(unittest.TestCase):
    def setUp(self):
        self.runtime = CognitiveRuntime("unknown")

    def tearDown(self):
        self.runtime.close()

    def test_persistent_session_duplicate_and_static_new_observation(self):
        r = self.runtime
        first = r.decide(observation())
        self.assertEqual(r.decide(observation()), first)
        self.assertEqual(r.memory.revision, 1)
        second = r.decide(observation(step=1))
        self.assertNotEqual(first["decision_id"], second["decision_id"])
        self.assertEqual(r.memory.revision, 2)
        self.assertTrue(r.memory.history[-1]["verification"])

    def test_cached_action_cannot_bypass_new_budget_limit(self):
        self.runtime.decide(observation())
        result = self.runtime.decide(observation(remaining_actions=0))
        self.assertEqual(result["status"], "stopped")
        self.assertNotIn("action", result)
        self.assertFalse(self.runtime.memory.history[-1]["verification"])

    def test_budget_exit_emits_no_game_action(self):
        answer = self.runtime.decide(observation(remaining_actions=0))
        self.assertEqual(answer["status"], "stopped")
        self.assertNotIn("action", answer)

    def test_reset_is_bounded_and_learns_across_attempts(self):
        self.runtime.max_resets = 1
        a = self.runtime.decide(observation(state="GAME_OVER"))
        self.assertEqual(a["action"], "RESET")
        b = self.runtime.decide(observation(step=1, state="GAME_OVER"))
        self.assertEqual(b["status"], "stopped")
        self.assertEqual(self.runtime.memory.attempt, 1)

    def test_win_records_final_observation_without_model(self):
        self.runtime.decide(observation())
        result = self.runtime.decide(observation(step=1, state="WIN", remaining_actions=0))
        self.assertEqual(result["status"], "done")
        self.assertIsNone(self.runtime.memory.pending)

    def test_skipped_driver_step_stops_instead_of_resending(self):
        self.runtime.decide(observation())
        answer = self.runtime.decide(observation(step=3))
        self.assertEqual(answer["status"], "stopped")
        self.assertIn("missing_observation", answer["reason"])

    def test_game_memory_cannot_cross_into_another_game(self):
        with self.assertRaises(ValueError):
            t = turn()
            t.obs["game_id"] = "other"
            t.observe()

class RecoveryContracts(unittest.TestCase):
    def test_no_legal_action_is_a_controlled_stop(self):
        r = CognitiveRuntime("unknown")
        try:
            result = r.decide(observation(available_actions=[]))
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(r.memory.stop_reason, "no_legal_action")
        finally:
            r.close()

    def test_invalid_frame_stops_without_action(self):
        r = CognitiveRuntime("unknown")
        try:
            result = r.decide(observation(grid=[[99]]))
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(r.memory.stop_reason, "invalid_observation")
        finally:
            r.close()

    def test_start_is_not_repeated_when_reset_does_not_start(self):
        r = CognitiveRuntime("unknown")
        try:
            self.assertEqual(r.decide(observation(state="NOT_PLAYED"))["action"], "RESET")
            result = r.decide(observation(step=1, state="NOT_PLAYED"))
            self.assertEqual(result["reason"], "start_failed")
            self.assertNotIn("action", result)
        finally:
            r.close()

    def test_rejected_hypothesis_cannot_drive_plan(self):
        t = turn()
        t.memory.hypotheses["h"] = Hypothesis(id="h", kind="control", claim="move right", status="refuted")
        t.memory.plan = [PlanNode(id="a", subgoal="go", hypothesis_ids=["h"],
                                  action=Action(action="ACTION1"), completion=[Predicate(kind="state", value="WIN")])]
        self.assertIsNone(t.ready_node())

    def test_frame_change_completion_uses_action_baseline(self):
        t = turn()
        pred = Predicate(kind="frame_changed", value=True)
        t.memory.plan = [PlanNode(id="a", subgoal="change", action=Action(action="ACTION1"),
                                  completion=[pred], effects=[pred])]
        t.act(); t.commit()
        nxt = turn(t.memory, step=1, grid=[[1, 0], [0, 0]])
        nxt.verify(); nxt.update()
        self.assertEqual(nxt.memory.plan[0].status, "done")


if __name__ == "__main__":
    unittest.main()
