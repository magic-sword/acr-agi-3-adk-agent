from copy import deepcopy
import unittest
from unittest.mock import patch

from agent.cognition.focused_workflow import FocusedRuntime
from agent.cognition.rules import RuleLearner
from agent.cognition.skill_library import build_skills
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call as _call, token, skill, ack


def call(name, args, _n=[0]):
    # A real model returns a distinct ID per tool call; multi-round stages need that.
    _n[0] += 1
    response = _call(name, args)
    response['choices'][0]['message']['tool_calls'][0]['id'] = f'call-{_n[0]}'
    return response

MOVES = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}


def board(player, target=(9, 1)):
    """12x12: floor 3, wall 4 at x=6 for y<=8, orange 2x2 player, yellow 2x2 target."""
    g = [[3]*12 for _ in range(12)]
    for y in range(9):
        g[y][6] = 4
    for (x, y), c in ((target, 11), (player, 12)):
        for j in range(2):
            for i in range(2):
                g[y+j][x+i] = c
    return g


def trained():
    rules, pos = RuleLearner(), (1, 1)
    for a in ['ACTION4', 'ACTION4', 'ACTION4', 'ACTION4', 'ACTION2', 'ACTION3', 'ACTION1']:
        d = MOVES[a]
        nxt = (pos[0]+d[0], pos[1]+d[1])
        blocked = any(board(pos)[nxt[1]+j][nxt[0]+i] == 4 for j in range(2) for i in range(2))
        after = pos if blocked else nxt
        rules.learn(board(pos), board(after), a)
        pos = after
    return rules, pos


class RuleSkillTests(unittest.TestCase):
    def test_learns_walls_and_plans_around_them(self):
        rules, pos = trained()
        self.assertEqual(pos, (3, 1))
        self.assertEqual(rules.blocking_colors(rules.movers()), {4})
        self.assertEqual(rules.predict(board((4, 1)), 'ACTION4'), {next(iter(rules.movers())): None})
        path = rules.plan_path(board(pos), (9, 1, 10, 2))
        # The wall at x=6 is open only for y>=9: a shortest route goes down 8, right 6, up 8.
        self.assertEqual((path.count('ACTION2'), path.count('ACTION4'), path.count('ACTION1')), (8, 6, 8))
        x, y = pos
        for a in path:
            x, y = x+MOVES[a][0], y+MOVES[a][1]
            self.assertFalse(any(board(pos)[y+j][x+i] == 4 for j in range(2) for i in range(2)), (a, x, y))
        self.assertEqual((x, y), (9, 1))

    def test_self_target_is_rejected_and_untried_directions_are_assumed(self):
        rules = RuleLearner()
        rules.learn(board((1, 1)), board((1, 2)), 'ACTION2')  # only DOWN has been tried
        with self.assertRaises(ValueError):
            rules.plan_path(board((1, 2)), (1, 2, 2, 3))
        self.assertIsNone(rules.plan_path(board((1, 2)), (4, 2, 5, 3)))  # RIGHT unknown without the prior
        path = rules.plan_path(board((1, 2)), (4, 2, 5, 3), available=list(MOVES))
        self.assertEqual(path, ['ACTION4']*3)
        self.assertEqual(rules.assumed_deltas(rules.movers(), list(MOVES)),
                         {'ACTION1': (0, -1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)})

    def test_blocked_presses_trigger_review_even_without_a_probe_limit(self):
        r = FocusedRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program')
        self.addCleanup(r.close)
        r.rules, pos = trained()
        wall = (4, 1)  # RIGHT from here is blocked by the wall
        works = []

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            if c['work'] == 'understand':
                return call('submit_scene', dict(observation_id=c['observation_id'], observed='Block.',
                    goal_hypothesis='Go right.', targets=[dict(concept='block', appearance='orange', role_hypothesis='player',
                    relations='left of wall', candidate_refs=[], object_id=None)]))
            if c['work'] == 'backchain':
                return call('submit_subgoal', dict(observation_id=c['observation_id'], desired_state='Block right.',
                    target_query='wall', intent='achieve', question='', expected_observation='Moves.', rationale='r'))
            if c['work'] == 'candidates':
                return call('submit_candidates', dict(observation_id=c['observation_id'], goal_id=c['goal_id'], candidates=[
                    dict(id='go', verb='move_to', target_query='right', expected_effect='Moves right.', rationale='r',
                         candidate_refs=[], object_refs=[], source='proposed', skill_name=None)], missing_info=''))
            if c['work'] == 'ground':
                right = skill(); right['steps'][0]['options'][0]['action'] = {'action': 'RIGHT'}
                if [t['function']['name'] for t in payload.get('tools', [])] == ['load_skill']:
                    return call('load_skill', {'skill_name': 'move-orange-block'})
                return call('submit_plan_choice', dict(observation_id=c['observation_id'], candidate_id='go',
                    procedure=right, probe_action_limit=None, next='execute', reason='Push right.'))
            if c['work'] == 'execute_step':
                return token('1')
            if c['work'] == 'reconcile':
                self.assertEqual(c['review']['trigger'], 'no_effect_streak')
                return call('submit_reconciliation', dict(observation_id=c['observation_id'], goal_id=c['plan']['goal_id'],
                    assessment='unexpected', evidence='RIGHT changed nothing.', goal_status='unknown', causal_notes='',
                    next='backchain', reason='Blocked.', next_question='Another way?'))
            raise AssertionError(c['work'])

        def frame(step):
            o = obs(step, board(wall)); o['available_actions'] = list(MOVES); return o
        with patch.object(LocalVisionLlm, '_complete', respond):
            for step in range(4):
                r.decide(frame(step))
                if step < 3:
                    r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
        self.assertEqual(works.count('reconcile'), 1)
        # Review comes after the third press that affected nothing, before any further press.
        self.assertEqual(works[:works.index('reconcile')].count('execute_step'), 3)

    def test_click_rule_is_conditioned_not_overwritten(self):
        def frame(slider_x):
            g = [[3]*12 for _ in range(12)]
            g[10][1] = g[10][2] = 9           # blue button
            g[4][slider_x] = 11               # yellow slider
            return g
        rules = RuleLearner()
        rules.learn(frame(3), frame(4), 'ACTION6', (1, 10))   # slider moves right
        rules.learn(frame(4), frame(5), 'ACTION6', (1, 10))
        rules.learn(frame(9), frame(9), 'ACTION6', (1, 10))   # at the end: no effect
        button = rules.clicked(frame(3), (1, 10))['sig']
        self.assertTrue(rules.state_dependent(button))
        self.assertEqual(rules.predict_outcome(frame(4), 'ACTION6', (1, 10))['moves'],
                         {(11, ((0, 0),)): (1, 0)})           # seen context: predicted
        self.assertEqual(rules.predict_outcome(frame(9), 'ACTION6', (1, 10))['moves'], {})
        self.assertIsNone(rules.predict_outcome(frame(6), 'ACTION6', (1, 10)))  # unseen, conflicting: abstain
        self.assertEqual(rules.replay_consistency(), 1.0)

    def test_skill_options_offer_destinations_but_never_the_mover(self):
        r = FocusedRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program')
        self.addCleanup(r.close)
        r.rules, pos = trained()
        o = obs(0, board(pos)); o['available_actions'] = list(MOVES)
        r._receive(o)
        by_color = {o['color_ids'][0]: o['object_id'] for o in r.memory.object_memory['objects']}
        r.memory.selected_candidate = dict(object_refs=[by_color['orange']])   # the mover, as models often write
        r.memory.understanding = dict(observation_id=r.obs['observation_id'], targets=[
            dict(object_id=by_color['orange'], concept='block'), dict(object_id=by_color['yellow'], concept='goal')])
        offers = r._skill_options(2)
        self.assertEqual([v['target'] for v in offers.values()], [by_color['yellow']])
        self.assertEqual(list(offers), ['2'])
        self.assertIn('goal', offers['2']['meaning'])

    def test_skill_carries_measured_table_and_exact_tools(self):
        rules, _ = trained()
        skills = {s.name: s for s in build_skills(rules)}
        move = skills['move-orange-block']
        self.assertEqual(move.frontmatter.metadata['adk_additional_tools'], ['plan_path', 'predict_effect'])
        self.assertIn('RIGHT: moves by (1, 0); 3/4 presses moved something', move.instructions)
        self.assertIn('stopped by colours: very dark gray', move.instructions)

    def test_fast_mode_hands_movement_to_program_skill_and_mismatch_is_reviewed(self):
        r = FocusedRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program')
        self.addCleanup(r.close)
        r.rules, pos = trained()
        state = dict(target=None)
        works = []

        def target_id():
            return next(o['object_id'] for o in r.memory.object_memory['objects'] if o['color_ids'] == ['yellow'])

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            if c['work'] == 'understand':
                state['target'] = target_id()
                return call('submit_scene', dict(observation_id=c['observation_id'], observed='Block and goal.',
                    goal_hypothesis='Reach the yellow square.', targets=[dict(concept='goal', appearance='yellow square',
                    role_hypothesis='goal', relations='right of wall', candidate_refs=[], object_id=state['target'])]))
            if c['work'] == 'backchain':
                self.assertEqual(c['measured_rules'][0]['name'], 'move-orange-block')
                self.assertIn('load_skill', [t['function']['name'] for t in payload['tools']])
                return call('submit_subgoal', dict(observation_id=c['observation_id'], desired_state='Block on goal.',
                    target_query='yellow square', intent='achieve', question='', expected_observation='Block on goal.',
                    rationale='Known movement.'))
            if c['work'] == 'candidates':
                return call('submit_candidates', dict(observation_id=c['observation_id'], goal_id=c['goal_id'], candidates=[
                    dict(id='go', verb='move_to', target_query='yellow square', expected_effect='Block on goal.',
                         rationale='r', candidate_refs=[], object_refs=[state['target']], source='proposed', skill_name=None)],
                    missing_info=''))
            if c['work'] == 'ground':
                return call('submit_plan_choice', dict(observation_id=c['observation_id'], candidate_id='go',
                    procedure=skill(), probe_action_limit=None, next='execute', reason='Walk to the goal.'))
            if c['work'] == 'execute_step':
                return token(next(k for k, v in c['choices'].items() if v.get('kind') == 'skill'))
            if c['work'] == 'reconcile':
                self.assertTrue(c['review']['trigger'].startswith('skill_'))
                return call('submit_reconciliation', dict(observation_id=c['observation_id'], goal_id=c['plan']['goal_id'],
                    assessment='unexpected', evidence='The block did not move.', goal_status='unknown', causal_notes='',
                    next='backchain', reason='Prediction failed.', next_question='What stops it?'))
            raise AssertionError(c['work'])

        def frame(step, at):
            o = obs(step, board(at)); o['available_actions'] = list(MOVES); return o

        route = r.rules.plan_path(board(pos), (9, 1, 10, 2))
        with patch.object(LocalVisionLlm, '_complete', respond):
            first = r.decide(frame(0, pos))
            self.assertEqual(first['action'], route[0], r.errors)
            ack(r)
            moved = (pos[0]+MOVES[route[0]][0], pos[1]+MOVES[route[0]][1])
            second = r.decide(frame(1, moved))
            # The program presses the next step itself: no second fast-mode model call.
            self.assertEqual(works.count('execute_step'), 1)
            self.assertEqual(second['action'], r.rules.plan_path(board(moved), (9, 1, 10, 2), available=list(MOVES))[0])
            ack(r)
            r.decide(frame(2, moved))  # predicted a move, measured none: counterexample and review
        stats = r.memory.skill_stats['move-orange-block']
        self.assertEqual(stats['predictions'], 2)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(len(stats['counterexamples']), 1)
        self.assertEqual(stats['version'], 2)
        # The mismatch stopped the first run and forced review before the skill was chosen again.
        self.assertEqual(works.count('reconcile'), 1)
        self.assertLess(works.index('reconcile'), len(works) - 1 - works[::-1].index('execute_step'))
        self.assertTrue(r.memory.macro['invocation_id'].endswith(':attempt:2'))
        skill_text = build_skills(r.rules, r.memory.skill_stats)[0]
        self.assertIn('counterexample at step 2', skill_text.instructions)
        self.assertIn('Predictions 1/2 correct', skill_text.description)

if __name__ == '__main__':
    unittest.main()
