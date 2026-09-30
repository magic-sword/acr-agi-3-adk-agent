import json
import unittest
from unittest.mock import patch

from agent.cognition.explorer import Explorer
from agent.cognition.goal_planner import Planner
from agent.cognition.goal_runtime import GoalRuntime
from agent.cognition.predicates import describe, holds
from agent.cognition.rules import RuleLearner, components
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, token
from test_rule_skills import MOVES, board, trained, call


def as_json(args):
    """Hypotheses arrive as JSON-schema output, not a tool call (V8)."""
    return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(args)}}]}


def ROLES(c):
    return [dict(id=c['objects'][0]['id'], role='background_or_other')]


def cells_of(grid, color):
    return {(x, y) for y, row in enumerate(grid) for x, v in enumerate(row) if v == color}


def slider(x):
    """Click the blue button: the yellow slider moves right by 2; a red mark stands at column 9."""
    g = [[3]*12 for _ in range(12)]
    g[10][1] = g[10][2] = 9
    g[4][x] = g[4][x+1] = 11
    g[1][9] = 8
    return g


class PredicateTests(unittest.TestCase):
    def test_relations(self):
        view = {'a': ({(1, 1), (2, 1)}, {'red'}), 'b': ({(0, 0), (3, 0), (0, 3), (3, 3)}, {'blue'}),
                'c': ({(3, 1)}, {'red'}), 'd': ({(2, 5), (3, 5)}, {'green'})}
        self.assertTrue(holds(dict(relation='inside', a='a', b='b'), view))
        self.assertTrue(holds(dict(relation='adjacent', a='a', b='c'), view))
        self.assertTrue(holds(dict(relation='same_color', a='a', b='c'), view))
        self.assertTrue(holds(dict(relation='same_column', a='a', b='d'), view))
        self.assertFalse(holds(dict(relation='same_row', a='a', b='d'), view))
        self.assertTrue(holds(dict(relation='same_shape', a='a', b='d'), view))
        self.assertTrue(holds(dict(relation='color_is', a='d', color='green'), view))
        self.assertTrue(holds(dict(relation='gone', a='zz'), view))


class ExplorerTests(unittest.TestCase):
    def test_tries_small_objects_first_then_walks_to_the_nearest_untested_state(self):
        g = [[3]*6 for _ in range(6)]
        g[0][0] = 9                       # a one-cell button
        for y in range(2, 6):
            g[y][2] = g[y][3] = 5         # a larger bar
        e = Explorer()
        a = e.observe(g, ['ACTION6'], set())
        akey, action, how = e.choose(a)
        self.assertEqual((action['x'], action['y'], how), (0, 0, 'untested_here'))
        # The first click changes the frame to state b, where nothing is left untested.
        e.sent(a, akey)
        g2 = [row[:] for row in g]; g2[0][0] = 8
        b = e.observe(g2, ['ACTION1'], set())
        self.assertEqual(e.edges[a][akey], b)
        # ACTION1 was already tried here and led back to a, which still has the bar untested.
        e.tested[b].add(('press', 'ACTION1'))
        e.edges[b][('press', 'ACTION1')] = a
        akey2, _, how2 = e.choose(b)
        self.assertEqual((akey2, how2), (('press', 'ACTION1'), 'walk_to_frontier'))

    def test_dead_end_keeps_moving_with_the_least_used_action(self):
        g = [[3]*6 for _ in range(6)]
        e = Explorer()
        a = e.observe(g, ['ACTION1', 'ACTION2'], set())
        g2 = [row[:] for row in g]; g2[0][0] = 8
        b = e.observe(g2, ['ACTION1', 'ACTION2'], set())
        for k in list(e.actions[a]):
            e.tested[a].add(k); e.edges[a][k] = b
        for k in list(e.actions[b]):
            e.tested[b].add(k); e.edges[b][k] = a
        e.uses[(a, ('press', 'ACTION1'))] = 3
        akey, _, how = e.choose(a)
        self.assertEqual((akey, how), (('press', 'ACTION2'), 'wander'))

    def test_hypothesis_objects_are_tried_before_more_salient_ones(self):
        g = [[3]*6 for _ in range(6)]
        g[0][0] = 9
        for y in range(2, 6):
            g[y][2] = g[y][3] = 5
        e = Explorer()
        a = e.observe(g, ['ACTION6'], set())
        bar = {(x, y) for y in range(2, 6) for x in (2, 3)}
        _, action, _ = e.choose(a, frozenset(bar))
        self.assertIn((action['x'], action['y']), bar)


class PlannerTests(unittest.TestCase):
    def test_movement_plan_goes_around_the_wall(self):
        rules, pos = trained()
        grid = board(pos)
        objects = {'player': cells_of(grid, 12), 'goal': cells_of(grid, 11)}
        plan = Planner(rules, grid, objects, list(MOVES)).plan([dict(relation='inside', a='player', b='goal')])
        self.assertEqual(len(plan), 22)  # down 8, right 6, up 8: the wall is open only at the bottom
        self.assertEqual(sorted(a for a, _ in plan).count('ACTION4'), 6)

    def test_click_plan_counts_repeats_for_a_remote_effect(self):
        rules = RuleLearner()
        rules.learn(slider(1), slider(3), 'ACTION6', (1, 10))
        grid = slider(3)
        objects = {'slider': cells_of(grid, 11), 'mark': cells_of(grid, 8)}
        plan = Planner(rules, grid, objects, ['ACTION6']).plan([dict(relation='same_column', a='slider', b='mark')])
        # (9 - 4) / 2 rounded up: three clicks put the two-cell slider under column 9.
        self.assertEqual([a for a, _ in plan], ['ACTION6'] * 3)

    def test_reachable_relations_list_what_the_rules_can_make_true(self):
        rules = RuleLearner()
        rules.learn(slider(1), slider(3), 'ACTION6', (1, 10))
        grid = slider(3)
        objects = {'slider': cells_of(grid, 11), 'mark': cells_of(grid, 8), 'button': cells_of(grid, 9)}
        found = {describe(a): n for a, n in Planner(rules, grid, objects, ['ACTION6']).reachable()}
        self.assertEqual(found['same_column(slider, mark)'], 3)
        self.assertNotIn('same_row(slider, mark)', found)          # clicks move the slider sideways only
        self.assertNotIn('same_column(mark, slider)', found)       # the mark does not move
        self.assertTrue(all(k.split('(')[1].startswith('slider') for k in found))

    def test_same_looking_buttons_with_opposite_effects_are_kept_apart(self):
        def twins(x):
            # Two identical blue buttons: the left one moves the slider left, the right one right.
            g = [[3]*12 for _ in range(12)]
            g[10][1] = g[10][2] = 9
            g[10][8] = g[10][9] = 9
            g[4][x] = g[4][x+1] = 11
            g[1][2] = 8
            return g
        rules = RuleLearner()
        rules.learn(twins(5), twins(7), 'ACTION6', (8, 10))    # right button: +2
        rules.learn(twins(7), twins(5), 'ACTION6', (1, 10))    # left button: -2
        self.assertTrue(rules.position_dependent(next(iter(rules.click_outcomes))))
        moves = lambda xy: list(rules.predict_outcome(twins(5), 'ACTION6', xy)['moves'].values())
        self.assertEqual((moves((1, 10)), moves((8, 10))), ([(-2, 0)], [(2, 0)]))
        self.assertEqual(len(rules.inverse_pairs()), 1)
        grid = twins(5)
        objects = {'slider': cells_of(grid, 11), 'mark': cells_of(grid, 8)}
        planner = Planner(rules, grid, objects, ['ACTION6'])
        plan = planner.plan([dict(relation='same_column', a='slider', b='mark')])
        self.assertEqual([planner.click_point(i) for _, i in plan], [(1, 10)] * 2)   # the left button, twice
        from agent.cognition.skill_library import build_skills
        self.assertTrue(any(k.name.startswith('opposite-blue-blue') for k in build_skills(rules)))

    def test_unaffected_conditions_are_the_missing_links(self):
        rules, pos = trained()
        grid = board(pos)
        objects = {'player': cells_of(grid, 12), 'goal': cells_of(grid, 11)}
        planner = Planner(rules, grid, objects, list(MOVES))
        atoms = [dict(relation='inside', a='player', b='goal'), dict(relation='color_is', a='goal', color='red')]
        self.assertIsNone(planner.plan(atoms))
        self.assertEqual(planner.unaffected(atoms), [atoms[1]])


class GoalRuntimeTests(unittest.TestCase):
    def runtime(self):
        r = GoalRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program')
        r.hypothesis_samples = 1   # deterministic; voting over samples has its own test
        self.addCleanup(r.close)
        return r

    def frame(self, step, grid, actions=None):
        o = obs(step, grid)
        o['available_actions'] = list(actions or MOVES)
        return o

    def test_hypothesis_is_executed_by_the_program_without_further_model_calls(self):
        r = self.runtime()
        r.rules, pos = trained()
        works = []

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            self.assertEqual(c['work'], 'hypothesize')
            ids = {row['color']: row['id'] for row in c['objects']}
            self.assertTrue(next(row for row in c['objects'] if row['color'] == 'orange')['controllable'])
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation='inside', a=ids['orange'], b=ids['yellow'], color=None)], rationale='reach it')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            first = r.decide(self.frame(0, board(pos)))
            r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
            step = MOVES[first['action']]
            moved = (pos[0]+step[0], pos[1]+step[1])
            second = r.decide(self.frame(1, board(moved)))
        self.assertEqual(works, ['hypothesize'])
        self.assertIn(first['action'], MOVES)
        self.assertIn(second['action'], MOVES)
        self.assertEqual(r.goals[0]['presses'], 2)

    def test_hypothesis_already_true_on_screen_is_sent_back_for_repair(self):
        r = self.runtime()
        r.rules, pos = trained()
        calls = []

        def respond(model, payload):
            c = context(payload)
            calls.append(c)
            ids = {row['color']: row['id'] for row in c['objects']}
            relation = 'same_row' if len(calls) == 1 else 'inside'   # same_row already holds here
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation=relation, a=ids['orange'], b=ids['yellow'], color=None)], rationale='r')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(self.frame(0, board(pos)))
        self.assertIn('every hypothesis already holds', calls[1]['correction'])
        self.assertEqual([g['atoms'][0]['relation'] for g in r.goals], ['inside'])

    def test_sampled_hypotheses_are_ranked_by_how_often_they_recur(self):
        r = self.runtime()
        r.hypothesis_samples = 3
        r.rules, pos = trained()
        calls = []

        def respond(model, payload):
            c = context(payload)
            calls.append(c)
            ids = {row['color']: row['id'] for row in c['objects']}
            once = dict(atoms=[dict(relation='same_column', a=ids['orange'], b=ids['yellow'], color=None)], rationale='r')
            often = dict(atoms=[dict(relation='inside', a=ids['orange'], b=ids['yellow'], color=None)], rationale='r')
            return as_json(dict(observation_id=c['observation_id'], analysis='a', roles=ROLES(c),
                                hypotheses=[once, often] if len(calls) == 1 else [often]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(self.frame(0, board(pos)))
        self.assertEqual(len(calls), 3)
        self.assertEqual([g['atoms'][0]['relation'] for g in r.goals], ['inside', 'same_column'])
        # V10: only the first sample sees what the rules can produce; the others judge from the screen.
        self.assertTrue(any(x.startswith('same_column(') for x in calls[0]['achievable_relations']))
        self.assertNotIn('achievable_relations', calls[1])

    def test_already_true_hypotheses_are_dropped_individually(self):
        r = self.runtime()
        r.rules, pos = trained()
        calls = []

        def respond(model, payload):
            c = context(payload)
            calls.append(c)
            ids = {row['color']: row['id'] for row in c['objects']}
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation='same_row', a=ids['orange'], b=ids['yellow'], color=None)], rationale='true'),
                dict(atoms=[dict(relation='inside', a=ids['orange'], b=ids['yellow'], color=None)], rationale='real')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(self.frame(0, board(pos)))
        self.assertEqual(len(calls), 1)  # no repair round: the batch had a usable hypothesis
        self.assertEqual([g['atoms'][0]['relation'] for g in r.goals], ['inside'])

    def test_satisfied_hypothesis_without_level_end_is_falsified_then_new_ones_are_asked(self):
        r = self.runtime()
        r.rules, pos = trained()
        calls = []

        def respond(model, payload):
            c = context(payload)
            calls.append(c)
            ids = {row['color']: row['id'] for row in c['objects']}
            atoms = [dict(relation='same_column', a=ids['orange'], b=ids['yellow'], color=None)]
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c),
                                                       hypotheses=[dict(atoms=atoms, rationale='r')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(self.frame(0, board(pos)))
            r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
            # The hypothesis becomes true on the next screen, but the level does not end.
            result = r.decide(self.frame(1, board((9, 9))))
        self.assertIn(result['action'], MOVES)
        self.assertEqual(r.goals[0]['status'], 'falsified')
        self.assertIn('level did not end', r.falsified[0][-1])

    def test_without_a_plan_exploration_steers_by_the_hypothesis_without_a_model_call(self):
        with patch.dict('os.environ', {'COGNITION_EXPLORER': '1'}):  # graph exploration is opt-in
            r = self.runtime()
        r.rules, pos = trained()
        works = []

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            ids = {row['color']: row['id'] for row in c['objects']}
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation='color_is', a=ids['yellow'], b=None, color='red')], rationale='recolour')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            result = r.decide(self.frame(0, board(pos)))
        self.assertEqual(works, ['hypothesize'])   # no probe stage
        self.assertIn(result['action'], MOVES)
        self.assertEqual(r.goals[0]['explore_steps'], 1)

    def test_orientation_tests_controls_before_hypothesising_and_passes_the_facts(self):
        r = self.runtime()                       # nothing learned yet
        requests = []

        def respond(model, payload):
            c = context(payload)
            requests.append((c, payload))
            ids = {row['color']: row['id'] for row in c['objects']}
            return as_json(dict(observation_id=c['observation_id'], roles=[
                dict(id=ids['orange'], role='controllable')], hypotheses=[
                dict(atoms=[dict(relation='inside', a=ids['orange'], b=ids['yellow'], color=None)], rationale='reach')]))

        positions = [(1, 2), (1, 1), (1, 2), (1, 2), (2, 2)]   # UP, DOWN, LEFT (blocked by the edge), RIGHT
        with patch.object(LocalVisionLlm, '_complete', respond):
            for step, pos in enumerate(positions):
                result = r.decide(self.frame(step, board(pos)))
                if requests:
                    break
                self.assertEqual(result['action'], f'ACTION{step+1}')   # every control once, no model call
                r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
        self.assertEqual(step, 4)
        c, payload = requests[0]
        self.assertEqual(c['work'], 'hypothesize')
        self.assertTrue(any(f.startswith('UP moves') for f in c['measured_effects']))
        self.assertTrue(any(f.startswith('RIGHT moves') for f in c['measured_effects']))
        orange = next(row for row in c['objects'] if row['color'] == 'orange')
        self.assertTrue(orange['controllable'])
        self.assertTrue(all(0 <= v <= 1000 for v in orange['bbox_2d']))       # 0-1000 grounding convention
        images = [p for m in payload['messages'] if isinstance(m['content'], list)
                  for p in m['content'] if p.get('type') == 'image_url']
        self.assertEqual(len(images), 1)                    # the numbered board
        self.assertNotIn('tools', payload)                  # JSON-schema output, not a tool call
        schema = payload['response_format']['json_schema']['schema']
        self.assertEqual(list(schema['properties']), ['observation_id', 'analysis', 'roles', 'hypotheses'])
        self.assertEqual(r.goals[0]['atoms'][0]['relation'], 'inside')

    def test_orientation_clicks_candidates_in_random_order_without_the_model(self):
        r = self.runtime()
        g = slider(3)
        g[11] = [7] * 12                         # a status bar along the bottom edge
        with patch.object(LocalVisionLlm, '_complete', side_effect=AssertionError('no model call expected')):
            first = r.decide(self.frame(0, g, actions=['ACTION6']))
        self.assertEqual(first['action'], 'ACTION6')
        self.assertNotEqual(first['y'], 11)       # never the status bar
        self.assertIn(g[first['y']][first['x']], (9, 11, 8))

    def test_bundled_conditions_are_split_and_a_reached_one_is_kept(self):
        r = self.runtime()
        r.rules, pos = trained()
        requests = []

        def respond(model, payload):
            c = context(payload)
            requests.append(c)
            ids = {row['color']: row['id'] for row in c['objects']}
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation='same_column', a=ids['orange'], b=ids['yellow'], color=None),
                            dict(relation='same_color', a=ids['orange'], b=ids['yellow'], color=None)],
                     rationale='partly impossible')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            first = r.decide(self.frame(0, board(pos)))
            self.assertIn(first['action'], MOVES)            # the same_column part is planned and pressed
            r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
            r.decide(self.frame(1, board((9, 9))))           # same_column now holds, the level goes on
        # Each condition is its own goal: the unplannable same_color did not take same_column down.
        self.assertEqual([len(g['atoms']) for g in r.goals], [1, 1])
        column = next(g for g in r.goals if g['atoms'][0]['relation'] == 'same_column')
        self.assertEqual(column['status'], 'falsified')
        self.assertIn('level did not end', r.falsified[0][-1])
        self.assertEqual([a['relation'] for a in r.achieved], ['same_column'])   # kept in later plans

    def test_without_new_facts_the_program_experiments_instead_of_asking_again(self):
        r = self.runtime()
        r.rules, pos = trained()
        works = []

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            ids = {row['color']: row['id'] for row in c['objects']}
            if c['work'] == 'probe':
                return call('submit_probe', dict(observation_id=c['observation_id'], item='none', rationale='x'))
            return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                dict(atoms=[dict(relation='same_column', a=ids['orange'], b=ids['yellow'], color=None)],
                     rationale='r')]))

        with patch.object(LocalVisionLlm, '_complete', respond):
            first = r.decide(self.frame(0, board(pos)))
            r.record_execution('action_dispatched'); r.record_execution('action_acknowledged')
            r._discard(r.goals[0], 'refuted')                # every hypothesis is spent
            step = MOVES[first['action']]
            # The press repeated a known rule: nothing new, so move somewhere new rather than ask.
            result = r.decide(self.frame(1, board((pos[0]+step[0], pos[1]+step[1]))))
        self.assertEqual(works, ['hypothesize'])
        self.assertIn(result['action'], MOVES)
        self.assertEqual(r._experiments, 1)

    def test_hypothesis_follows_an_object_whose_id_changed(self):
        r = self.runtime()
        r.rules, pos = trained()
        r._receive(self.frame(0, board(pos)))
        yellow = next(o for o in r._current_objects() if o['color_ids'] == [11])   # colour IDs, 11 = yellow
        goal = dict(id='H1', atoms=[dict(relation='inside', a='e0o99', b=None, color=None)], status='active',
                    descriptors={'e0o99': ['yellow']}, anchors={'e0o99': [9, 1, 10, 2]}, probed=[])
        self.assertTrue(r._refresh(goal))
        self.assertEqual(goal['atoms'][0]['a'], yellow['object_id'])
        lost = dict(goal, descriptors={'e0o98': ['green']}, anchors={'e0o98': [0, 0, 1, 1]},
                    atoms=[dict(relation='inside', a='e0o98', b=None, color=None)])
        self.assertFalse(r._refresh(lost))

    def test_missing_link_asks_the_model_what_to_probe_and_the_program_runs_it(self):
        with patch.dict('os.environ', {'COGNITION_EXPLORER': '0'}):
            r = self.runtime()
        r.rules, pos = trained()
        works = []

        def respond(model, payload):
            c = context(payload)
            works.append(c['work'])
            if c['work'] == 'hypothesize':
                ids = {row['color']: row['id'] for row in c['objects']}
                return as_json(dict(observation_id=c['observation_id'], roles=ROLES(c), hypotheses=[
                    dict(atoms=[dict(relation='color_is', a=ids['yellow'], b=None, color='red')], rationale='recolour')]))
            self.assertEqual(c['work'], 'probe')
            self.assertEqual(c['unreachable_conditions'][0].split('(')[0], 'color_is')
            untested = {u['id']: u['what'] for u in c['untested_items']}
            visit = next(k for k, v in untested.items() if v.startswith('move the controllable object onto'))
            return call('submit_probe', dict(observation_id=c['observation_id'], item=visit, rationale='touch it'))

        with patch.object(LocalVisionLlm, '_complete', respond):
            result = r.decide(self.frame(0, board(pos)))
        self.assertEqual(works, ['hypothesize', 'probe'])
        self.assertIn(result['action'], MOVES)
        self.assertEqual(r.probe['kind'], 'visit')


if __name__ == '__main__':
    unittest.main()
