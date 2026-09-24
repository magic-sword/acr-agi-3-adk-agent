"""Visible controller inputs and cursor-only clicks at the execution boundary."""
import unittest

from agent.controls import BUTTON_TO_ACTION, controller_context
from agent.cognition.engine import CognitiveTurn, new_memory
from agent.cognition.state import Action, PlanNode, Predicate
from agent.cognition.validation import validate_action


class ControllerTests(unittest.TestCase):
    def observation(self):
        return {'game_id': 'controller', 'state': 'NOT_FINISHED', 'step': 0,
                'available_actions': [f'ACTION{i}' for i in range(1, 8)],
                'width': 4, 'height': 3, 'grid': [[0]*4 for _ in range(3)],
                'cursor': {'x': 2, 'y': 1}, 'remaining_actions': 5}

    def test_displayed_buttons_map_to_exact_engine_actions(self):
        obs = self.observation()
        for button, expected in BUTTON_TO_ACTION.items():
            if button == 'RESET':
                continue
            result = validate_action({'action': button.lower()}, obs)
            self.assertEqual(result.action, expected)
            if button == 'CLICK':
                self.assertEqual((result.x, result.y), (2, 1))
            else:
                self.assertIsNone(result.x)
                self.assertIsNone(result.y)
        self.assertEqual(obs['cursor'], {'x': 2, 'y': 1})

    def test_click_rejects_ambiguous_missing_and_invalid_positions(self):
        obs = self.observation()
        for cursor in (None, {}, {'x': True, 'y': 1}, {'x': 2.0, 'y': 1},
                       {'x': -1, 'y': 0}, {'x': 4, 'y': 0}, {'x': 0, 'y': 3}):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                validate_action({'action': 'CLICK'}, {**obs, 'cursor': cursor})
        for raw in ({'action': 'CLICK', 'x': 1}, {'action': 'CLICK', 'y': 1},
                    {'action': 'UP', 'x': 1}, {'action': 'RESET'}, {'action': 'JUMP'}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_action(raw, obs)
        for button in ('UP', 'CLICK', 'UNDO'):
            with self.assertRaises(ValueError):
                validate_action({'action': button}, {**obs, 'available_actions': []})

    def test_click_samples_cursor_at_selection_then_freezes_for_commit(self):
        turn = CognitiveTurn(new_memory('controller'), self.observation())
        turn.observe()
        effect = Predicate(kind='state', value='WIN')
        turn.memory.plan = [PlanNode(id='click', subgoal='activate target',
                            action=Action(action='CLICK'), completion=[effect], effects=[effect])]
        # A stored CLICK does not carry coordinates from the time its plan was made.
        turn.obs['cursor'] = {'x': 3, 'y': 2}
        turn.act()
        turn.obs['cursor'] = {'x': 0, 'y': 0}
        result = turn.commit()
        self.assertEqual((result['action'], result['x'], result['y']), ('ACTION6', 3, 2))
        self.assertEqual(turn.memory.pending.action.x, 3)

    def test_model_context_translates_inputs_without_changing_host_evidence(self):
        host = {'observation': self.observation(), 'pending': {'action': {'action': 'ACTION6', 'x': 2, 'y': 1}}}
        view = controller_context(host)
        self.assertEqual(view['observation']['available_actions'], ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ACT', 'CLICK', 'UNDO'])
        self.assertEqual(view['pending']['action'], {'action': 'CLICK', 'x': 2, 'y': 1})
        self.assertEqual(host['pending']['action']['action'], 'ACTION6')


if __name__ == '__main__':
    unittest.main()
