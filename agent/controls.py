"""Pure controller translation. No game calls, cursor mutation, or budget changes."""
BUTTON_TO_ACTION = {
    'UP': 'ACTION1', 'DOWN': 'ACTION2', 'LEFT': 'ACTION3', 'RIGHT': 'ACTION4',
    'ACT': 'ACTION5', 'CLICK': 'ACTION6', 'UNDO': 'ACTION7', 'RESET': 'RESET',
}
ACTION_TO_BUTTON = {action: button for button, action in BUTTON_TO_ACTION.items()}


def ground_button(value: dict, observation: dict) -> dict:
    """Translate a button name; CLICK always needs explicit original-pixel coordinates."""
    value = dict(value)
    name = value.get('action')
    if isinstance(name, str):
        name = name.strip().upper()
    if not isinstance(name, str) or name not in BUTTON_TO_ACTION:
        return value
    value['action'] = BUTTON_TO_ACTION[name]
    return value


def controller_context(value):
    """Present controller names in model context while preserving host records."""
    if isinstance(value, list):
        return [controller_context(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: controller_context(item) for key, item in value.items()}
    if isinstance(result.get('action'), str):
        result['action'] = ACTION_TO_BUTTON.get(result['action'], result['action'])
    if 'available_actions' in result:
        result['available_actions'] = [ACTION_TO_BUTTON.get(a, a) for a in result['available_actions']]
    return result
