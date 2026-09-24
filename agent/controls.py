"""Pure controller translation. No game calls, cursor mutation, or budget changes."""
BUTTON_TO_ACTION = {
    'UP': 'ACTION1', 'DOWN': 'ACTION2', 'LEFT': 'ACTION3', 'RIGHT': 'ACTION4',
    'ACT': 'ACTION5', 'CLICK': 'ACTION6', 'UNDO': 'ACTION7', 'RESET': 'RESET',
}
ACTION_TO_BUTTON = {action: button for button, action in BUTTON_TO_ACTION.items()}


def ground_button(value: dict, observation: dict) -> dict:
    """Resolve a visible button; CLICK samples the current cursor at action selection.

    Legacy ACTION names are accepted by the caller for existing host clients.
    A click never invents a default position or accepts a second coordinate source.
    """
    value = dict(value)
    name = value.get('action')
    if isinstance(name, str):
        name = name.strip().upper()
    if not isinstance(name, str) or name not in BUTTON_TO_ACTION:
        return value
    if name == 'CLICK':
        if value.get('x') is not None or value.get('y') is not None:
            raise ValueError('CLICK uses the current cursor; omit x and y')
        cursor = observation.get('cursor')
        if not isinstance(cursor, dict) or any(type(cursor.get(k)) is not int for k in ('x', 'y')):
            raise ValueError('CLICK requires a current cursor; use move_cursor first')
        value.update(x=cursor['x'], y=cursor['y'])
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
