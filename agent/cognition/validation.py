"""Strict parsing and legal, explicitly positioned actions."""
from __future__ import annotations

from agent.controls import ground_button
from .state import Action, ActionIntent


def validate_intent(raw: dict | ActionIntent, obs: dict) -> ActionIntent:
    """Check controller availability without prematurely binding a click to pixels."""
    value = raw.model_dump() if isinstance(raw, ActionIntent) else dict(raw)
    value = ground_button(value, obs)
    intent = ActionIntent.model_validate(value)
    if intent.action not in set(obs.get('available_actions', [])) - {'RESET'}:
        raise ValueError('unavailable action intent: '+intent.action)
    if intent.action == 'ACTION6' and not intent.target_query.strip():
        raise ValueError('CLICK requires a target_query identifying a visible instance and part')
    if intent.action != 'ACTION6' and intent.target_query:
        raise ValueError('target_query belongs to CLICK; directional actions need no cursor')
    return intent


def validate_action(raw: dict | Action, obs: dict, *, reset: bool = False) -> Action:
    value = raw.model_dump() if isinstance(raw, Action) else dict(raw)
    value = ground_button(value, obs)
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
