"""Host imports for the action-grounding skill's pure controller implementation."""
from importlib import import_module

_controls = import_module('agent.skills.action-grounding.scripts.controls')
BUTTON_TO_ACTION = _controls.BUTTON_TO_ACTION
ACTION_TO_BUTTON = _controls.ACTION_TO_BUTTON
ground_button = _controls.ground_button
controller_context = _controls.controller_context
