"""ADK skills (SKILL.md structure) built from program-learned rules; see docs/skills-design-ja.md.

L1 (name, description) tells the model what a rule is for; L2 holds the measured table and
how to use it. Exact geometry stays in program tools listed in adk_additional_tools, which
read the current frame in-process (skill scripts cannot see live state).
"""
from google.adk.skills import models

from agent.controls import ACTION_TO_BUTTON
from .perception import COLORS

MOVE_TOOLS = ['plan_path', 'predict_effect']


def color_name(c):
    return COLORS[c] if isinstance(c, int) and 0 <= c < len(COLORS) else str(c)


def _kebab(text):
    out = ''.join(ch if ch.isalnum() else '-' for ch in text.lower())
    return '-'.join(p for p in out.split('-') if p)[:64]


def move_skill_name(rules):
    sigs = rules.movers()
    return _kebab(f"move-{'-'.join(sorted({color_name(s[0]) for s in sigs}))}-block") if sigs else None


def click_skill_name(sig):
    return _kebab(f"click-{color_name(sig[0])}-{len(sig[1])}-cells")


def skill_name_for(rules, prediction):
    return click_skill_name(prediction['clicked']) if prediction['kind'] == 'click' else move_skill_name(rules)


def _validation(stats):
    """Prediction record appended to a skill: its reliability is measured, not asserted."""
    if not stats:
        return '', 'Not yet used for a prediction.'
    replay = stats.get('replay')
    summary = (f" Predictions {stats['hits']}/{stats['predictions']} correct; version {stats['version']}; "
               f"{stats['status']}.")
    lines = [f"Validation: {stats['hits']}/{stats['predictions']} predictions matched; version {stats['version']}; "
             f"replay of this level's transitions {'n/a' if replay is None else f'{replay:.0%}'} consistent; status {stats['status']}."]
    for c in stats.get('counterexamples', []):
        lines.append(f"- counterexample at step {c['step']} ({c['control']}): {c['detail']}")
    return summary, '\n'.join(lines)


def build_skills(rules, stats=None):
    stats = stats or {}
    skills = []
    sigs = rules.movers()
    if sigs:
        colors = sorted({color_name(s[0]) for s in sigs})
        name = move_skill_name(rules)
        summary, validation = _validation(stats.get(name))
        deltas = rules.direction_deltas(sigs)
        rows = [f"- {ACTION_TO_BUTTON.get(a, a)}: moves by {d}; {rules.effective[a]}/{rules.tries[a]} presses moved something"
                for a, d in sorted(deltas.items())]
        blocking = ', '.join(sorted(color_name(c) for c in rules.blocking_colors(sigs))) or 'none observed yet'
        entered = ', '.join(sorted(color_name(c) for c in rules.entered_colors(sigs))) or 'none observed yet'
        skills.append(models.Skill(
            frontmatter=models.Frontmatter(
                name=name,
                description=(f"Direction controls move the {' and '.join(colors)} block (the controllable object). "
                             "The program can execute moves to an object." + summary),
                metadata={'adk_additional_tools': MOVE_TOOLS}),
            instructions='\n'.join([
                'Measured in this level (program tally, not a model interpretation):', *rows,
                f'- entered colours: {entered}', f'- stopped by colours: {blocking}',
                'plan_path(target_object_id) returns the exact presses to an object and whether it is reachable; '
                'predict_effect(control) checks one press. Unknown colours are assumed passable until a press '
                'is stopped there.', validation])))
    for sig, moved in rules.click_moves.items():
        key = ('click', sig)
        recolor = rules.recolor.get((sig[1], sig[0]))
        effects = [f"{color_name(s[0])} object moves by {d}" for s, d in moved.items()]
        if recolor is not None and recolor != sig[0]:
            effects.append(f"the clicked object turns {color_name(recolor)}")
        if not effects:
            effects = ['no object effect observed']
        if rules.state_dependent(sig):
            effects.append(f"state-dependent: {len(rules.click_outcomes[sig])} different outcomes, predicted only in a seen context")
        name = click_skill_name(sig)
        if any(s.name == name for s in skills):
            continue
        summary, validation = _validation(stats.get(name))
        skills.append(models.Skill(
            frontmatter=models.Frontmatter(
                name=name,
                description=(f"Clicking a {color_name(sig[0])} object of {len(sig[1])} cells: {'; '.join(effects)}. "
                             f"{rules.effective[key]}/{rules.tries[key]} clicks had an effect." + summary),
                metadata={'adk_additional_tools': ['predict_effect']}),
            instructions=('Measured effect of the last click on this appearance: ' + '; '.join(effects) + '. '
                          'The effect may depend on the board state; predict_effect(control="CLICK", object_id=...) '
                          'returns the learned effect for a current object.\n' + validation)))
    if rules.ticks():
        skills.append(models.Skill(
            frontmatter=models.Frontmatter(
                name='step-counter',
                description='Some cells change on almost every action regardless of the control: a step or '
                            'time counter, not a result of the action and not a goal.'),
            instructions='Ignore the counter when judging whether an action had an effect; prefer short plans.'))
    return skills
