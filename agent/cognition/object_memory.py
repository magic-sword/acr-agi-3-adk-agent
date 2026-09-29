"""Revisable object hypotheses above measured regions; no semantic identities.

Only one-to-one region tracks carry identity without a new uncertainty segment.
Containment, contact, common motion and model grouping never imply causal links.
Snapshots in trials are immutable; lineage is consulted as tentative evidence only.
"""
from copy import deepcopy
from itertools import combinations

from .geometry import bounds, adjacent
from .region_masks import support, encode, decode


def empty(epoch=0, serial=0):
    return dict(epoch=epoch, serial=serial, observation_id=None, objects=[], motion_pairs={})


def _new_id(state):
    state['serial'] += 1
    return f"e{state['epoch']}o{state['serial']}"


def _region(state, candidate, observation_id, previous=None, status='new'):
    points = support(candidate)
    uncertain = status == 'appearance_hypothesis'
    return dict(object_id=previous['object_id'] if previous else _new_id(state),
                observation_id=observation_id, kind='region',
                candidate_refs=[candidate['id']], track_ids=[candidate['track_id']] if candidate.get('track_id') else [],
                bbox=candidate['bbox'], mask_runs=encode(points), support_known=bool(points),
                color_ids=candidate.get('color_ids', candidate.get('colors', [])),
                sources=candidate.get('sources', ['program']),
                identity_status=status,
                identity_segment=(previous['identity_segment'] + int(uncertain)) if previous else 0,
                previous_version=previous['observation_id'] if previous else None,
                predecessors=list(previous.get('predecessors', [])) if previous else [], part_ids=[])


def _group(state, members, observation_id, kind, old, evidence):
    ids = sorted(o['object_id'] for o in members)
    prior = next((o for o in old if o['kind'] == kind and sorted(o['part_ids']) == ids), None)
    points = set().union(*(decode(o['mask_runs']) for o in members))
    box = bounds(points)
    segments = {o['object_id']: o['identity_segment'] for o in members}
    offsets = {o['object_id']: [o['bbox'][0]-box[0], o['bbox'][1]-box[1]] for o in members}
    layout_changed = bool(prior and prior.get('part_offsets') != offsets)
    segment_changed = bool(prior and (prior.get('part_segments') != segments or layout_changed))
    return dict(object_id=prior['object_id'] if prior else _new_id(state),
                observation_id=observation_id, kind=kind,
                candidate_refs=sorted({r for o in members for r in o['candidate_refs']}),
                track_ids=[], bbox=box, mask_runs=encode(points), support_known=True,
                color_ids=sorted({c for o in members for c in o['color_ids']}), sources=['grouping'],
                identity_status='appearance_hypothesis' if segment_changed else 'tracked' if prior else 'new',
                identity_segment=prior['identity_segment'] + int(bool(segment_changed)) if prior else 0,
                previous_version=prior['observation_id'] if prior else None,
                predecessors=list(prior.get('predecessors', [])) if prior else [],
                part_ids=ids, part_segments=segments, part_offsets=offsets,
                relative_layout_changed=layout_changed, grouping_evidence=evidence)


def update(state, record):
    """Update all region identities before proposing additive, bounded groups."""
    contiguous = state.get('observation_id') == record.get('before_observation_id')
    if not contiguous or record['status'] in ('initial', 'boundary', 'unavailable'):
        state = empty(state.get('epoch', 0) + 1, state.get('serial', 0))
    old = state['objects']
    observation_id = record['observation_id']
    old_regions = [o for o in old if o['kind'] == 'region']
    candidates = record['candidates']
    by_track = {t: o for o in old_regions for t in o['track_ids']}
    # Program-only sensor uses different before/after prefixes. Map via its
    # measured geometry, never by reusing a frame-local candidate ID.
    old_by_box = {}
    for o in old_regions:
        old_by_box.setdefault(tuple(o['bbox']), []).append(o)
    program_links = {}
    if record.get('proposal_mode') != 'sam_initial':
        for change in record['changes']:
            before = change['before']
            matches = old_by_box.get(tuple(before['bbox']), [])
            if len(matches) == 1:
                program_links[change['after']['id']] = (
                    matches[0], 'appearance_hypothesis' if change.get('changed', {}).get('appearance') or
                    change.get('changed', {}).get('size') else 'tracked')
        # Same support/appearance at the same location is an observed continuity.
        for c in candidates:
            matches = old_by_box.get(tuple(c['bbox']), [])
            if len(matches) == 1 and c['id'] not in program_links:
                unchanged = any(p['after_id'] == c['id'] for p in record['unchanged'])
                if unchanged:
                    program_links[c['id']] = (matches[0], 'tracked')
    paired, objects, pending = set(), [], []
    for c in candidates:
        prior = by_track.get(c.get('track_id'))
        status = 'tracked'
        if prior is None and c['id'] in program_links:
            prior, status = program_links[c['id']]
        if prior and prior['object_id'] not in paired:
            objects.append(_region(state, c, observation_id, prior, status))
            paired.add(prior['object_id'])
        else:
            pending.append(c)
    lost = [o for o in old_regions if o['object_id'] not in paired]
    blocked_tracks = {u['candidate'].get('track_id') for u in record['unresolved']
                      if u.get('side') == 'before' and u.get('reason') in ('multiple_exact_matches', 'competing_tracks')}
    candidate_points = {c['id']: support(c) for c in pending}
    old_points = {o['object_id']: decode(o['mask_runs']) for o in lost}
    for c in pending:
        pts = candidate_points[c['id']]
        matches = [o for o in lost if pts and pts == old_points[o['object_id']] and
                   not blocked_tracks.intersection(o['track_ids'])]
        reverse = [x for x in pending if candidate_points[x['id']] == pts]
        if len(matches) == len(reverse) == 1 and matches[0]['object_id'] not in paired:
            prior = matches[0]
            o = _region(state, c, observation_id, prior, 'appearance_hypothesis')
            paired.add(prior['object_id'])
        else:
            o = _region(state, c, observation_id)
            # Overlap is possible lineage, never automatic inheritance of trials.
            links = [p['object_id'] for p in lost if pts and old_points[p['object_id']] and
                     len(pts & old_points[p['object_id']]) / min(len(pts), len(old_points[p['object_id']])) >= .5]
            o['predecessors'] = links[:8]
            o['predecessors_omitted'] = max(0, len(links) - 8)
        objects.append(o)

    # Contact groups use original components, keeping every part independently.
    local = [o for o in objects if 'component' in o['sources'] and 0 < len(decode(o['mask_runs'])) <= 128]
    local = local[:128]
    points = {o['object_id']: decode(o['mask_runs']) for o in objects}
    neighbors = {o['object_id']: set() for o in local}
    vicinity = {o['object_id']: {(x+dx, y+dy) for x, y in points[o['object_id']]
                               for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx)+abs(dy) <= 2}
                for o in local}
    motion = {c['after']['track_id']: c['delta_xy'] for c in record['changes']
              if c.get('after', {}).get('track_id') and c.get('delta_xy')}
    pair_counts = {}
    motion_neighbors = {o['object_id']: set() for o in local}
    for a, b in combinations(local, 2):
        aid, bid = a['object_id'], b['object_id']
        if not vicinity[aid].intersection(points[bid]):
            continue
        if adjacent(points[aid], points[bid]):
            neighbors[aid].add(bid); neighbors[bid].add(aid)
        da = motion.get(next(iter(a['track_ids']), None))
        db = motion.get(next(iter(b['track_ids']), None))
        key = '|'.join(sorted([aid, bid]))
        if da and any(da) and da == db:
            pair_counts[key] = state['motion_pairs'].get(key, 0) + 1
        elif a['identity_status'] == b['identity_status'] == 'tracked' and not da and not db:
            pair_counts[key] = state['motion_pairs'].get(key, 0)
        if pair_counts.get(key, 0) >= 2:
            motion_neighbors[aid].add(bid); motion_neighbors[bid].add(aid)
    by_id = {o['object_id']: o for o in objects}
    groups = []
    omitted = 0
    for kind, graph in [('spatial_group', neighbors), ('co_motion_group', motion_neighbors)]:
        seen = set()
        for seed in graph:
            if seed in seen:
                continue
            stack, ids = [seed], set()
            while stack:
                ident = stack.pop()
                if ident in ids:
                    continue
                ids.add(ident); stack.extend(graph[ident] - ids)
            seen.update(ids)
            if len(ids) < 2:
                continue
            union = set().union(*(points[i] for i in ids))
            box = bounds(union)
            if len(ids) > 8 or (box[2]-box[0]+1)*(box[3]-box[1]+1) > 256 or len(groups) >= 64:
                omitted += 1
                continue
            groups.append(_group(state, [by_id[i] for i in sorted(ids)], observation_id, kind, old,
                'adjacent native masks; may be separate touching objects' if kind == 'spatial_group' else
                'at least two common nonzero translations; may be synchronized separate objects'))
    for prior in old:
        if prior['kind'] == 'interpreted_group' and all(i in by_id for i in prior['part_ids']):
            groups.append(_group(state, [by_id[i] for i in prior['part_ids']], observation_id,
                                 'interpreted_group', old, prior['grouping_evidence']))
    objects += groups
    points.update({o['object_id']: decode(o['mask_runs']) for o in groups})
    current_ids = {o['object_id'] for o in objects}
    lost_groups = [o for o in old if o['kind'] != 'region' and o['object_id'] not in current_ids]
    for o in objects:
        if o['identity_status'] != 'new':
            continue
        pts = points[o['object_id']]
        links = list(o['predecessors'])
        for p in lost_groups:
            pp = decode(p['mask_runs'])
            if pts and pp and len(pts & pp) / min(len(pts), len(pp)) >= .5:
                links.append(p['object_id'])
        o['predecessors'] = links[:8]
        o['predecessors_omitted'] = o.get('predecessors_omitted', 0) + max(0, len(links)-8)
    relations = []
    # Only immediate containment edges. Geometric inclusion is a part hypothesis.
    for child in objects:
        cp = points[child['object_id']]
        if not cp:
            continue
        parents = [p for p in objects if cp < points[p['object_id']]]
        immediate = []
        for parent in sorted(parents, key=lambda p: len(points[p['object_id']])):
            pp = points[parent['object_id']]
            if not any(points[mid['object_id']] < pp for mid in immediate):
                immediate.append(parent)
                relations.append(dict(kind='part_of_hypothesis', child=child['object_id'], parent=parent['object_id']))
    state.update(objects=objects, observation_id=observation_id, motion_pairs=pair_counts,
                 relations=relations, grouping_omitted=omitted,
                 grouping_limited=len([o for o in objects if 'component' in o['sources']]) > 128)
    return state


def index(state):
    return dict(observation_id=state['observation_id'],
        columns=['object_id', 'candidate_refs', 'bbox_inclusive', 'color_ids', 'grouping_method', 'identity_status', 'part_ids'],
        rows=[[o[k] for k in ('object_id', 'candidate_refs', 'bbox', 'color_ids', 'kind', 'identity_status', 'part_ids')]
              for o in state['objects']],
        relations=deepcopy(state.get('relations', [])), grouping_omitted=state.get('grouping_omitted', 0),
        grouping_limited=state.get('grouping_limited', False),
        note='Grouping methods are host metadata, not object names or game goals. Describe visible colors/shapes/positions. '
             'All entries are hypotheses; part-of, contact and co-motion do not prove physical unity or causation.')


def bind_target(state, target):
    """Host binding; model-selected IDs must already have passed validation."""
    objects = state['objects']
    refs = set(target['candidate_refs'])
    obj = next((o for o in objects if o['object_id'] == target.get('object_id')), None)
    if obj is None and refs:
        exact = [o for o in objects if set(o['candidate_refs']) == refs]
        # Multiple grouping explanations for identical support stay alternatives.
        if len(exact) == 1:
            obj = exact[0]
        elif not exact:
            members = [o for o in objects if o['kind'] == 'region' and set(o['candidate_refs']) <= refs]
            if len(members) > 1 and all(o['support_known'] for o in members):
                obj = _group(state, members, state['observation_id'], 'interpreted_group', [],
                             'model selected several regions; not a measured physical unity')
                objects.append(obj)
    if obj:
        target['object_id'] = obj['object_id']
        if not refs:
            target['candidate_refs'] = list(obj['candidate_refs'])
        target['identity_status'] = obj['identity_status']
    return target


def snapshot(state, ids):
    return [deepcopy(o) for o in state['objects'] if o['object_id'] in ids]


def allowed_refs(state, ident):
    obj = next((o for o in state['objects'] if o['object_id'] == ident), None)
    if obj is None:
        return set()
    pixels = decode(obj['mask_runs'])
    return set(obj['candidate_refs']) | {r for o in state['objects'] if o['support_known'] and
        decode(o['mask_runs']) <= pixels for r in o['candidate_refs']}


def trial_card(trial):
    # Exact geometry and target snapshots stay in audit records.
    keys = ('observation_id', 'invocation_id', 'purpose', 'verb', 'target_query', 'expected_effect',
            'assessment', 'evidence', 'causal_interpretation', 'actual_trials', 'target_object_ids', 'baseline', 'acted_part_ids',
            'execution_status', 'actual_trials_omitted')
    return {k: deepcopy(trial[k]) for k in keys if k in trial}


def history(state, trials, ids, verb=None, limit=3):
    current = {o['object_id']: o for o in state['objects'] if o['object_id'] in ids}
    matches = []
    seen_invocations = set()
    for t in reversed(trials):
        if t.get('execution_status') == 'not_executed_or_unacknowledged':
            continue
        bound = {o['object_id']: o for o in t.get('target_objects', [])}
        direct = set(current) & set(bound)
        lineage = {p for o in current.values() for p in o['predecessors']} & set(bound)
        if not direct and not lineage:
            continue
        invocation = t.get('invocation_id')
        if invocation and invocation in seen_invocations:
            continue
        if invocation:
            seen_invocations.add(invocation)
        certain = bool(direct) and not lineage and all(current[i]['identity_segment'] == bound[i]['identity_segment'] for i in direct)
        matches.append(dict(trial=trial_card(t),
            correspondence='tracked' if certain else 'tentative_identity_or_lineage',
            matched_object_ids=sorted(direct), prior_object_ids=sorted(lineage),
            same_verb=verb is not None and t.get('verb') == verb))
    matches.sort(key=lambda m: (m['correspondence'] != 'tracked', not m['same_verb']))
    return dict(trials=matches[:limit], omitted=max(0, len(matches)-limit),
                note='Conditions, acted part and effect scope still apply. No retrieved trial is not proof of never tried.')


def observations(previous, current, before_grid, after_grid, action):
    """Per-object measured effects and exact pre-action mask hit, not causality."""
    if not before_grid or not after_grid or len(before_grid) != len(after_grid) or len(before_grid[0]) != len(after_grid[0]):
        return []
    after = {o['object_id']: o for o in current['objects']}
    rows = []
    for old in previous['objects']:
        if not old['support_known']:
            continue
        points = decode(old['mask_runs'])
        new = after.get(old['object_id'])
        changed = sum(before_grid[y][x] != after_grid[y][x] for x, y in points)
        hit = ((action.get('x'), action.get('y')) in points) if action.get('action') in ('CLICK', 'ACTION6') else None
        delta = [new['bbox'][i]-old['bbox'][i] for i in (0, 1)] if new else None
        rows.append(dict(object_id=old['object_id'], object_kind=old['kind'], before_version=old['observation_id'],
            after_version=current['observation_id'], correspondence=new['identity_status'] if new else 'unresolved',
            changed_pixels_on_previous_support=changed, delta_xy=delta, clicked_previous_mask=hit,
            scope='previous visible mask; zero change is not proof of no effect elsewhere'))
    return rows
