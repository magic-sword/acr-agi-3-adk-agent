"""Shared, versioned puzzle notes. Host evidence is never model-editable."""
from copy import deepcopy
import json
from pathlib import Path
import re

GOAL = 'Discover the current game goal and solve it using observed evidence.'
SYSTEM_BOOKMARKS = {'current_goal', 'latest_result', 'current_subgoal', 'active_experiment', 'latest_review'}


class Notebook:
    def __init__(self, game_id, run_id, directory=None, emit=None):
        self.game_id, self.run_id = game_id, run_id
        self.directory = Path(directory) if directory else None
        self.emit = emit or (lambda event, **data: None)
        self.pages, self.bookmarks = {}, {}
        self.segment, self.sequence = 0, 0
        self.observation_id = None
        self.source_ids = set()
        self._commit('goal', 'goal', 'Current goal', GOAL, [], author='host')
        self.bookmarks['current_goal'] = 'goal'
        self._save_bookmarks()

    def _save(self, name, value):
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / name
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
            temp.replace(path)

    def _save_bookmarks(self):
        self._save('bookmarks.json', {'game_id': self.game_id, 'run_id': self.run_id,
                                    'segment': self.segment, 'bookmarks': self.bookmarks})

    def _commit(self, page_id, kind, title, text, evidence_ids, *, author='agent',
                status='open', reason='', data=None):
        old = self.pages.get(page_id)
        page = {'id': page_id, 'revision': old['revision']+1 if old else 1,
                'kind': kind, 'title': title, 'text': text, 'status': status,
                'author': author, 'game_id': self.game_id, 'run_id': self.run_id,
                'segment': self.segment, 'observation_id': self.observation_id,
                'evidence_ids': list(evidence_ids)}
        if reason:
            page['reason'] = reason
        if data is not None:
            page['data'] = deepcopy(data)
        self._save(f'{page_id}-v{page["revision"]}.json', page)
        self.pages[page_id] = page
        self.emit('note_changed', before=deepcopy(old), after=deepcopy(page))
        return deepcopy(page)

    def _editable(self, page_id, expected_revision):
        page = self.pages.get(page_id)
        if page is None:
            raise ValueError('unknown note ID')
        if page['kind'] in ('result', 'subgoal', 'experiment'):
            raise ValueError('observed results are host-owned; write an interpretation note')
        if page['segment'] != self.segment:
            raise ValueError('note belongs to a previous level/reset; create a new scoped note')
        if page['revision'] != expected_revision:
            raise ValueError('stale note revision; read the current note before updating')
        return page

    def _evidence(self, ids):
        if len(ids) > 8 or not set(ids) <= self.source_ids | self.pages.keys():
            raise ValueError('use at most eight recorded observation, experience or note IDs')

    def write(self, kind, title, text, evidence_ids, note_id='', expected_revision=0):
        if kind not in ('goal', 'hypothesis', 'plan', 'interpretation'):
            raise ValueError('kind must be goal, hypothesis, plan or interpretation')
        if not 1 <= len(title.strip()) <= 80 or not 1 <= len(text.strip()) <= (400 if kind == 'goal' else 1000):
            raise ValueError('title must be 1..80 characters; text 1..400 for goal, otherwise 1..1000')
        self._evidence(evidence_ids)
        if kind in ('hypothesis', 'interpretation') and not evidence_ids:
            raise ValueError('hypotheses and interpretations need recorded evidence references')
        if kind == 'goal' and note_id != 'goal':
            raise ValueError('update the existing goal page with its current revision')
        if note_id:
            old = self._editable(note_id, expected_revision)
            if old['kind'] != kind:
                raise ValueError('note kind cannot change')
        else:
            if expected_revision != 0:
                raise ValueError('new notes use revision zero')
            if sum(p['kind'] in ('goal', 'hypothesis', 'plan', 'interpretation') and
                   p['status'] == 'open' and p['segment'] == self.segment
                   for p in self.pages.values()) >= 128:
                raise ValueError('active note limit reached; retire unused notes')
            self.sequence += 1
            note_id = f'note-{self.sequence}'
        return self._commit(note_id, kind, title.strip(), text.strip(), evidence_ids)

    def erase(self, note_id, expected_revision, reason):
        old = self._editable(note_id, expected_revision)
        if note_id == 'goal':
            raise ValueError('revise the current goal instead of erasing it')
        if not 1 <= len(reason.strip()) <= 300:
            raise ValueError('provide a withdrawal reason of 1..300 characters')
        result = self._commit(note_id, old['kind'], old['title'], old['text'],
                              old['evidence_ids'], status='withdrawn', reason=reason.strip())
        for name, ref in list(self.bookmarks.items()):
            if ref == note_id:
                self.bookmark(name, '')
        return result

    def bookmark(self, name, note_id):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', name):
            raise ValueError('bookmark name must be 1..32 lowercase letters, digits, _ or -')
        if name in SYSTEM_BOOKMARKS:
            raise ValueError('system bookmarks are maintained automatically')
        if note_id:
            page = self.pages.get(note_id)
            if not page or page['status'] != 'open' or page['segment'] != self.segment:
                raise ValueError('bookmark must refer to an open note in the current segment')
            if name not in self.bookmarks and sum(k not in SYSTEM_BOOKMARKS
                                                  for k in self.bookmarks) >= 6:
                raise ValueError('custom bookmark limit is six')
        old = self.bookmarks.get(name)
        if note_id:
            self.bookmarks[name] = note_id
        else:
            self.bookmarks.pop(name, None)
        self._save_bookmarks()
        result = {'name': name, 'before': old, 'after': note_id or None}
        self.emit('bookmark_changed', **result)
        return result

    def system_bookmark(self, name, page_id):
        if name not in SYSTEM_BOOKMARKS:
            raise ValueError('not a system bookmark')
        if page_id:
            self.bookmarks[name] = page_id
        else:
            self.bookmarks.pop(name, None)
        self._save_bookmarks()
        self.emit('bookmark_changed', name=name, after=page_id or None)

    def goal_path(self, page_id=None):
        page_id = page_id or self.bookmarks.get('current_subgoal', 'goal')
        path = []
        while page_id:
            page = self.pages[page_id]
            path.append(page)
            page_id = page.get('data', {}).get('parent_id')
        return deepcopy(list(reversed(path)))

    def begin_segment(self, boundary):
        self.segment += 1
        self.bookmarks = {'current_goal': 'goal'}
        self.emit('notebook_boundary', segment=self.segment, boundary=boundary,
                  bookmarks=deepcopy(self.bookmarks))
        self._commit('goal', 'goal', 'Current goal', GOAL, [], author='host')
        self._save_bookmarks()

    def record_observation(self, observation, outcome, boundary=''):
        self.observation_id = observation['observation_id']
        self.source_ids.add(self.observation_id)
        if boundary:
            self.begin_segment(boundary)
        data = {'observation_id': self.observation_id, 'step': observation['step'],
                'state': observation['state'], 'boundary': boundary, 'outcome': deepcopy(outcome)}
        refs = [outcome['experience_id']] if outcome else []
        self.source_ids.update(refs)
        page_id = f'result-{observation["step"]}'
        page = self._commit(page_id, 'result', f'Observed step {observation["step"]}',
                            'Host-recorded observation and issued action. Experiment verdicts are separate records.',
                            refs, author='host', data=data)
        self.bookmarks['latest_result'] = page_id
        self._save_bookmarks()
        self.emit('bookmark_changed', name='latest_result', after=page_id)
        return page

    def opening(self):
        """Small mandatory view. Never includes the full note collection."""
        result = self.pages.get(self.bookmarks.get('latest_result'))
        def experiment_view(name):
            page = deepcopy(self.pages.get(self.bookmarks.get(name)))
            if page:
                page['data'].pop('before_pixels', None)
                page['data'].pop('before_context', None)
                if name == 'latest_review':
                    data = page['data']
                    # Open the verdict, not a copyable old action proposal. The
                    # full frozen plan is still accessible by experiment ID.
                    return {'id': page['id'], 'revision': page['revision'], 'segment': page['segment'],
                            'data': {k: data[k] for k in ('subgoal_id', 'status', 'measurement', 'review', 'reviewer')},
                            'question': data['plan']['question'],
                            'expected': data['plan']['expected']}
            return page
        return deepcopy({'segment': self.segment, 'handoff': self.handoff(), 'goal': self.pages['goal'],
                         'goal_path': self.goal_path(),
                         'active_experiment': experiment_view('active_experiment'),
                         'latest_review': experiment_view('latest_review'),
                         'latest_result': result,
                         'bookmarks': [{'name': name, 'id': ref,
                                        'revision': self.pages[ref]['revision'],
                                        'title': self.pages[ref]['title']}
                                       for name, ref in self.bookmarks.items()]})

    def handoff(self):
        """A derived decision view; the versioned review remains the source of truth."""
        page = self.pages.get(self.bookmarks.get('latest_review'))
        if not page or page['segment'] != self.segment:
            return None
        data = page['data']; review = data['review']
        return deepcopy({
            'experiment_id': page['id'], 'review_revision': page['revision'],
            'verdict': review['verdict'], 'observed_fact': review['finding'],
            'evidence_ids': review['evidence_ids'],
            'understanding': review.get('understanding'),
            'subgoal_status': review['subgoal_status'],
            'next_step': review['next_step'], 'update': review['update'],
            'revision_required': review['verdict'] != 'supported',
            'remaining_predeclared_attempts': max(0, data['plan']['max_attempts'] - data['attempt']),
        })

    def read(self, reference='', query='', offset=0, include_previous=False):
        if type(offset) is not int or offset < 0 or len(query) > 200:
            raise ValueError('offset must be nonnegative and query at most 200 characters')
        if reference:
            ref = self.bookmarks.get(reference, reference)
            if ref not in self.pages:
                raise ValueError('unknown note or bookmark; read the index first')
            result = {'page': deepcopy(self.pages[ref]),
                      'current_segment': self.segment}
        else:
            # IDs/bookmarks are exact; search returns only bounded previews.
            pages = [p for p in reversed(list(self.pages.values()))
                     if p['status'] == 'open' and (include_previous or p['segment'] == self.segment)
                     and query.casefold() in (p['title']+' '+p['text']+' '+
                                              json.dumps(p.get('data', {}))).casefold()]
            result = {'notes': [{k: p[k] for k in ('id', 'revision', 'kind', 'title', 'segment')}
                                | {'preview': p['text'][:120]} for p in pages[offset:offset+6]],
                      'next_offset': offset+6 if offset+6 < len(pages) else None}
        self.emit('notebook_read', reference=reference, query=query, offset=offset,
                  include_previous=include_previous, result=deepcopy(result))
        return result
