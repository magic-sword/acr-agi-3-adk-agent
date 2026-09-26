"""Evidence-grounded, provisional objects and questions; never a game-rule oracle.

Connected components are geometric candidates, including background/HUD pixels.
Semantic groupings are replaceable. Exact or uniquely overlapping same-color
regions retain a provisional identity; this is not a physical-identity proof.
"""
from copy import deepcopy
from hashlib import sha256
import json
from agent.controls import ACTION_TO_BUTTON

COLOR_NAMES = ('white','light gray','gray','dark gray','very dark gray','black',
               'magenta','pink','red','blue','light blue','yellow','orange','maroon','green','purple')


def identity(prefix, value):
    return prefix + sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def components(grid):
    seen, result = set(), []
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if (x, y) in seen:
                continue
            pending, cells = [(x, y)], []
            seen.add((x, y))
            while pending:
                a, b = pending.pop(); cells.append((a, b))
                for u, v in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if (0 <= v < len(grid) and 0 <= u < len(row) and
                            (u,v) not in seen and grid[v][u] == color):
                        seen.add((u,v)); pending.append((u,v))
            cells.sort()
            xs, ys = zip(*cells)
            result.append({'id':identity('c-', [color,cells]), 'color':color,'color_name':COLOR_NAMES[color],
                'area':len(cells), 'region':{'x':min(xs),'y':min(ys),
                    'width':max(xs)-min(xs)+1,'height':max(ys)-min(ys)+1}, 'cells':cells})
    return result


class WorldMemory:
    def __init__(self, notebook):
        self.notebook = notebook
        self.segment = notebook.segment
        self.observation_id = None
        self.interpreted_id = None
        self.candidates, self.objects, self.questions = {}, {}, {}
        self.relations, self.findings = [], []
        self.bindings = {}
        self.lineage = {}
        self.omitted = 0
        self.legal_actions = []
        self.component_sequence = 0

    def observe(self, obs):
        if self.segment != self.notebook.segment:
            self.__init__(self.notebook)
        if self.observation_id == obs['observation_id']:
            return
        self.observation_id = obs['observation_id']
        self.legal_actions = [ACTION_TO_BUTTON[a] for a in obs.get('available_actions', [])
                              if a in ACTION_TO_BUTTON and a!='RESET']
        all_candidates = components(obs.get('grid') or [])
        # Bounded context, evenly sampled over scan order, including tiny regions.
        # Never silently pretend this subset describes every pixel or object.
        count = len(all_candidates)
        selected = all_candidates if count <= 64 else [all_candidates[i*count//64] for i in range(64)]
        previous = self.candidates
        matches = []
        for c in selected:
            cells=set(c['cells'])
            matches.append([oid for oid,old in previous.items() if c['color']==old['color'] and
                len(cells & set(old['cells'])) / len(cells | set(old['cells'])) >= .8])
        assignments = {}
        for index,c in enumerate(selected):
            possible=matches[index]
            if len(possible)==1 and sum(possible[0] in other for other in matches)==1:
                oid=possible[0]
                match='exact_pixels' if c['cells']==previous[oid]['cells'] else 'overlap_hypothesis'
            else:
                self.component_sequence += 1
                oid=f'c-{self.component_sequence}'
                match='new_or_ambiguous'
            assignments[oid]={**c,'id':oid,'correspondence':match}
        self.candidates = assignments
        self.omitted = count-len(selected)
        for obj in self.objects.values():
            obj['visible'] = set(obj['component_ids']) <= self.candidates.keys()

    def candidate_view(self):
        return [{k:v for k,v in c.items() if k != 'cells'} for c in self.candidates.values()]

    def validate(self, answer):
        keys = [o.key for o in answer.objects]
        if len(keys) != len(set(keys)) or 'scene' in keys:
            raise ValueError('object keys must be unique and cannot be scene')
        for o in answer.objects:
            if not set(o.component_ids) <= self.candidates.keys():
                raise ValueError('objects must reference current component_ids')
            if not set(o.previous_ids) <= self.objects.keys():
                raise ValueError('previous_ids must reference supplied prior objects')
            if o.previous_ids and not o.identity_reason.strip():
                raise ValueError('revising object identity requires an identity_reason')
        for r in answer.relations:
            if r.subject not in keys or r.other not in keys or r.subject == r.other:
                raise ValueError('relations must join distinct supplied object keys')
        for q in answer.questions:
            if q.subject not in keys + ['scene'] or not set(q.observe+q.context) <= set(keys):
                raise ValueError('questions must reference supplied object keys or scene')
            if q.action not in self.legal_actions:
                raise ValueError('question action must be a currently legal control')
            if len(set(q.alternatives)) < 2:
                raise ValueError('a question needs distinct possible answers')

    def apply(self, answer):
        self.validate(answer)
        aliases, objects = {}, {}
        for o in answer.objects:
            refs = sorted(set(o.component_ids))
            oid = identity('object-', [self.segment,refs])
            aliases[o.key] = oid
            objects[oid] = {'id':oid, 'component_ids':refs, 'description':o.description,
                'possible_roles':o.possible_roles, 'status':'hypothesis', 'visible':True,
                'previous_ids':o.previous_ids, 'identity_reason':o.identity_reason,
                'components':[{k:v for k,v in self.candidates[c].items() if k!='cells'} for c in refs],
                'observation_id':self.observation_id}
            self.lineage[oid] = set(self.lineage.get(oid,())) | {p for p in o.previous_ids if p!=oid}
        self.objects = objects  # Full replaceable interpretation, not immutable classes.
        self.lineage = dict(list(self.lineage.items())[-256:])
        self.relations = [{'subject':aliases[r.subject], 'other':aliases[r.other],
            'description':r.description, 'status':'hypothesis', 'observation_id':self.observation_id}
            for r in answer.relations]
        proposed = {}
        for q in answer.questions:
            subject = aliases.get(q.subject, 'scene')
            observed = sorted(set(aliases[k] for k in q.observe))
            conditions = sorted(set(aliases[k] for k in q.context))
            signature = [self.segment,subject,observed,conditions,q.property,q.action]
            qid = identity('question-',signature)
            prior = self.questions.get(qid, {})
            proposed[qid] = {'id':qid, 'subject':subject, 'observe':observed,
                'context':conditions,'action':q.action,
                'property':q.property, 'question':q.question, 'alternatives':q.alternatives,
                'status':prior.get('status','open'), 'current':True,
                'finding_ids':prior.get('finding_ids',[])}
        # Evidence survives rewording and omission/reintroduction of a question.
        for qid, q in proposed.items():
            known = [f['experiment_id'] for f in self.findings if f['question_id']==qid and f['resolved_test']]
            if known:
                q.update(status='tested',finding_ids=known)
        historical = {k:{**q,'current':False} for k,q in list(self.questions.items())[-64:] if k not in proposed}
        self.questions = {**historical,**proposed}
        self.interpreted_id = self.observation_id
        self.save()

    def question(self, qid):
        if qid not in self.questions:
            raise ValueError('select a supplied question_id; do not invent a question')
        return deepcopy(self.questions[qid])

    def object(self, oid):
        obj = self.objects.get(oid)
        if not obj or not obj['visible']:
            raise ValueError('target_object_id must identify a current grounded object')
        result = deepcopy(obj)
        result['components'] = [{k:v for k,v in self.candidates[c].items() if k!='cells'}
                                for c in obj['component_ids']]
        return result

    def contains(self, oid, x, y):
        obj = self.objects.get(oid)
        if not obj or not obj['visible']:
            raise ValueError('target_object_id must identify a current grounded object')
        return any((x,y) in self.candidates[c]['cells'] for c in obj['component_ids'])

    def overlaps(self, oid, region):
        obj = self.object(oid)
        return any(region.x <= x < region.x+region.width and region.y <= y < region.y+region.height
                   for c in obj['component_ids'] for x,y in self.candidates[c]['cells'])

    def view(self, qid=None):
        questions = [q for q in self.questions.values() if q['current']]
        objects, relations, findings = list(self.objects.values()), self.relations, self.findings[-12:]
        if qid and qid in self.questions:
            q = self.questions[qid]; questions = [q]
            ids = {q['subject'],*q['observe'],*q['context']}
            if q['subject']=='scene':
                ids.update(self.objects)
            objects = [o for o in objects if o['id'] in ids]
            relations = [r for r in relations if r['subject'] in ids or r['other'] in ids]
            ancestors=set(ids)
            for _ in range(4):
                ancestors.update(p for oid in list(ancestors) for p in self.lineage.get(oid,()))
            findings = [f for f in self.findings if f['question_id']==qid or
                        any(o['id'] in ancestors for o in f['objects'])][-4:]
        else:
            questions = sorted(questions,key=lambda q:q['status']!='open')
        brief = [{k:f[k] for k in ('experiment_id','question_id','action','conditions','expected',
                   'measurement','observed_change','verdict','finding','target_grounding','evidence_ids','qualification')}
                 for f in findings]
        return deepcopy({'objects':objects,'relations':relations,'questions':questions,
            'conditional_findings':brief,'omitted_components':self.omitted,
            'uninterpreted_components':len(self.candidates.keys()-{c for o in self.objects.values() for c in o['component_ids']}),
            'qualification':'Objects/roles/relations are provisional. Tested means a scoped trial, not a universal rule. Missing objects and global/temporal effects remain possible.'})

    def bind(self, experiment_id, qid, target_id, obs):
        q = self.question(qid)
        self.bindings[experiment_id] = {'question_id':qid,'question':q,
            'target_object':self.object(target_id) if target_id else None,
            'objects':self.view(qid)['objects'], 'before_id':obs['observation_id'],
            'target_grounding':'pixel_mask' if target_id else 'not_applicable'}

    def record(self, page):
        binding = self.bindings.pop(page['id'], None)
        if binding is None:
            return
        d=page['data']; outcome=d['outcome']; review=d['review']
        resolved = bool(outcome['acknowledged'] and not outcome.get('boundary') and
                        review['verdict'] in ('supported','unsupported'))
        fact = {**binding, 'experiment_id':page['id'], 'action':d['action'],
            'conditions':d['plan']['conditions'], 'expected':d['plan']['expected'],
            'measurement':d.get('measurement'), 'verdict':review['verdict'],
            'observed_change':{k:deepcopy(outcome[k]) for k in (
                'acknowledged','boundary','frame_changed','changed_cell_count','changed_cells') if k in outcome},
            'finding':review['finding'], 'after_id':d['after_id'],
            'evidence_ids':review['evidence_ids'], 'resolved_test':resolved,
            'qualification':'Observed outcome under these conditions; no universal affordance or causal law inferred.'}
        self.findings.append(fact)
        # Full history is versioned on disk. Keep a bounded working memory.
        self.findings=self.findings[-64:]
        q=self.questions.get(binding['question_id'])
        if q and resolved:
            q['status']='tested'; q['finding_ids'].append(page['id'])
        self.save()

    def save(self):
        self.notebook._commit('world','world','対象・関係・未解決の疑問',
            '画素に基づく対象候補と暫定解釈。操作結果は条件付きで保持する。',
            [self.observation_id] if self.observation_id else [], author='host',
            data={**self.view(),'observation_id':self.observation_id,
                  'candidates':self.candidate_view(),'conditional_findings':deepcopy(self.findings)})
