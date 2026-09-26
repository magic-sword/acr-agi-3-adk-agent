"""Append-only evidence/interpretations and bounded single-token memory navigation.

Host entries record measurements; model entries remain attributed interpretations.
Source links mean 'available when written', not that the source proves a claim.
"""
from copy import deepcopy
import json
import time

CATEGORIES = {
    'experience': ('Actions, outcomes and their interpretations', {'outcome','review'}),
    'interpretations': ('Objects, relations, goals and causal hypotheses', {'scene','hypothesis','goals'}),
    'questions': ('Unresolved questions', {'question'}),
    'procedures': ('Reusable procedures', {'procedure'}),
}
PAGE_SIZE = 4
MAX_SELECTED = 4
MAX_CHARS = 9000
READ_SECONDS = 4.0


def append_note(memory, observation_id, kind, title, body, *, author, sources=()):
    if any(key not in memory.notes for key in sources):
        raise ValueError('unknown memory source')
    memory.note_sequence += 1
    key = f'n{memory.note_sequence}'
    entry = {'id':key,'kind':kind,'title':title[:180], 'author':author,
             'observation_id':observation_id,'episode':memory.episode,
             'sources':list(dict.fromkeys(sources)), 'body':deepcopy(body)}
    memory.notes[key] = entry
    return deepcopy(entry)


def start_reader(work, observation_id, question, eligible):
    return {'for_work':work,'observation_id':observation_id,'question':question,
            'eligible':list(eligible),'path':['all' if len(eligible)<=PAGE_SIZE else 'root'],
            'page':0,'selected':[], 'calls':0,'end_reason':None}


def current_ids(reader, notes):
    location=reader['path'][-1]
    if location=='selected':return list(reader['selected'])
    remaining=[key for key in reversed(reader['eligible']) if key not in reader['selected']]
    if location=='all':return remaining
    if location in CATEGORIES:
        return [key for key in remaining if notes[key]['kind'] in CATEGORIES[location][1]]
    return []


def reader_context(reader, notes):
    location=reader['path'][-1]
    return {'for_work':reader['for_work'],'question':reader['question'],
            'path':reader['path'],'page':reader['page'],
            'opened_record':deepcopy(notes[location]) if location in notes else None,
            'selected':[{'id':key,'kind':notes[key]['kind'],'title':notes[key]['title']} for key in reader['selected']]}


def reader_options(reader, notes):
    location=reader['path'][-1];options={}
    if location=='root':
        categories=[(key,title) for key,(title,kinds) in CATEGORIES.items()
                    if any(notes[n]['kind'] in kinds for n in reader['eligible'] if n not in reader['selected'])]
        options={str(i+1):{'kind':'open_memory','key':key,'title':title} for i,(key,title) in enumerate(categories)}
    elif location in notes:
        size=sum(len(json.dumps(notes[n],ensure_ascii=False)) for n in reader['selected'])
        if location in reader['selected'] or (len(reader['selected'])<MAX_SELECTED and
                size+len(json.dumps(notes[location],ensure_ascii=False))<=MAX_CHARS):
            options['7']={'kind':'toggle_memory','key':location,
                          'meaning':'Remove from working set' if location in reader['selected'] else 'Keep for the next deliberation'}
    else:
        ids=current_ids(reader,notes)
        start=reader['page']*PAGE_SIZE
        for i,key in enumerate(ids[start:start+PAGE_SIZE]):
            note=notes[key]
            options[str(i+1)]={'kind':'open_memory','key':key,'title':note['title'],
                              'record_kind':note['kind'],'author':note['author'],'episode':note['episode'],
                              'observation_id':note['observation_id']}
        if len(ids)>PAGE_SIZE:
            options['6']={'kind':'memory_page','page':(reader['page']+1)%((len(ids)+PAGE_SIZE-1)//PAGE_SIZE),
                          'meaning':'Next page (wraps to first)'}
    if len(reader['path'])>1:options['5']={'kind':'memory_back','meaning':'Back to previous list'}
    if reader['selected'] and location not in notes and location!='selected':
        options['7']={'kind':'open_memory','key':'selected','title':'Review/remove selected records'}
    options['8']={'kind':'finish_memory','meaning':'Hand the selected records to deliberation; missing knowledge can be inferred or probed there'}
    return options


def navigate(reader, option):
    kind=option['kind']
    if kind=='open_memory':reader['path'].append(option['key']);reader['page']=0
    elif kind=='memory_back':reader['path'].pop();reader['page']=0
    elif kind=='memory_page':reader['page']=option['page']
    elif kind=='toggle_memory':
        key=option['key']
        if key in reader['selected']:reader['selected'].remove(key)
        else:reader['selected'].append(key)
        reader['path'].pop();reader['page']=0
    elif kind!='finish_memory':raise ValueError('invalid memory navigation')


class NotebookRuntime:
    def _write_note(self, kind, title, body, *, author, sources=()):
        entry=append_note(self.memory,self.obs['observation_id'],kind,title,body,author=author,sources=sources)
        self._record('artifacts','memory_written',record=entry)
        return entry['id']

    def _write_stage_notes(self, work, value):
        """Archive accepted outputs without granting model writes to outcome entries."""
        m=self.memory;data=value.model_dump()
        sources=list(dict.fromkeys([*m.memory_brief.get('selected',[]),*sorted(self._mandatory_note_ids(work))]))
        ids=[]
        def write(kind,title,body):
            ids.append(self._write_note(kind,title,body,author=work,sources=sources))
        if work=='understand':
            write('scene',value.observed,{k:data[k] for k in ('concepts','targets','observed','goal_hypothesis')})
            if value.causal_hypotheses:write('hypothesis',value.causal_hypotheses,{'hypothesis':value.causal_hypotheses})
            if value.question:write('question',value.question,{'question':value.question})
        elif work=='reconcile':
            write('review',value.evidence,data)
        elif work=='backchain':
            write('goals',value.rationale,data)
        elif work=='ground' and value.next=='execute':
            for skill in value.plan.skills:
                write('procedure',skill.name+': '+skill.when_to_use,skill.model_dump())
        elif work=='ground':
            write('question',m.handoff_question,{'question':m.handoff_question,'reason':value.reason})
        m.stage_notes[work]=ids

    def _mandatory_note_ids(self, work):
        m=self.memory
        ids=set(m.stage_notes.get('reconcile',[]))
        if m.last_outcome_id:ids.add(m.last_outcome_id)
        if work!='understand':ids.update(m.stage_notes.get('understand',[]))
        if work in ('backchain','ground'):ids.update(m.stage_notes.get('backchain',[]))
        if work=='reconcile' and m.active_skill:
            skill=m.skills[m.active_skill['name']]
            ids.update(k for k,v in m.notes.items() if v['kind']=='procedure' and v['body']==skill)
        return ids

    def _selected_skills(self):
        m=self.memory
        return {note['body']['name']:deepcopy(note['body']) for key in m.memory_brief.get('selected',[])
                for note in [m.notes[key]] if note['kind']=='procedure'}

    async def _read_memory(self, work):
        m=self.memory
        eligible=[key for key in m.notes if key not in self._mandatory_note_ids(work)]
        m.reader=start_reader(work,self.obs['observation_id'],m.handoff_question,eligible)
        m.memory_brief={}
        started=time.monotonic()
        deadline=started+min(READ_SECONDS,self.time_left()/5)
        reason='no_optional_records'
        if eligible:
            m.phase='read_memory'
            self._machine_transition('read_for_'+work)
            reason='read_time_budget'
            while self.time_left()>0 and time.monotonic()<deadline:
                m.reader['calls']+=1
                choice=await self._fast('read_memory',reader_options(m.reader,m.notes),
                                        timeout=min(self.time_left(),deadline-time.monotonic()))
                if choice is None:
                    reason='read_time_budget' if time.monotonic()>=deadline else 'unreadable_selection'
                    break
                if choice['kind']=='finish_memory':reason='selected';break
                navigate(m.reader,choice)
                self._record('artifacts','memory_navigation',reader=reader_context(m.reader,m.notes))
                self._machine_transition('memory_navigation')
                self._snapshot()
            m.phase=work
            self._machine_transition('memory_to_'+work)
        m.reader['end_reason']=reason
        m.memory_brief={'for_work':work,'observation_id':self.obs['observation_id'],
                        'question':m.handoff_question,'selected':list(m.reader['selected']),
                        'end_reason':reason}
        self._record('artifacts','memory_prepared',brief=m.memory_brief,
                     records=[m.notes[key] for key in m.reader['selected']],seconds=time.monotonic()-started,
                     calls=m.reader['calls'])
        self._snapshot()
