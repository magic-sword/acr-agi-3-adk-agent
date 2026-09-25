"""Readable notes preserve what was actually presented, with no future revisions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.agent_monitor import Timeline, Run
from scripts.notebook_view import notebook_html


def page(text, revision=1, kind='plan', **extra):
    return {'id':'note-1', 'revision':revision, 'kind':kind, 'title':'操作の仮説',
            'text':text, 'status':'open', 'segment':0, 'author':'agent', **extra}


def opening():
    root=page('ステージをクリアする', kind='goal', id='goal')
    sub=page('操作できる対象を探す', kind='subgoal', id='subgoal-1',
             data={'parent_id':'goal','done_when':'反応する対象が見つかる','status':'active'})
    return {'segment':0, 'goal':root, 'goal_path':[root, sub],
            'active_experiment':None, 'latest_review':None, 'latest_result':None,
            'bookmarks':[{'name':'clue','id':'note-1','revision':1,'title':'操作の仮説'}]}


class ReadableNotebookTests(unittest.TestCase):
    def timeline(self, events):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        for index, (journal, row) in enumerate(events, 1):
            with (root/f'r.{journal}.jsonl').open('a') as f:
                f.write(json.dumps(dict(row, sequence=index))+'\n')
        return Timeline(Run(root,'r')).load()

    def test_opening_reads_and_writes_remain_distinct_and_keep_received_revision(self):
        view=opening()
        old=page('最初は一度だけクリックする')
        new=page('繰り返さず別の対象を試す',2)
        t=self.timeline([
            ('observations',{'event':'observation_received','step':0}),
            ('states',{'event':'state_entered','state':'DECIDE','work':'experiment_design','step':0,'input':{'notebook':view}}),
            ('notebook',{'event':'notebook_opened','step':0,'work':'experiment_design','view':view}),
            ('notebook',{'event':'notebook_read','step':0,'reference':'clue','result':{'page':old}}),
            ('notebook',{'event':'note_changed','step':0,'before':old,'after':new}),
            ('states',{'event':'state_entered','state':'RUN','step':0,'input':{}}),
            ('states',{'event':'state_exited','state':'RUN','step':0,'output':{'notebook':dict(view,goal=new)}}),
            ('observations',{'event':'observation_received','step':1}),
        ])
        before=notebook_html(t.snapshot(2))
        self.assertIn('LLM呼び出し開始時の提示内容',before)
        self.assertIn('索引のみ・本文の参照記録なし',before)
        self.assertNotIn(old['text'],before)  # An index entry never invents a read.
        self.assertNotIn(new['text'],before)
        read=notebook_html(t.snapshot(3))
        self.assertIn('本文を追加参照',read)
        self.assertIn(old['text'],read)
        self.assertNotIn(new['text'],read)
        changed=notebook_html(t.snapshot(6))
        self.assertIn('第1版 → 第2版',changed)
        self.assertIn('変更前',changed)
        self.assertIn(new['text'],changed)
        self.assertEqual(t.snapshot(6)['notebook']['opening']['goal']['text'],'ステージをクリアする')
        self.assertEqual(t.snapshot(6)['notebook']['reads'][0]['result']['page']['revision'],1)
        following=notebook_html(t.snapshot(7))
        self.assertNotIn(old['text'],following)
        self.assertNotIn(new['text'],following)
        self.assertIsNone(t.snapshot(7)['notebook']['opening'])

    def test_a_new_job_and_model_invocation_clear_prior_reads_but_preserve_step_changes(self):
        view=opening();note=page('前の呼び出しだけで読んだ内容')
        t=self.timeline([
            ('notebook',{'event':'notebook_opened','step':0,'work':'experiment_design','view':view}),
            ('notebook',{'event':'notebook_read','step':0,'reference':'note-1','result':{'page':note}}),
            ('notebook',{'event':'note_changed','step':0,'before':None,'after':page('このステップの書き込み')}),
            ('states',{'event':'state_entered','state':'DECIDE','work':'skill_creation','step':0,'input':{'notebook':view}}),
            ('notebook',{'event':'notebook_opened','step':0,'work':'skill_creation','view':view}),
            ('notebook',{'event':'notebook_read','step':0,'result':{'page':note}}),
            ('notebook',{'event':'notebook_opened','step':0,'work':'skill_creation','view':view}),
        ])
        prepared=notebook_html(t.snapshot(3))
        self.assertIn('LLMへの提示記録はまだありません',prepared)
        self.assertNotIn(note['text'],prepared)
        self.assertIn('このステップの書き込み',prepared)
        self.assertEqual(t.snapshot(6)['notebook']['reads'],())
        self.assertEqual(len(t.snapshot(5)['notebook']['reads']),1)

    def test_index_results_and_summary_pages_do_not_claim_full_reads(self):
        view=opening()
        view['latest_review']={'id':'experiment-1','revision':3,'segment':0,
            'question':'この場所は反応するか','expected':{'description':'対象が動く'},
            'data':{'review':{'verdict':'unsupported','finding':'対象は動かなかった',
                            'update':'別の対象を調べる','subgoal_status':'active','next_step':'revise'},
                    'measurement':{'changed_cells_in_region':0},'reviewer':'host'}}
        view['bookmarks'].append({'name':'latest_review','id':'experiment-1','revision':3,'title':'最新の判定'})
        t=self.timeline([
            ('notebook',{'event':'notebook_opened','step':0,'view':view}),
            ('notebook',{'event':'notebook_read','step':0,'query':'操作','result':{
                'notes':[{'id':'note-1','revision':1,'title':'操作の仮説','preview':'ここは抜粋'}],'next_offset':6}}),
        ])
        html=notebook_html(t.snapshot(1))
        for text in ('要点のみ提示','索引を検索（本文の参照ではありません）','ここは抜粋','不支持','別の対象を調べる'):
            self.assertIn(text,html)
        self.assertNotIn('本文を追加参照',html)
        self.assertNotIn('"changed_cells_in_region"',html)
        self.assertIn('範囲内で変わったセル数',html)

        # A full active page and its abbreviated verdict can share an ID/version.
        # Render once and keep the stronger, full-content visibility label.
        view['active_experiment'] = deepcopy(view['latest_review'])
        view['active_experiment']['kind'] = 'experiment'
        t=self.timeline([('notebook',{'event':'notebook_opened','step':0,'view':view})])
        html=notebook_html(t.snapshot(0))
        self.assertEqual(html.count('対象は動かなかった'),1)
        self.assertIn('本文を提示',html)
        self.assertNotIn('要点のみ提示',html)

    def test_model_html_is_escaped_in_titles_text_bookmarks_reads_and_changes(self):
        bad='<img src=x onerror=alert(1)>'
        view=opening();view['goal_path'][0]['text']=bad
        view['bookmarks'][0]['name']=bad
        note=page(bad,title=bad)
        t=self.timeline([
            ('notebook',{'event':'notebook_opened','step':0,'view':view}),
            ('notebook',{'event':'notebook_read','step':0,'reference':bad,'result':{'page':note}}),
            ('notebook',{'event':'note_changed','step':0,'before':None,'after':note}),
        ])
        html=notebook_html(t.snapshot(2))
        self.assertNotIn('<img',html)
        self.assertIn('&lt;img',html)
        self.assertNotIn('<pre',html)
        self.assertNotIn('"evidence_ids"',html)

    def test_jupyter_reader_opens_on_demand_and_never_uses_raw_json_renderer(self):
        from scripts.notebook_monitor import BenchmarkReplay
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);evaluation=root/'sample';folder=evaluation/'game/cognition';folder.mkdir(parents=True)
            (evaluation/'manifest.json').write_text('{}')
            (folder.parent/'result.json').write_text('{}')
            view=opening()
            (folder/'r.states.jsonl').write_text(json.dumps({'sequence':1,'event':'state_entered',
                'step':0,'state':'DECIDE','work':'experiment_design','input':{'notebook':view}})+'\n')
            (folder/'r.notebook.jsonl').write_text(json.dumps({'sequence':2,'event':'notebook_opened','step':0,'view':view})+'\n')
            w=BenchmarkReplay(root);self.addCleanup(w.close)
            self.assertTrue(w.notes_button.disabled)
            with patch('scripts.notebook_monitor.notebook_html',wraps=notebook_html) as render:
                w.load();render.assert_not_called()
                self.assertFalse(w.notes_button.disabled)
                with patch('scripts.notebook_monitor.json_html',side_effect=AssertionError('raw JSON rendered')):
                    w.notes_button.click()
                    self.assertEqual(w.details.selected_index,6)
                    self.assertIn('目標の道筋',w.panels[6].value)
                    w.slider.value=w.slider.max
                w.details.selected_index=None
                count=render.call_count
                w.slider.value=0
                self.assertEqual(render.call_count,count)
