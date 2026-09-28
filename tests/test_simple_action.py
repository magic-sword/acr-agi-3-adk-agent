import json
from pathlib import Path
import tempfile
import unittest
from scripts.benchmark_simple_action import build_cases, measure, parse, prepare, render, request_for
from scripts.benchmark_recognition import decode


class SimpleActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=build_cases()

    @staticmethod
    def support(grid,color):
        return {(x,y) for y,row in enumerate(grid) for x,c in enumerate(row) if c==color}

    def test_movement_answers_against_raw_pixel_support(self):
        counts={}
        for c in self.cases:
            if c['group']!='motion':continue
            t=c['trials'][0]
            changes=[self.support(t['before'],col)!=self.support(t['after'],col) for col in [9,12]]
            expected={(True,False):'blue only',(False,True):'orange only',(True,True):'both blue and orange',(False,False):'neither'}[tuple(changes)]
            self.assertEqual(c['truth_text'],expected)
            counts[expected]=counts.get(expected,0)+1
            for col,changed in zip([9,12],changes):
                if not changed:continue
                b=self.support(t['before'],col);a=self.support(t['after'],col)
                self.assertEqual(len(b),len(a))
                dx=min(x for x,y in a)-min(x for x,y in b);dy=min(y for x,y in a)-min(y for x,y in b)
                self.assertEqual(abs(dx)+abs(dy),c['magnitude'])
        self.assertEqual(set(counts.values()),{4})

    def test_panel_and_binding_truth_independent_of_extractor(self):
        names=['upper-left panel','upper-right panel','lower-left panel','lower-right panel']
        for c in self.cases:
            if c['group']=='panel':
                t=c['trials'][0];regions=set()
                for y in range(64):
                    for x in range(64):
                        if t['before'][y][x]!=t['after'][y][x]:regions.add((0 if y<32 else 2)+(0 if x<32 else 1))
                self.assertEqual(c['truth_text'],names[next(iter(regions))] if regions else 'none')
                self.assertLessEqual(len(regions),1)
            elif c['group']=='binding':
                actions={t['action'] for t in c['trials'] if self.support(t['before'],12)!=self.support(t['after'],12)}
                expected='both A and B' if len(actions)==2 else next(iter(actions))+' only' if actions else 'neither A nor B'
                self.assertEqual(c['truth_text'],expected)

    def test_measured_changes_match_raw_pixels_and_are_complete(self):
        for c in self.cases:
            for t,m in zip(c['trials'],measure(c)):
                diff=t['before']!=t['after']
                events=m['comparison']['changes']
                self.assertEqual(bool(events),diff)
                self.assertTrue(all(e['kind'] in ['moved','appearance_changed'] for e in events))
                self.assertEqual(len(m['before']['instances']),4 if c['group']=='panel' else 2)

    def test_inputs_have_expected_images_and_frozen_question(self):
        palette=json.loads(Path('outputs/instance-state-20260928-final/cases.json').read_text())[0]['palette']
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for c in self.cases:
                for t in c['trials']:
                    for frame in ['before','after']:render(t[frame],palette).save(out/f"{c['name']}-{t['id']}-{frame}.png")
                m=measure(c);questions=[]
                for arm in ['images','states','facts']:
                    p=request_for(c,m,arm,out);content=p['messages'][1]['content'];questions.append(content[-1])
                    images=[v for v in content if v['type']=='image_url']
                    self.assertEqual(len(images),(1 if c['after_only'] else 2*len(c['trials'])) if arm=='images' else 0)
                    self.assertNotIn('truth',json.dumps(p))
                    if arm=='images':
                        text=' '.join(v['text'] for v in content if v['type']=='text')
                        self.assertNotIn('top-left (',text);self.assertNotIn('dx=',text)
                self.assertEqual(questions[0],questions[1]);self.assertEqual(questions[1],questions[2])

    def test_parser_does_not_treat_explanatory_text_as_single_answer(self):
        c=self.cases[0]
        self.assertEqual(parse(' B.\n',c),'B')
        self.assertEqual(parse('X',c),'X')
        for s in ['A or B','The answer is A','', 'E']:
            with self.assertRaises(ValueError):parse(s,c)

    def test_factual_parser_accepts_only_exact_letter_and_own_option(self):
        from scripts.report_simple_action import reported_choice
        for c in self.cases:
            for o in c['options']:
                self.assertEqual(reported_choice(o['letter']+'. '+o['text'],c),(o['letter'],'exact_option_label'))
                wrong=next(x['letter'] for x in c['options'] if x['letter']!=o['letter'])
                self.assertEqual(reported_choice(wrong+'. '+o['text'],c),(None,'unparseable'))

    def test_label_matching_does_not_guess_from_free_explanation(self):
        from scripts.report_simple_action import reported_choice
        c=self.cases[0]
        for text in ['A because it is blue','The answer is A. blue','blue','A. blue or orange']:
            self.assertEqual(reported_choice(text,c),(None,'unparseable'))


if __name__=='__main__':unittest.main()
