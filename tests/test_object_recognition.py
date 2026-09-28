import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_object_recognition import make_cases, render, request, PROMPT
from scripts.preview_change_emphasis import masks, rgb


class ObjectRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The synthetic fixtures do not depend on ignored measurement artifacts.
        palette={'0':'#ffffff','1':'#cccccc','3':'#666666','4':'#333333',
                 '8':'#f93c31','9':'#1e93ff','11':'#ffdc00','12':'#ff851b'}
        grid=[[3]*64 for _ in range(64)]
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'cases.json'
            source.write_text(json.dumps([{'name':name,'before':grid,'after':grid,'palette':palette}
                                          for name in ('forward','reverse','same_frame')]))
            cls.cases=make_cases(source)

    def test_counterfactual_changes_only_intended_object(self):
        for c in self.cases:
            if c['kind']!='synthetic': continue
            changed,_=masks(c['before'],c['after'])
            colors={c[k][y][x] for x,y in changed for k in ('before','after')}
            expected={'synthetic_block':{3,12,9},'synthetic_cross':{3,0},'synthetic_recolor':{3,8,11},'synthetic_static':set()}
            self.assertEqual(colors,expected[c['name']])

    def test_annotated_changed_pixels_retain_rgb(self):
        for c in self.cases:
            _,mask=masks(c['before'],c['after'])
            for k in ('before','after'):
                im=render(c[k],c['palette'],mask)
                for x,y in mask:
                    for dx,dy in [(0,0),(5,5),(2,3)]:
                        self.assertEqual(im.getpixel((16+x*6+dx,16+y*6+dy)),rgb(c['palette'][str(c[k][y][x])]))
            if not mask:
                self.assertEqual(render(c['before'],c['palette']).tobytes(),render(c['after'],c['palette'],mask).tobytes())

    def test_no_answer_leak_and_matching_budgets(self):
        for word in ('orange','blue','cross','player','up action'):
            self.assertNotIn(word,PROMPT.lower())
        a,b=request([],False,0),request([],True,0)
        self.assertEqual(a['messages'],b['messages'])
        self.assertEqual(a['max_tokens'],b['max_tokens'])
        self.assertEqual((b['temperature'],b['top_p'],b['top_k'],b['min_p'],b['presence_penalty']),(.7,.8,20,0,1.5))


if __name__=='__main__': unittest.main()
