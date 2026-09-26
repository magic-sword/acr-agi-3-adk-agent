"""Check replay integrity and overlap measurement without a model server."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.benchmark_parallel import restore_images, measure, report, response_signature, complete


class ParallelBenchmarkTests(unittest.TestCase):
    def test_image_replay_verifies_bytes_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'image.bin').write_bytes(b'image')
            payload={'messages':[{'content':[{'type':'image_url','image_url':{
                'path':'image.bin','sha256':hashlib.sha256(b'image').hexdigest(),'mime_type':'image/png'}}]}]}
            actual=restore_images(payload,root)
            self.assertIn('data:image/png;base64,',actual['messages'][0]['content'][0]['image_url']['url'])
            self.assertNotIn('url',payload['messages'][0]['content'][0]['image_url'])
            (root/'image.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):restore_images(payload,root)

    def test_overlap_runs_independent_streams_but_preserves_reader_order(self):
        barrier=threading.Barrier(2);seen=[]
        case={'name':'example','kind':'action','foreground':'foreground','background':['first','second']}
        def complete(base,payload,slot,origin,cache):
            if payload in ('foreground','first'):barrier.wait(timeout=2)
            seen.append((payload,slot))
            return {'start':0 if payload!='second' else 1,'end':2 if payload=='second' else 1,'seconds':1}
        with patch('scripts.benchmark_parallel.complete',complete):r=measure('url',case,2,'overlap')
        self.assertEqual([p for p,s in seen if s==1],['first','second'])
        self.assertIn(('foreground',0),seen)
        self.assertEqual(r['pair_seconds'],2)
        self.assertEqual(r['background_wait_after_foreground'],1)

    def test_report_excludes_warmup_and_distinguishes_latency_from_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cases.json').write_text(json.dumps([{'name':'case'}]))
            for slots in (1,2):
                rows=[]
                for mode in (['serial'] if slots==1 else ['serial','overlap']):
                    for warmup in (True,False):
                        rows.append({'case':'case','mode':mode,'slots':slots,'warmup':warmup,
                            'pair_seconds':100 if warmup else 10 if mode=='serial' else 8,
                            'foreground_seconds':4 if mode=='serial' else 5,'background_seconds':6,
                            'background_wait_after_foreground':6 if mode=='serial' else 3,
                            'foreground':{'response_signature':'same','usage':{'completion_tokens':20}}})
                (root/f'measurements-{slots}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch('builtins.print'):report(root)
            summary=json.loads((root/'summary.json').read_text())[0]
            self.assertEqual(summary['speedup_from_overlap'],1.25)
            self.assertEqual(summary['foreground_slowdown'],1.25)
            self.assertEqual(summary['arms']['2-overlap']['n'],1)

    def test_signature_ignores_call_ids_and_json_argument_formatting(self):
        def response(call_id,args):return {'choices':[{'message':{'tool_calls':[
            {'id':call_id,'function':{'name':'submit','arguments':args}}]}}]}
        self.assertEqual(response_signature(response('a','{"x": 1}')),
                         response_signature(response('b','{"x":1}')))

    def test_rejects_cache_off_profile_if_server_still_reuses_kv(self):
        response={'timings':{'cache_n':20},'choices':[{'message':{'content':'1'}}]}
        with patch('scripts.benchmark_parallel.request_json',return_value=response):
            with self.assertRaisesRegex(ValueError,'server reused prompt KV'):
                complete('url',{'max_tokens':1,'grammar':'root ::= "1"'},0,0,cache=False)
