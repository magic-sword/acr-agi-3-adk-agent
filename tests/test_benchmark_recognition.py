import unittest
from copy import deepcopy
from PIL import Image

from scripts.benchmark_recognition import crop_sheet, decode, encode, extract


class RecognitionBenchmarkTests(unittest.TestCase):
    def test_crops_preserve_pixels_and_cover_borders(self):
        image=Image.new('RGB',(428,452),'black')
        for x,y,color in [(0,0,'red'),(63,0,'blue'),(0,63,'green'),(63,63,'yellow')]:
            image.paste(color,(32+x*6,44+y*6,38+x*6,50+y*6))
        part=encode(image);before=deepcopy(part);sheet=decode(crop_sheet(part))
        for x,y,color in [(6,26,(255,0,0)),(737,26,(0,0,255)),
                          (6,777,(0,128,0)),(737,777,(255,255,0))]:
            self.assertEqual(sheet.getpixel((x,y)),color)
        self.assertEqual(part,before)

    def test_wrong_observation_and_truncation_cannot_pass(self):
        schema={'type':'object','properties':{'observation_id':{'type':'string','enum':['current']}},
                'required':['observation_id'],'additionalProperties':False}
        payload={'tools':[{'function':{'parameters':schema}}]}
        response={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
            {'function':{'name':'submit_understanding','arguments':'{"observation_id":"old"}'}}]}}]}
        with self.assertRaisesRegex(ValueError,'enum mismatch'):extract(response,payload)
        response['choices'][0]['finish_reason']='length'
        with self.assertRaisesRegex(ValueError,'truncated'):extract(response,payload)


if __name__=='__main__':unittest.main()
