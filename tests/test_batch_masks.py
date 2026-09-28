import unittest
from PIL import Image,ImageDraw
from scripts.benchmark_batch_masks import bounds,mask_image,render,background,masked_fraction,payload


class BatchMaskTests(unittest.TestCase):
    def test_half_open_bounds_and_overlapping_union(self):
        self.assertEqual(bounds([0,0,1000,1000]),(0,0,384,384))
        self.assertEqual(bounds([1,2,3,4]),(0,0,2,2))
        mask=mask_image([[0,0,500,500],[250,250,750,750]])
        self.assertEqual(sum(v>0 for v in mask.getdata()),192*192*2-96*96)
        self.assertEqual(mask.getpixel((288,288)),0)

    def test_fill_and_mosaic_do_not_change_unmasked_area_or_original(self):
        im=Image.new('RGB',(384,384),'black');d=ImageDraw.Draw(im)
        d.rectangle((100,100,140,140),fill='red');d.rectangle((270,270,280,280),fill='blue')
        before=im.tobytes();boxes=[[200,200,500,500]]
        self.assertEqual(background(im),(0,0,0))
        filled=render(im,boxes,'fill');mosaic=render(im,boxes,'mosaic')
        self.assertEqual(filled.getpixel((120,120)),(0,0,0))
        for modified in (filled,mosaic):
            self.assertEqual(modified.crop((220,220,384,384)).tobytes(),im.crop((220,220,384,384)).tobytes())
        self.assertEqual(im.tobytes(),before)
        self.assertNotEqual(mosaic.tobytes(),before)
        self.assertEqual(render(im,boxes+boxes,'fill').tobytes(),filled.tobytes())

    def test_hidden_target_is_measured_separately_from_retrieval(self):
        g=dict(support=[[x,y] for x in range(3) for y in range(3)])
        self.assertAlmostEqual(masked_fraction([[0,0,1000/64,1000/64]],g),1/9)
        self.assertEqual(masked_fraction([[0,0,3*1000/64,3*1000/64]],g),1)

    def test_treatment_changes_no_decoding_or_schema(self):
        im=Image.new('RGB',(384,384),'black')
        pp=[payload(im,a,'additional') for a in ('outline','mosaic','fill')]
        for p in pp:
            self.assertEqual(p['max_tokens'],2048)
            self.assertFalse(p['cache_prompt'])
            self.assertIn('ALL additional',p['messages'][1]['content'][1]['text'])
        self.assertEqual(pp[0]['messages'][1],pp[1]['messages'][1])
        for a in ('outline','mosaic','fill'):
            p=payload(im,a,'gate');self.assertEqual(p['max_tokens'],1);self.assertIn('"Y" | "N"',p['grammar'])


if __name__=='__main__':unittest.main()
