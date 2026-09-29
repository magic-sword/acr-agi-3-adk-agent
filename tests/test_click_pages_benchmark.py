"""Coverage and boundedness of exhaustive candidate paging."""
import unittest
from scripts.benchmark_click_pages import pages


class ClickPagesTests(unittest.TestCase):
    def test_every_candidate_is_offered_exactly_once(self):
        for count in (0, 1, 8, 9, 18, 29, 83):
            candidates=[{'id':str(i)} for i in range(count)]
            for order in (0,1):
                batches=pages(candidates,order)
                ids=[o['id'] for batch in batches for o in batch]
                self.assertEqual(sorted(ids),sorted(o['id'] for o in candidates))
                self.assertEqual(len(ids),len(set(ids)))
                self.assertTrue(all(1<=len(batch)<=8 for batch in batches))
                self.assertEqual(candidates,[{'id':str(i)} for i in range(count)])

    def test_worst_case_tournament_is_bounded(self):
        # 83 candidates: 11 first pages, two semifinal pages, one final page.
        current=[{'id':str(i)} for i in range(83)];calls=0
        while current:
            batches=pages(current,0);calls+=len(batches)
            winners=[batch[0] for batch in batches]
            if len(winners)<=1:break
            self.assertLess(len(winners),len(current));current=winners
        self.assertEqual(calls,14)


if __name__=='__main__':unittest.main()
