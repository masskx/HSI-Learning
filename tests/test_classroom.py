import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hsi_learning.classroom import classification_summary,spatial_split,threshold_experiment,svm_pipeline,overlap_fraction

class ClassroomTests(unittest.TestCase):
    def test_absent_class_is_json_null(self):
        import json
        report=classification_summary([1,1,2],[1,2,2],[1,2,3])
        self.assertIsNone(report['recall'][2])
        self.assertEqual(report['support'],[2,1,0])
        self.assertAlmostEqual(report['aa'],.75)
        json.dumps(report,allow_nan=False)
        with self.assertRaises(ValueError): classification_summary([1],[4],[1,2,3])
    def test_labels_follow_ids(self):
        r=classification_summary([1,2,2,3],[1,1,2,3],[3,1,2])
        self.assertEqual(r['support'],[1,1,2]); self.assertEqual(r['recall'],[1,1,.5])
    def test_spatial_windows_do_not_overlap(self):
        gt=np.ones((13,50),dtype=int); seen=[]
        for ids in spatial_split(gt,9).values():
            pixels=set()
            for r,c in zip(*np.unravel_index(ids,gt.shape)):
                pixels.update((rr,cc) for rr in range(max(0,r-4),min(13,r+5)) for cc in range(max(0,c-4),min(50,c+5)))
            seen.append(pixels)
        self.assertTrue(all(seen)); self.assertFalse(seen[0]&seen[1] or seen[0]&seen[2] or seen[1]&seen[2])
    def test_threshold_exercise(self):
        args=([1,1,3,3],[1,2,1,2],{'msp':np.array([18.5,0,18.5,25])},[1,2],{'msp':np.arange(20.)})
        hi,a,_=threshold_experiment(*args,acceptance=.95)
        lo,b,_=threshold_experiment(*args,acceptance=.8)
        self.assertLess(lo['msp'],hi['msp'])
        self.assertGreater(b['msp']['unknown_recall'],a['msp']['unknown_recall'])
        self.assertGreater(b['msp']['known_false_rejection'],a['msp']['known_false_rejection'])
        self.assertEqual(lo,threshold_experiment([1,2,4,4],*args[1:],acceptance=.8)[0])
    def test_pipeline_training_only(self):
        x=np.array([[0.,1],[1,2],[10,11],[11,12]])
        m=svm_pipeline().fit(x,[1,1,2,2]); scaler=m.named_steps['standardscaler']
        m.predict([[9999,9999]]); np.testing.assert_array_equal(scaler.mean_,x.mean(0))
    def test_overlap_is_pairwise(self):
        self.assertAlmostEqual(overlap_fraction((0,1)),72/81); self.assertEqual(overlap_fraction((9,0)),0)
if __name__=='__main__': unittest.main()
