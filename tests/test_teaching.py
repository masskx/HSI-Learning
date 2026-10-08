import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hsi_learning.teaching import (EpisodeSampler, PatchClassifier, calibrate_threshold,
    fit_preprocessing, grad_cam, prototype_logits, prototypes, rejection_predictions, transform_cube)
from hsi_learning.teaching_runs import META_CLASSES


class TeachingTests(unittest.TestCase):
    def test_class_disjoint(self):
        a,b,c = map(set,META_CLASSES.values())
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(a|b|c,set(range(1,17)))

    def test_episodes(self):
        y=np.repeat([1,2,3],20); ids=np.arange(len(y))
        a=EpisodeSampler(y,ids,[1,2,3],42,15)
        b=EpisodeSampler(y,ids,[1,2,3],42,15)
        val=EpisodeSampler(y,ids,[1,2,3],1042,15)
        for _ in range(3):
            ea=a.sample(3,5,5)
            val.sample(3,5,5)
            eb=b.sample(3,5,5)
            np.testing.assert_array_equal(ea.support_ids,eb.support_ids)
            self.assertFalse(set(ea.support_ids)&set(ea.query_ids))
            np.testing.assert_array_equal(y[ea.support_ids],ea.classes[ea.support_y])
        with self.assertRaises(ValueError): a.sample(5,5,5)
        with self.assertRaises(ValueError): a.sample(3,19,2)

    def test_prototypes_shuffle(self):
        x=torch.tensor([[0.,0.],[2.,0.],[8.,0.],[10.,0.]])
        y=torch.tensor([0,0,1,1]); order=torch.tensor([3,0,2,1])
        torch.testing.assert_close(prototypes(x,y,[0,1]),prototypes(x[order],y[order],[0,1]))
        logits=prototype_logits(x,y,torch.tensor([[1.,0.],[4.,0.]]),2)
        torch.testing.assert_close(logits,torch.tensor([[0.,-64.],[-9.,-25.]]))

    def test_threshold(self):
        val=np.arange(100.)
        before=calibrate_threshold(val)
        test=np.array([0.,999.]); test[:]=99  # cannot influence calibrator inputs
        self.assertEqual(before,calibrate_threshold(val))
        np.testing.assert_array_equal(rejection_predictions([2,3],[0,99],before,17),[2,17])
        self.assertEqual(roc_auc_score([0,0,1,1],[0,1,4,5]),1)

    def test_preprocessing_roundtrip(self):
        x=np.random.default_rng(42).normal(size=(8,8,5)).astype('float32')
        z,p=fit_preprocessing(x,np.arange(20),3)
        np.testing.assert_allclose(z,transform_cube(x,p))
        np.testing.assert_allclose(z.reshape(-1,3)[:20].mean(0),0,atol=1e-5)

    def test_cam(self):
        torch.manual_seed(42); model=PatchClassifier(3,2)
        layer=model.encoder.features[7]
        before=len(layer._forward_hooks)
        cam, logits=grad_cam(model,layer,torch.randn(2,3,9,9))
        self.assertEqual(cam.shape,(2,9,9)); self.assertEqual(logits.shape,(2,2))
        self.assertTrue(torch.isfinite(cam).all()); self.assertTrue((cam>=0).all())
        self.assertEqual(len(layer._forward_hooks),before)
        self.assertTrue(model.training)

    def test_cam_is_class_conditioned(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1, bias=False)
                with torch.no_grad():
                    self.conv.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))

            def forward(self, x):
                return self.conv(x).mean(dim=(-2, -1))

        model = Toy()
        x = torch.zeros(1, 2, 3, 3)
        x[0, 0, 0, 0] = 1
        x[0, 1, 2, 2] = 1
        first, _ = grad_cam(model, model.conv, x, torch.tensor([0]))
        second, _ = grad_cam(model, model.conv, x, torch.tensor([1]))
        self.assertEqual(int(first.flatten().argmax()), 0)
        self.assertEqual(int(second.flatten().argmax()), 8)
        self.assertFalse(torch.equal(first, second))
        hooks = len(model.conv._forward_hooks)
        with self.assertRaises(RuntimeError):
            grad_cam(model, model.conv, x, torch.tensor([8]))
        self.assertEqual(len(model.conv._forward_hooks), hooks)

    def test_preprocessor_ignores_heldout_values(self):
        x = np.random.default_rng(7).normal(size=(8, 8, 5)).astype('float32')
        ids = np.arange(20)
        _, first = fit_preprocessing(x, ids, 3)
        changed = x.copy()
        changed.reshape(-1, 5)[20:] += 1000
        _, second = fit_preprocessing(changed, ids, 3)
        for key in first:
            np.testing.assert_allclose(first[key], second[key])

    def test_receptive_fields_include_pooling(self):
        def rf(layers):
            size, jump = 1, 1
            for kernel, stride in layers:
                size += (kernel - 1) * jump
                jump *= stride
            return size
        self.assertEqual(rf([(7,1),(2,2),(5,1),(2,2),(3,1)]),26)
        self.assertEqual(rf([(3,1),(2,2),(3,1),(3,1)]),12)

    def test_kappa_majority(self):
        self.assertEqual(cohen_kappa_score([0]*900+[1]*90+[2]*10,[0]*1000),0)


if __name__=='__main__': unittest.main()
