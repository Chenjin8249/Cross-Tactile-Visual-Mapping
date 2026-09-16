"""Small loss tests and full U-Net meta shape test; no training data required."""
import ast
from pathlib import Path
import unittest
import torch
from model import Model
from losses import Loss, compute_joint


class Checks(unittest.TestCase):
    def test_loss_supervises_both_modalities(self):
        torch.manual_seed(42)
        x, y = torch.zeros(2, 1601), torch.zeros(2, 1, 256, 256)
        px, py = torch.ones_like(x, requires_grad=True), torch.full_like(y, 0.5, requires_grad=True)
        zx = torch.randn(2, 512, 1, 1, requires_grad=True)
        zy = torch.randn(2, 512, 1, 1, requires_grad=True)
        out = Loss()(zx, zy, px, py, x, y)
        self.assertAlmostEqual(out['mse_loss_S'].item(), 1.0)
        self.assertAlmostEqual(out['mse_loss_T'].item(), 0.25)
        torch.testing.assert_close(out['loss'], 0.2*out['crossview_contrastive_Loss'] + 0.5)
        out['loss'].backward()
        for tensor in [px, py, zx, zy]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_full_unet_shape_and_branch_graph(self):
        # Meta checks graph and tensor shapes without allocating the full model.
        with torch.device('meta'):
            model = Model()
            z1, z2, px, py = model(torch.empty(2, 1601), torch.empty(2, 1, 256, 256))
            objective = (px**2).mean() + (py**2).mean()
            objective.backward()
        self.assertEqual(tuple(px.shape), (2,1601))
        self.assertEqual(tuple(py.shape), (2,1,256,256))
        self.assertEqual(tuple(z1.shape), (2,512,1,1))
        self.assertEqual(tuple(z2.shape), (2,512,1,1))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_joint_is_valid_for_signed_and_zero_latents(self):
        for z in [torch.randn(2,512,1,1), torch.zeros(2,512,1,1)]:
            joint = compute_joint(z, z)
            self.assertTrue(torch.isfinite(joint).all())
            self.assertTrue((joint > 0).all())
            self.assertAlmostEqual(joint.sum().item(),1.0,places=5)

    def test_optimizer_factory(self):
        module=ast.parse(Path(__file__).with_name('utils.py').read_text())
        fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='configure_optimizers')
        namespace={'optim':torch.optim}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'factory','exec'),namespace)
        from types import SimpleNamespace
        optimizer=namespace['configure_optimizers'](torch.nn.Linear(2,2),SimpleNamespace(learning_rate=5e-4))
        self.assertIsInstance(optimizer,torch.optim.Adam)
        self.assertEqual(optimizer.defaults['betas'],(0.9,0.999))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
