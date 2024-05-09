import unittest

import torch

import bwb.distributions.utils as utils
from bwb import _logging

_log = _logging.get_logger(__name__)


def get_generator(seed: int, device: torch.device) -> torch.Generator:
    """
    Function that returns a generator with a seed and device.

    :param seed: The seed of the generator.
    :param device: The device of the generator.
    :return: A generator with the seed and device.
    """
    return torch.Generator(device=device).manual_seed(seed)


class UtilsTest(unittest.TestCase):
    def setUp(self):
        _logging.set_level(_logging.DEBUG)
        self.seed = torch.Generator().seed()
        _log.info(f"Seed: {self.seed}")

    def test_grayscale_parser_valid_input(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        shape = (28, 28)
        weights = torch.rand((784,), device='cpu', generator=gen)
        support = torch.randint(0, 28, (784, 2), device='cpu', generator=gen)
        result = utils.grayscale_parser(shape, weights, support)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(result.device.type, 'cpu')

    def test_grayscale_parser_invalid_shape(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        shape = (30, 30)
        weights = torch.rand((784,), device='cpu', generator=gen)
        support = torch.randint(0, 28, (784, 2), device='cpu', generator=gen)
        with self.assertRaises(ValueError):
            utils.grayscale_parser(shape, weights, support)

    def test_partition_valid_input(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        X = torch.rand((784, 2), device='cpu', generator=gen)
        mu = torch.rand((784,), device='cpu', generator=gen)
        alpha = 0.5
        X_new, mu_new = utils.partition(X, mu, alpha)
        self.assertTrue(X_new.shape[0] >= X.shape[0])
        self.assertTrue(mu_new.shape[0] >= mu.shape[0])
        self.assertTrue(torch.isclose(torch.sum(mu_new), torch.tensor(1.0)))

    def test_partition_invalid_alpha(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        X = torch.rand((784, 2), device='cpu', generator=gen)
        mu = torch.rand((784,), device='cpu', generator=gen)
        alpha = -0.5
        with self.assertRaises(ValueError):
            utils.partition(X, mu, alpha)

    def test_grayscale_parser_valid_input_cpu_float32(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        shape = (28, 28)
        weights = torch.rand((784,), dtype=torch.float32,
                             device='cpu', generator=gen)
        support = torch.randint(0, 28, (784, 2), dtype=torch.int32,
                                device='cpu', generator=gen)
        result = utils.grayscale_parser(shape, weights, support)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(result.device.type, 'cpu')

    def test_grayscale_parser_valid_input_gpu_float64(self):
        if torch.cuda.is_available():
            gen = get_generator(self.seed, torch.device('cuda'))
            shape = (28, 28)
            weights = torch.rand((784,), dtype=torch.float64,
                                 device='cuda', generator=gen)
            support = torch.randint(0, 28, (784, 2), dtype=torch.int32,
                                    device='cuda', generator=gen)
            result = utils.grayscale_parser(shape, weights, support)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(result.device.type, 'cuda')

    def test_partition_valid_input_cpu_float32(self):
        gen = get_generator(self.seed, torch.device('cpu'))
        X = torch.rand((784, 2), dtype=torch.float32,
                       device='cpu', generator=gen)
        mu = torch.rand((784,), dtype=torch.float32,
                        device='cpu', generator=gen)
        alpha = 0.5
        X_new, mu_new = utils.partition(X, mu, alpha)
        self.assertTrue(X_new.shape[0] >= X.shape[0])
        self.assertTrue(mu_new.shape[0] >= mu.shape[0])
        self.assertTrue(torch.isclose(torch.sum(mu_new), torch.tensor(1.0)))

    def test_partition_valid_input_gpu_float64(self):
        if torch.cuda.is_available():
            gen = get_generator(self.seed, torch.device('cuda'))
            X = torch.rand((784, 2), dtype=torch.float64,
                           device='cuda', generator=gen)
            mu = torch.rand((784,), dtype=torch.float64,
                            device='cuda', generator=gen)
            alpha = 0.5
            X_new, mu_new = utils.partition(X, mu, alpha)
            self.assertTrue(X_new.shape[0] >= X.shape[0])
            self.assertTrue(mu_new.shape[0] >= mu.shape[0])
            self.assertTrue(torch.isclose(
                torch.sum(mu_new),
                torch.tensor(1.0, device='cuda', dtype=torch.float64)
            ))
