"""Small CPU regressions; no accelerator is needed by the test suite."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
