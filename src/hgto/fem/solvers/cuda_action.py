"""Capture one fixed linear preconditioner; never capture the changing operator."""

import torch


class CapturedAction:
    def __init__(self, action, example):
        if not example.is_cuda:
            raise ValueError("CUDA action requires a CUDA tensor")
        self.input = torch.zeros_like(example)
        stream = torch.cuda.Stream(device=example.device)
        current = torch.cuda.current_stream(example.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                action(self.input)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, stream=stream):
            self.output = action(self.input)
        current.wait_stream(stream)

    def __call__(self, residual):
        self.input.copy_(residual)
        self.graph.replay()
        return self.output.clone()
