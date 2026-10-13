import pytest
import torch

# Parity tests run in fp32 on every device available. fp16 can flip near-tie
# tokens between devices, so it is reserved for performance runs.
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    return request.param
