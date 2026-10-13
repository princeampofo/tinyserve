"""Start the server: `python -m tinyserve [--device mps] [--max-batch-size 8] ...`"""

import argparse
import logging

import torch
import uvicorn
from transformers import AutoTokenizer

from tinyserve.api.server import create_app
from tinyserve.engine.engine import Engine
from tinyserve.model.weights import DEFAULT_MODEL, load_model

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(prog="tinyserve")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", choices=DTYPES, help="default: float16 on a GPU, float32 on CPU")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-wait-ms", type=float, default=50, help="how long to wait to fill a batch"
    )
    parser.add_argument("--max-model-len", type=int, default=4096, help="prompt + output tokens")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # fp16 halves memory traffic on a GPU, but CPUs run fp16 matmuls slowly.
    dtype = (
        DTYPES[args.dtype]
        if args.dtype
        else (torch.float32 if args.device == "cpu" else torch.float16)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config, weights = load_model(args.model, args.device, dtype)
    engine = Engine(
        weights, config, tokenizer, args.max_batch_size, batch_wait_s=args.batch_wait_ms / 1000
    )
    engine.start()

    app = create_app(engine, tokenizer, args.model, args.max_model_len)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
