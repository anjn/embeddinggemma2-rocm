"""Command line entry points; optional preparation/server dependencies load on demand."""

import argparse, json, os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="eg2")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--output")
    for name in ("prepare", "prepare-text"):
        c = sub.add_parser(name)
        c.add_argument("--model", required=True)
        c.add_argument("--output", required=True)
        c.add_argument("--batches", nargs="+", type=int, default=[1, 4, 32, 128])
        c.add_argument(
            "--precision",
            choices=["standard", "fast"],
            default="fast" if name == "prepare" else "standard",
        )
        if name == "prepare":
            c.add_argument("--budgets", nargs="+", type=int, default=[70, 140, 280])
            c.add_argument("--aspect", nargs=2, type=int, default=[16, 9])
            c.add_argument("--vision-batch", type=int)
        else:
            c.add_argument(
                "--lengths", nargs="+", type=int, default=[32, 128, 512, 2048, 8192]
            )
    c = sub.add_parser("info")
    c.add_argument("--bundle", required=True)
    c.add_argument("--device", type=int, default=0)
    c = sub.add_parser("serve")
    c.add_argument("--bundle", required=True)
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--max-pending", type=int, default=32)
    c = sub.add_parser("benchmark")
    c.add_argument("--backend", choices=["native", "torch", "llama"], required=True)
    c.add_argument("--bundle")
    c.add_argument("--model")
    c.add_argument("--llama-library")
    c.add_argument("--mmproj")
    c.add_argument("--budget", type=int, choices=[70, 140, 280], required=True)
    c.add_argument("--batch", type=int, choices=[1, 4, 32, 128], required=True)
    c.add_argument("--warmup", type=int, default=3)
    c.add_argument("--repeat", type=int, default=5)
    c.add_argument("--output", required=True)
    c.add_argument(
        "--attention",
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Vision attention backend; text attention stays eager for correctness",
    )
    c.add_argument(
        "--torch-precision", choices=["fp32", "mixed", "bf16"], default="bf16"
    )
    c.add_argument("--flash", choices=["on", "off"], default="on")
    c.add_argument("--vision-batch", type=int)
    c = sub.add_parser("tune")
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--budget", type=int, choices=[70, 140, 280], default=70)
    c.add_argument("--batch", type=int, choices=[1, 4, 32, 128], default=128)
    c.add_argument("--vision-batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    c.add_argument("--repeat", type=int, default=5)
    a = p.parse_args(argv)
    if a.command == "fetch":
        from .preparation import fetch_model

        print(fetch_model(a.output))
    elif a.command in ("prepare", "prepare-text"):
        if any(b < 1 or b > 128 for b in a.batches):
            p.error("batches must be 1..128")
        os.environ.setdefault("MIGRAPHX_DISABLE_MLIR", "0")
        os.environ.setdefault("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "attention")
        from .preparation import compile_images, prepare_text

        if a.command == "prepare":
            if any(b not in (70, 140, 280) for b in a.budgets) or min(a.aspect) < 1:
                p.error("Invalid budget or aspect ratio")
            print(
                compile_images(
                    a.model,
                    a.output,
                    a.budgets,
                    a.batches,
                    tuple(a.aspect),
                    a.precision,
                    a.vision_batch,
                )
            )
        else:
            print(prepare_text(a.model, a.output, a.batches, a.lengths, a.precision))
    elif a.command == "info":
        from .runtime import Embedder

        with Embedder(a.bundle, a.device) as model:
            print(json.dumps(model.capabilities, indent=2))
    elif a.command == "serve":
        import uvicorn
        from .server import create_app

        uvicorn.run(
            create_app(a.bundle, a.device, max_pending=a.max_pending),
            host=a.host,
            port=a.port,
        )
    elif a.command == "benchmark":
        from .benchmark import run

        run(a)
    elif a.command == "tune":
        from .tuning import tune

        tune(a)


if __name__ == "__main__":
    main()
