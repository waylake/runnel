#!/usr/bin/env python3
"""Run the native two-stage MoE probe on selected real Ornith layers.

Weights are exported temporarily from safetensors and removed/overwritten by
later layers; only hashes and measurements are written to the result JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,10,20,30,39")
    parser.add_argument("--experts", default="3,17,29,42,50,61,7,55")
    parser.add_argument("--bits", choices=["q4", "q8"], required=True)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--threads", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",")]
    exporter = Path(__file__).with_name("export_ornith_moe_layer.py")
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="runnel-ornith-") as temporary:
        temp_dir = Path(temporary)
        for layer in layers:
            dataset = temp_dir / f"layer-{layer}.bin"
            manifest = temp_dir / f"layer-{layer}.manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(exporter),
                    "--model",
                    str(args.model),
                    "--layer",
                    str(layer),
                    "--experts",
                    args.experts,
                    "--output",
                    str(dataset),
                    "--manifest",
                    str(manifest),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            native_output = temp_dir / f"layer-{layer}.json"
            with native_output.open("w") as output_stream:
                subprocess.run(
                    [
                        str(args.binary),
                        "--bits",
                        args.bits,
                        "--total-experts",
                        "256",
                        "--experts",
                        args.experts,
                        "--groups",
                        str(args.groups),
                        "--threads",
                        str(args.threads),
                        "--warmups",
                        str(args.warmups),
                        "--trials",
                        str(args.trials),
                        "--two-stage",
                        "--dataset",
                        str(dataset),
                    ],
                    check=True,
                    stdout=output_stream,
                )
            native_record = json.loads(native_output.read_text())
            manifest_record = json.loads(manifest.read_text())
            records.append(
                {
                    "layer": layer,
                    "manifest": manifest_record,
                    "native": native_record,
                }
            )

    document = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "engine": "Runnel native Metal two-stage MoE",
            "model_name": args.model.name,
            "bits": args.bits,
            "layers": layers,
            "experts": args.experts,
            "groups": args.groups,
            "threads": args.threads,
            "warmups": args.warmups,
            "trials": args.trials,
            "weights_included": False,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
