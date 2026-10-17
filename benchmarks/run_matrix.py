"""Run direct-call backends in separate processes, in both orders.

Runner config: [{"id":"native", "backend":"native", "command":["eg2"],
                 "args":["--bundle","artifacts/my-gpu"]}, ...].
Commands must share the working directory/output mount when using containers.
"""

import argparse, json, subprocess, time, os, signal
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--output", default="benchmarks/results/local")
p.add_argument("--budgets", nargs="+", type=int, default=[70, 140, 280])
p.add_argument("--batches", nargs="+", type=int, default=[1, 4, 32, 128])
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--timeout", type=int, default=1800)
a = p.parse_args()
if a.warmup < 3 or a.repeat < 5:
    p.error("Published runs require warmup>=3 and repeat>=5")
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
runners = json.loads(Path(a.config).read_text())
assert len({r["id"] for r in runners}) == len(runners)
for budget in a.budgets:
    for batch in a.batches:
        for round_, order in enumerate((runners, runners[::-1])):
            for runner in order:
                path = root / f"{runner['id']}-t{budget}-b{batch}-r{round_}.json"
                if path.exists():
                    continue
                if batch in runner.get("skip_batches", []):
                    path.write_text(
                        json.dumps(
                            dict(
                                status="validation_failed",
                                budget=budget,
                                batch=batch,
                                backend=runner["backend"],
                                runner_id=runner["id"],
                                round=round_,
                                reason=runner["skip_reason"],
                            ),
                            indent=2,
                        )
                    )
                    continue
                cmd = runner["command"] + [
                    "benchmark",
                    "--backend",
                    runner["backend"],
                    *runner.get("args", []),
                    "--budget",
                    str(budget),
                    "--batch",
                    str(batch),
                    "--warmup",
                    str(a.warmup),
                    "--repeat",
                    str(a.repeat),
                    "--output",
                    str(path),
                ]
                cid = path.with_suffix(".cid")
                cid.unlink(missing_ok=True)
                if len(cmd) >= 2 and Path(cmd[0]).name == "docker" and cmd[1] == "run":
                    cmd[2:2] = ["--cidfile", str(cid.resolve())]
                print("run", runner["id"], budget, batch, round_, flush=True)
                start = time.time()
                try:
                    with path.with_suffix(".log").open("w") as log:
                        process = subprocess.Popen(
                            cmd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            process.wait(timeout=a.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                        result = process
                    if not path.exists():
                        path.write_text(
                            json.dumps(
                                dict(
                                    status="process_error",
                                    returncode=result.returncode,
                                    budget=budget,
                                    batch=batch,
                                    backend=runner["backend"],
                                ),
                                indent=2,
                            )
                        )
                    elif result.returncode:
                        record = json.loads(path.read_text())
                        record["process_returncode"] = result.returncode
                        if record.get("status") == "ok":
                            record["status"] = "process_error"
                        path.write_text(json.dumps(record, indent=2))
                except subprocess.TimeoutExpired:
                    path.write_text(
                        json.dumps(
                            dict(
                                status="timeout",
                                budget=budget,
                                batch=batch,
                                backend=runner["backend"],
                                timeout=a.timeout,
                            ),
                            indent=2,
                        )
                    )
                finally:
                    if cid.exists():
                        container = cid.read_text().strip()
                        if container:
                            subprocess.run(
                                ["docker", "rm", "-f", container],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=30,
                            )
                        cid.unlink(missing_ok=True)
                record = json.loads(path.read_text())
                record.update(
                    runner_id=runner["id"],
                    round=round_,
                    wall_seconds=time.time() - start,
                )
                path.write_text(json.dumps(record, indent=2))
