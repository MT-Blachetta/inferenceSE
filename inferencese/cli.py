"""Command line interface: ``inferencese <command>`` or ``python -m inferencese``."""
from __future__ import annotations

import argparse
import json
import sys
import time

import yaml

from .profile.loader import list_profiles, set_path

GiB = float(1 << 30)


def _overrides(args) -> dict:
    """--set a.b.c=value pairs (YAML values) plus the convenience flags."""
    ov: dict = {}
    for item in getattr(args, "set", None) or []:
        key, _, val = item.partition("=")
        set_path(ov, key.strip(), yaml.safe_load(val))
    for flag, path in (("backend", "runtime.backend"), ("device", "runtime.device"),
                       ("vram", "memory.tiers.gpu.capacity"), ("ram", "memory.tiers.ram.capacity"),
                       ("policy", "memory.policy.name"), ("threads", "runtime.threads")):
        v = getattr(args, flag, None)
        if v is not None:
            set_path(ov, path, v)
    return ov


def _engine(args, **kw):
    from .runtime.engine import Engine
    return Engine(args.profile, args.model, _overrides(args),
                  log=(lambda *a: print(*a, file=sys.stderr)) if not args.quiet else None, **kw)


def cmd_profiles(args):
    from .profile.loader import Profile
    for name in list_profiles():
        p = Profile(name)
        print(f"{name:<18} {p.doc.get('description', '')}")


def cmd_hardware(args):
    from .profile.hardware import detect
    print(yaml.safe_dump(detect(args.path), sort_keys=False))


def cmd_inspect(args):
    from .formats import open_model
    cat = open_model(args.path)
    s = cat.summary()
    print(f"{s['tensors']:,} tensors, {s['bytes'] / GiB:,.2f} GiB in {s['files']} file(s)")
    for dt, v in s["dtypes"].items():
        print(f"  {dt:<6}{v['tensors']:>9,} tensors {v['bytes'] / GiB:>12,.3f} GiB")
    if args.tensors:
        for e in sorted(cat.tensors.values(), key=lambda e: (e.path, e.offset)):
            print(f"{e.name}  {e.dtype}{list(e.shape)}  {e.nbytes:,} B  @{e.offset}")
    if args.manifest:
        cat.save_manifest(args.manifest)
        print(f"manifest written to {args.manifest}")


def cmd_plan(args):
    from .tools.plan import format_plan, plan
    p = plan(args.profile, args.model, _overrides(args))
    print(json.dumps(p, indent=1) if args.json else format_plan(p))


def cmd_simulate(args):
    from .tools.plan import simulate
    print(f"{'zipf':>5}{'repeat':>8}{'SSD GiB/tok':>13}{'H2D GiB/tok':>13}{'expert hit':>12}"
          f"{'s/token':>9}{'tok/s':>8}")
    for z in args.zipf:
        r = simulate(args.profile, args.model, _overrides(args), steps=args.steps,
                     warm_steps=args.warm_steps, zipf=z, repeat=args.repeat, seed=args.seed)
        s = r["seconds_per_token"]
        hit = r["expert_hit_rate"]
        print(f"{z:>5.2f}{args.repeat:>8.2f}{r['ssd_read_gib_per_token']:>13.2f}"
              f"{r['host_to_gpu_gib_per_token']:>13.2f}{(hit if hit is not None else 0):>12.3f}"
              f"{s['total']:>9.2f}{r['tokens_per_second']:>8.3f}", flush=True)


def _progress(quiet):
    def cb(done, total):
        if not quiet:
            print(f"\r[prefill {done}/{total} tokens]", end="", file=sys.stderr, flush=True)
            if done >= total:
                print(file=sys.stderr)
    return cb


def cmd_run(args):
    eng = _engine(args)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "seed": args.seed}
    try:
        def answer(ids):
            show_reasoning = False
            for channel, piece in eng.stream_text(ids, max_new_tokens=args.max_tokens,
                                                  sampling=sampling, channels=args.chat,
                                                  on_progress=_progress(args.quiet)):
                if channel == "reasoning":
                    if args.hide_reasoning:
                        continue
                    if not show_reasoning:
                        print("[thinking] ", end="")
                        show_reasoning = True
                elif show_reasoning:
                    print("\n[answer] ", end="")
                    show_reasoning = False
                print(piece, end="", flush=True)
                yield channel, piece
            print()

        if args.interactive:
            messages = [{"role": "system", "content": args.system}] if args.system else []
            while True:
                try:
                    user = input("\n> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if user.strip() in ("/exit", "/quit"):
                    break
                if user.strip() == "/stats":
                    print(json.dumps(eng.stats(), indent=1))
                    continue
                messages.append({"role": "user", "content": user})
                ids = eng.tokenizer.apply_chat(messages, add_generation_prompt=True)
                reply = "".join(p for c, p in answer(ids) if c == "content")
                messages.append({"role": "assistant", "content": reply})
        else:
            prompt = args.prompt if args.prompt is not None else sys.stdin.read()
            if args.chat:
                messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
                    {"role": "user", "content": prompt}]
                ids = eng.tokenizer.apply_chat(messages, add_generation_prompt=True)
            else:
                ids = eng.prompt_ids(prompt)
            for _ in answer(ids):
                pass
        if args.stats:
            print(json.dumps(eng.stats(), indent=1), file=sys.stderr)
        elif not args.quiet:
            t = eng.stats()["timing"]
            m = eng.mem.stats()
            print(f"[prefill {t['prefill_tokens']} tok @ {t['prefill_tok_per_s']} tok/s | "
                  f"decode {t['decode_tokens']} tok @ {t['decode_tok_per_s']} tok/s | "
                  f"cache hit rate {m['hit_rate']} | SSD read {m['ssd']['bytes_read'] / GiB:.2f} GiB]",
                  file=sys.stderr)
    finally:
        eng.close()


def cmd_serve(args):
    from .server.api import serve
    eng = _engine(args)
    httpd = serve(eng, args.host, args.port, api_key=args.api_key, verbose=args.verbose)
    print(f"[inferenceSE] serving {eng.profile.name} on http://{args.host}:{args.port} "
          f"(OpenAI-compatible API under /v1, chat UI at /)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        eng.close()


def cmd_replica(args):
    from .tools.replica import make_replica
    info = make_replica(args.profile, args.out, fill_below=args.fill_below, seed=args.seed)
    print(f"sparse replica in {args.out}: {info['files']} files, apparent size "
          f"{info['apparent'] / GiB:,.1f} GiB, {info['written'] / 2**20:,.1f} MiB actually written")


def cmd_new_profile(args):
    from .profile.builder import create_profile
    path = create_profile(args.repo, args.name, args.out, revision=args.revision)
    print(f"profile skeleton written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inferencese", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        p.add_argument("profile", help="profile name or folder")
        if model:
            p.add_argument("-m", "--model", help="model folder or file (overrides the profile)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override any profile setting, e.g. memory.tiers.ram.capacity=64GiB")
        p.add_argument("--backend", choices=["auto", "numpy", "torch"])
        p.add_argument("--device")
        p.add_argument("--vram", help="VRAM cache size, e.g. 90GiB")
        p.add_argument("--ram", help="RAM cache size, e.g. 100GiB")
        p.add_argument("--policy", choices=["plan", "lru", "lfu"])
        p.add_argument("--threads", type=int)
        p.add_argument("-q", "--quiet", action="store_true")

    sub.add_parser("profiles", help="list installed optimization profiles").set_defaults(fn=cmd_profiles)
    p = sub.add_parser("hardware", help="show detected hardware")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(fn=cmd_hardware)
    p = sub.add_parser("inspect", help="list the tensors of a model (reads headers only)")
    p.add_argument("path")
    p.add_argument("--tensors", action="store_true")
    p.add_argument("--manifest", help="write a compact tensor index (.json.xz)")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("plan", help="show what the profile keeps in VRAM, RAM and on SSD")
    common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("simulate", help="estimate decode speed from the memory plan")
    common(p)
    p.add_argument("--zipf", type=float, nargs="+", default=[0.0, 0.6, 1.0, 1.3])
    p.add_argument("--repeat", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--warm-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("run", help="generate text")
    common(p)
    p.add_argument("-p", "--prompt")
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat format")
    p.add_argument("-i", "--interactive", action="store_true", help="interactive chat")
    p.add_argument("--system")
    p.add_argument("-n", "--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--hide-reasoning", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("serve", help="OpenAI-compatible HTTP server with a web chat UI")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--api-key")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("replica", help="build a sparse, data-free copy of a model's file layout")
    p.add_argument("profile")
    p.add_argument("out")
    p.add_argument("--fill-below", default="64KiB",
                   help="give tensors smaller than this random values (default 64KiB)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_replica)
    p = sub.add_parser("new-profile", help="scrape a Hugging Face repository into a profile skeleton")
    p.add_argument("repo", help="e.g. moonshotai/Kimi-K3.1-Pro")
    p.add_argument("name")
    p.add_argument("--out", default="profiles")
    p.add_argument("--revision", default="main")
    p.set_defaults(fn=cmd_new_profile)

    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        args.fn(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
