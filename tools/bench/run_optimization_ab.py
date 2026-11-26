#!/usr/bin/env python3
"""Replay and compare the Qwen3.8-27B NVFP4 5090D optimization workloads.

`run` executes a selected workload or suite on a fresh, isolated Serve process.
Start Serve separately with the same model, runtime flags, and JSONL logging.
Use a fresh process for each sample and for each different `--case` selection.
Compare fixed/FP32 to fixed/FP32 first, then fixed to adaptive on the same
revision, then FP32 to BF16 with the speculation policy held constant.
Use five independent rounds for performance decisions. The ledger case covers
low acceptance only if its fixed-K7 baseline measures at most 70% acceptance
and emits at least 512 tokens. Keep the native ReplaySSM, perplexity, and
release recovery gates alongside the HTTP cases.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen3.8-27b-nvfp4"
ASYMMETRIC = "p32768-p65536-c2"
LEDGER = "free-decode-ledger"
NEEDLES = {"retrieval-p32768": 32768, "retrieval-p65536": 65536}
SUPPLEMENTAL = (ASYMMETRIC, LEDGER, *NEEDLES)
ENGINE_KEYS = ("max_context", "max_concurrency", "prefill_chunk", "kv_cache", "vision",
               "cuda_graph", "prefix_reuse", "proposal_head", "speculative_backend",
               "speculative_draft_window", "host_mapped_embedding",
               "host_mapped_dflash2_codebooks", "host_pinned_vision_weights",
               "log_stats_interval_ms", "context_capacity_mode")


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration(startup, model_sha256):
    engine = startup["engine"]
    environment = startup["environment"]
    schema = startup["schema_version"]
    return {
        "model_sha256": model_sha256,
        "public_model_id": startup["server"]["public_model_id"],
        "hardware": {key: environment[key] for key in (
            "gpu_name", "gpu_uuid", "cuda_driver_version", "cuda_runtime_version")},
        "engine": {key: engine[key] for key in ENGINE_KEYS},
        "host_capacity_bytes": engine["context_cache"]["host_capacity_bytes"],
        "sampling_defaults": startup["sampling_defaults"],
        # Schema 32 predates the switches and uses FP32 state and fixed draft width.
        "linear_state_dtype": "fp32" if schema == 32 else engine["linear_state_dtype"],
        "speculative_length_policy": "fixed" if schema == 32 else engine["speculative_length_policy"],
    }


def setup_evidence(startup):
    return {key: startup["engine"][key] for key in (
        "context_pools", "context_token_capacity_estimate")}


def matched_fixtures():
    """Reproduce the byte-exact text and JSON fields of the November 21 release cases."""
    fixed = " ".join(str(i) for i in range(128))
    question = ("Return a JSON string containing exactly the integers 0 through 127 "
                "in ascending order, separated by one space, with no extra text.")

    def lane(identity, padding, depth, *, constrained):
        payload = {
            "model": MODEL, "max_tokens": 1024 if constrained else 64, "stream": True,
            "temperature": 0, "thinking": {"type": "disabled"},
            "system": [{"type": "text", "text": identity + " x" * padding,
                        "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
            "messages": [{"role": "user", "content": question if constrained else "Reply OK."}],
        }
        if constrained:
            payload["output_config"] = {"format": {"type": "json_schema",
                                                   "schema": {"type": "string", "enum": [fixed]}}}
        return {"protocol": "anthropic_messages", "path": "/v1/messages", "payload": payload,
                "expected_prompt_tokens": depth, "expected_output": fixed if constrained else "OK",
                "json_output": constrained}

    cases = [{"id": "ordinary-cold-p2048-c1",
              "requests": [lane("Ordinary-cold-2048.", 2018, 2048, constrained=False)]}]
    for depth, padding in ((2048, 1990), (8192, 8134), (32768, 32709), (65536, 65477)):
        for concurrency in (1, 2):
            cases.append({"id": f"p{depth}-c{concurrency}", "requests": [
                lane(f"Matched-{depth}-{concurrency}-{i}.", padding, depth, constrained=True)
                for i in range(concurrency)]})
    return cases


def fixture(case_id, archived):
    by_id = {case["id"]: case for case in archived}
    if case_id in by_id:
        return by_id[case_id]
    if case_id == ASYMMETRIC:
        lanes = [by_id["p32768-c2"]["requests"][0], by_id["p65536-c2"]["requests"][1]]
        return {"id": case_id, "requests": lanes}
    if case_id == LEDGER:
        lane = deepcopy(by_id["ordinary-cold-p2048-c1"]["requests"][0])
        rows = [{"id": i, "rack": digest(["rack", i])[:9],
                 "ticket": digest(["ticket", i])[:11]} for i in range(1, 49)]
        expected = [{"id": row["id"], "key": f"{row['rack']}:{row['ticket']}"}
                    for row in rows]
        lane["payload"].update(
            max_tokens=2500,
            system="Benchmark inventory reconciliation. Follow the output format exactly.",
            messages=[{"role": "user", "content":
                       "For every input row, return a JSON object with its integer id and a key "
                       "formed as rack:ticket. Keep input order, include every row, and return "
                       "only the JSON array, with no Markdown.\n" + json.dumps(rows)}],
        )
        lane.update(expected_output=expected, json_output=True, expected_prompt_tokens=None)
    else:
        lane = deepcopy(by_id[f"p{NEEDLES[case_id]}-c1"]["requests"][0])
        expected = {key: digest([case_id, key])[:12] for key in ("north", "east", "west")}
        payload = lane["payload"]
        del payload["output_config"]
        payload["max_tokens"] = 256
        block = payload["system"][0]
        filler = block["text"].split(" x")
        for offset, (key, code) in zip((8, 47, 93), expected.items()):
            filler[len(filler) * offset // 100] = (
                f"\nInventory record: active {key} verification code = {code}.\n")
        block["text"] = f"Retrieval benchmark {case_id}.\n" + " x".join(filler[1:])
        payload["messages"][0]["content"] = (
            "Read the inventory records across the system text. Return only a JSON object "
            "mapping north, east and west to their active verification codes, with no Markdown.")
        lane.update(expected_output=expected, json_output=True, expected_prompt_tokens=None,
                    minimum_prompt_tokens=NEEDLES[case_id] - 128)
    return {"id": case_id, "requests": [lane]}


def score(text, lane):
    expected = lane["expected_output"]
    if not lane["json_output"]:
        equal = text.strip().strip(".") == expected
        return {"correct": int(equal), "total": 1, "exact": equal}
    try:
        actual = json.loads(text)
    except ValueError:
        actual = None
    # JSON types matter (e.g. `true` is not a valid integer id).
    equal = digest(actual) == digest(expected)
    if isinstance(expected, dict):
        correct = sum(actual.get(k) == v for k, v in expected.items()) if isinstance(actual, dict) else 0
    elif isinstance(expected, list):
        correct = (sum(digest(a) == digest(b) for a, b in zip(actual, expected))
                   if isinstance(actual, list) else 0)
    else:
        correct = int(equal)
    return {"correct": correct, "total": len(expected) if isinstance(expected, (list, dict)) else 1,
            "exact": equal}


def measurement(case, row, startup, *, include_text=False):
    require(case["id"] == row["id"] and len(case["requests"]) == len(row["requests"]),
            f"{case['id']}: wrong case or lane count")
    lanes = []
    for index, (expected, record) in enumerate(zip(case["requests"], row["requests"])):
        label = f"{row['id']}/{index}"
        require(record["outcome"] == "success" and record["output_text"].strip(),
                f"{label}: incomplete response ({record['outcome']})")
        require(record["request_payload"] == expected["payload"], f"{label}: payload differs")
        done, = record["native_events"]
        require(done["event"] == "request_done" and
                done["server_instance_id"] == startup["server_instance_id"] and
                done["schema_version"] == startup["schema_version"] and
                record["wire_request_id"] and
                done["request"]["http_request_id"] == record["wire_request_id"],
                f"{label}: native log does not match wire request")
        result = done["result"]
        prompt = record["usage"]["input_tokens"]
        output = record["usage"]["output_tokens"]
        exact_prompt = expected["expected_prompt_tokens"]
        require(prompt == result["prompt_tokens"] and
                (exact_prompt is None or prompt == exact_prompt) and
                prompt >= expected.get("minimum_prompt_tokens", 1) and
                result["prefix_cache_hit_tokens"] == 0 and
                result["computed_prefill_tokens"] == prompt, f"{label}: not the expected cold prompt")
        require(output > 0 and result["completion_tokens"] == output and
                len(result["generated_token_ids"]) == output, f"{label}: invalid completion evidence")
        spec = done["speculative"]
        require(spec["backend"] == startup["engine"]["speculative_backend"],
                f"{label}: request did not use the configured speculative backend")
        drafted = spec["drafted_tokens"]
        accepted = spec["accepted_tokens"]
        require(0 <= accepted <= drafted, f"{label}: invalid speculative counters")
        decode_ns = record["last_output_ns"] - record["first_output_ns"]
        lanes.append({
            "payload_sha256": digest(expected["payload"]),
            "generated_token_ids_sha256": digest(result["generated_token_ids"]),
            "output_text_sha256": digest(record["output_text"]),
            "quality": score(record["output_text"], expected),
            "input_tokens": prompt,
            "output_tokens": output,
            "finish_reason": result["finish_reason"],
            "ttft_ms": record["ttft_ns"] / 1e6,
            "max_stream_gap_ms": (record["max_output_gap_ns"] / 1e6
                                  if record["max_output_gap_ns"] is not None else None),
            "output_tokens_per_second_after_first": (output - 1) * 1e9 / decode_ns
            if decode_ns > 0 and output > 1 else None,
            "prefill_units": done["engine_timing"]["units"]["prefill"],
            "prefill_service_ms": done["timings_seconds"]["prefill"] * 1000,
            "decode_service_ms": done["timings_seconds"]["decode"] * 1000,
            "speculative": spec,
            "draft_acceptance": accepted / drafted if drafted else None,
            **({"output_text": record["output_text"]} if include_text else {}),
        })
    # The archive's runner-level makespan included record serialization. Derive
    # the same wire interval for every version instead.
    end = max(record["completed_ns"] for record in row["requests"])
    start = min(record["sent_ns"] for record in row["requests"])
    return {"id": row["id"], "makespan_seconds": (end - start) / 1e9, "lanes": lanes}


def good_quality(lane):
    quality = lane["quality"]
    return quality["exact"] and lane["finish_reason"] == "stop_token"


def summarize(samples):
    groups = {}
    for sample in samples:
        for case in sample["cases"]:
            groups.setdefault(case["id"], []).append(case)
    result = {}
    for name, cases in groups.items():
        ranged = [case["makespan_seconds"] for case in cases]
        lanes = []
        for index, group in enumerate(zip(*(case["lanes"] for case in cases), strict=True)):
            first = group[0]
            for lane in group:
                require((lane["payload_sha256"], lane["input_tokens"]) ==
                        (first["payload_sha256"], first["input_tokens"]),
                        f"{name}/{index}: inputs differ between samples")
            medians = {}
            for key in ("ttft_ms", "max_stream_gap_ms", "output_tokens_per_second_after_first",
                        "prefill_units", "prefill_service_ms", "decode_service_ms", "draft_acceptance"):
                values = [lane[key] for lane in group]
                medians[key] = median(values) if None not in values else None
            lanes.append({
                "payload_sha256": first["payload_sha256"],
                "input_tokens": first["input_tokens"],
                "output_tokens": sorted({lane["output_tokens"] for lane in group}),
                "token_sequence_hashes": sorted({lane["generated_token_ids_sha256"] for lane in group}),
                "exact_quality_samples": sum(map(good_quality, group)),
                "medians": medians,
            })
        result[name] = {"samples": len(cases), "lanes": lanes,
                        "makespan_seconds": {"min": min(ranged), "median": median(ranged),
                                             "max": max(ranged)}}
        if name == LEDGER:
            result[name]["long_low_acceptance_samples"] = sum(
                good_quality(lane) and lane["output_tokens"] >= 512 and
                lane["draft_acceptance"] is not None and lane["draft_acceptance"] <= 0.70
                for case in cases for lane in case["lanes"]
            )
    return result


def collect_logs(stream, startup, records, deadline):
    wanted = {row["wire_request_id"] for row in records}
    require(None not in wanted and "" not in wanted and len(wanted) == len(records),
            "missing or repeated wire request ID")
    matched = {}
    while time.monotonic() < deadline:
        offset = stream.tell()
        line = stream.readline()
        if not line.endswith(b"\n"):
            stream.seek(offset)
            if wanted == set(matched):
                break
            time.sleep(0.1)
            continue
        event = json.loads(line)
        if event.get("event") == "server_start":
            raise ValueError("server restarted during measurement")
        if event.get("event") != "request_done" or event["server_instance_id"] != startup["server_instance_id"]:
            continue
        identifier = event["request"]["http_request_id"]
        require(identifier in wanted, "unrelated request reached the validation server")
        require(identifier not in matched, f"ambiguous request_done: {identifier}")
        matched[identifier] = event
    require(wanted == set(matched), "request_done missing from the selected server log")
    for record in records:
        record["native_events"] = [matched[record["wire_request_id"]]]


def run(args, fixtures):
    target = urlsplit(args.base_url)
    require(target.scheme == "http" and target.hostname and target.port and
            (target.hostname == "localhost" or ipaddress.ip_address(target.hostname).is_loopback) and
            not (target.username or target.password or target.path.strip("/") or target.query or target.fragment),
            "use the loopback URL of a dedicated validation server")
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision) and
            re.fullmatch(r"[0-9a-f]{64}", args.model_sha256), "provide the full release and model SHA")
    source = Path(os.environ.get("NINFER_SOURCE_ROOT", ROOT)).resolve()
    sys.path.insert(0, str(source))
    from tools.bench.ttft.execution import CaseContext
    from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest

    if args.case == "matched":
        selected = fixtures
    elif args.case == "supplemental":
        selected = [fixture(name, fixtures) for name in SUPPLEMENTAL]
    else:
        selected = [fixture(args.case, fixtures)]
    cases = []
    with args.request_log.open("rb") as stream:
        startup = None
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line.endswith(b"\n"):
                stream.seek(offset)
                break
            if line.strip():
                event = json.loads(line)
                if event.get("event") == "server_start":
                    require(startup is None, "use a fresh request log for this server instance")
                    startup = event
                require(event.get("event") not in {"request_start", "request_done"},
                        "start with a dedicated server that has not served any generation requests")
        require(startup is not None and startup["schema_version"] in (32, 33),
                "request log must contain a Serve startup with schema 32 or 33")
        config = configuration(startup, args.model_sha256)
        require(config["public_model_id"] == selected[0]["requests"][0]["payload"]["model"] and
                config["engine"]["max_concurrency"] >= max(len(case["requests"]) for case in selected),
                "Serve model or concurrency does not fit the fixture")
        client = NInferServeClient(args.base_url, args.timeout_seconds, os.environ.get("NINFER_API_KEY"))
        require(client.discover_model() == config["public_model_id"], "endpoint and fixture model differ")
        for case in selected:
            print(f"Running {case['id']}", file=sys.stderr)
            context = CaseContext(client, config["public_model_id"], args.timeout_seconds)
            try:
                handles = context.barrier(
                    (f"lane-{i}", ProtocolRequest(lane["protocol"], lane["path"], lane["payload"]))
                    for i, lane in enumerate(case["requests"]))
                for handle in handles:
                    context.require_success(handle)
                records = context.records()
                require(not context.failures, f"{case['id']}: {context.failures}")
                collect_logs(stream, startup, records, time.monotonic() + 30)
                cases.append(measurement(case, {"id": case["id"], "requests": records}, startup,
                                         include_text=case["id"] in (LEDGER, *NEEDLES)))
            finally:
                context.cancel_live()
    samples = [{"server_instance_id": startup["server_instance_id"], "schema_version": startup["schema_version"],
                "setup": setup_evidence(startup), "cases": cases}]
    return {"revision": args.revision, "configuration": config, "samples": samples,
            "summary": summarize(samples)}


def combine(paths):
    first, *others = [read(path) for path in paths]
    samples = list(first["samples"])
    for item in others:
        require((item["revision"], item["configuration"]) ==
                (first["revision"], first["configuration"]), "sample configurations differ")
        samples.extend(item["samples"])
    require(len({sample["server_instance_id"] for sample in samples}) == len(samples),
            "samples must come from independent server instances")
    return first, samples, summarize(samples)


def compare(args):
    before, old_samples, a = combine(args.before)
    after, new_samples, b = combine(args.after)
    require({sample["server_instance_id"] for sample in old_samples}.isdisjoint(
        sample["server_instance_id"] for sample in new_samples), "comparison reuses a server instance")
    old, new = before["configuration"], after["configuration"]
    switches = {"linear_state_dtype", "speculative_length_policy"}
    require({k: v for k, v in old.items() if k not in switches} ==
            {k: v for k, v in new.items() if k not in switches},
            "check model, hardware and non-target engine settings before comparing")
    changed = {key: [old[key], new[key]] for key in switches if old[key] != new[key]}
    require(len(changed) <= 1 and (not changed or before["revision"] == after["revision"]),
            "isolate each switch on the same source revision")
    require(a.keys() == b.keys(), "cases differ; compare the same selection on both sides")
    cases, failures = {}, []
    for name, case in a.items():
        current = b[name]
        lane_comparisons = []
        for i, (left, right) in enumerate(zip(case["lanes"], current["lanes"], strict=True)):
            require((left["payload_sha256"], left["input_tokens"]) ==
                    (right["payload_sha256"], right["input_tokens"]), f"{name}/{i}: inputs differ")
            equal = left["token_sequence_hashes"] == right["token_sequence_hashes"] and (
                len(left["token_sequence_hashes"]) == 1)
            quality_passed = (left["exact_quality_samples"] == case["samples"] and
                              right["exact_quality_samples"] == current["samples"])
            if not quality_passed or (not equal and "linear_state_dtype" not in changed):
                failures.append(f"{name}/{i}: quality oracle or greedy token sequence differs")
            same_work = quality_passed and left["output_tokens"] == right["output_tokens"] and (
                len(left["output_tokens"]) == 1)
            base_tps = left["medians"]["output_tokens_per_second_after_first"]
            next_tps = right["medians"]["output_tokens_per_second_after_first"]
            lane_comparisons.append({
                "greedy_tokens_equal": equal, "same_output_length": same_work,
                "before": left, "after": right,
                "ttft_change_pct": round(100 * (
                    right["medians"]["ttft_ms"] / left["medians"]["ttft_ms"] - 1), 3),
                "stream_tps_change_pct": round(100 * (next_tps / base_tps - 1), 3)
                if same_work and base_tps and next_tps is not None else None,
            })
        cases[name] = {"before_samples": case["samples"], "after_samples": current["samples"],
                       "before_makespan_seconds": case["makespan_seconds"],
                       "after_makespan_seconds": current["makespan_seconds"],
                       "lanes": lane_comparisons}
        if name == LEDGER:
            cases[name]["before_long_low_acceptance_samples"] = case["long_low_acceptance_samples"]
            cases[name]["after_long_low_acceptance_samples"] = current["long_low_acceptance_samples"]
    return {"before_revision": before["revision"], "after_revision": after["revision"],
            "changed_switches": changed, "quality_failures": failures, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the frozen and new workloads")
    live = commands.add_parser("run", help="replay fixtures against an isolated loopback server")
    live.add_argument("--case", required=True, choices=("matched", "supplemental", *SUPPLEMENTAL))
    live.add_argument("--base-url", required=True)
    live.add_argument("--request-log", required=True, type=Path)
    live.add_argument("--revision", required=True, help="full source revision from the tested image")
    live.add_argument("--model-sha256", required=True, help="SHA-256 of the served model artifact")
    live.add_argument("--timeout-seconds", default=900, type=int)
    delta = commands.add_parser("compare", help="compare one or more independent samples on each side")
    delta.add_argument("--before", type=Path, nargs="+", required=True)
    delta.add_argument("--after", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            report = compare(args)
        elif args.command == "list":
            report = {"matched": [case["id"] for case in matched_fixtures()], "supplemental": list(SUPPLEMENTAL)}
        else:
            require(args.timeout_seconds > 0, "timeout must be positive")
            report = run(args, matched_fixtures())
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if report.get("quality_failures") or any(
            lane["exact_quality_samples"] != case["samples"]
            for case in report.get("summary", {}).values() for lane in case["lanes"]
        ):
            return 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
