#!/usr/bin/env python3
"""Verify provider prompt-caching visibility through an OpenAI-compatible Responses API.

This complements ``devtools/check_prompt_cache.py`` by testing ``/v1/responses``.

Usage examples:

    export API_BASE="https://cody.ib-inet.com"
    export API_KEY="..."
    export MODEL="gpt-5-nano"
    uv run python devtools/check_responses_prompt_cache.py

    uv run python devtools/check_responses_prompt_cache.py --variant both --requests 5
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, cast
from urllib.response import addinfourl

Variant = Literal["shared-prefix", "identical-full", "both"]
JSONDict = dict[str, object]


@dataclass(frozen=True)
class CliArgs:
    api_base: str
    api_key: str
    model: str
    endpoint: str
    repeat_sentences: int
    requests: int
    variant: Variant
    max_output_tokens: int
    show_full_response: bool


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None


@dataclass(frozen=True)
class RequestResult:
    label: str
    response_data: JSONDict
    usage: UsageStats


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Send multiple nearly identical requests to an OpenAI-compatible Responses "
            "API endpoint and print usage fields that may indicate provider prompt caching."
        )
    )
    _ = parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE", ""),
        help="Gateway base URL, e.g. https://cody.ib-inet.com",
    )
    _ = parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", ""),
        help="Gateway API key",
    )
    _ = parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", ""),
        help="Model ID to call through the gateway",
    )
    _ = parser.add_argument(
        "--endpoint",
        default=os.environ.get("RESPONSES_ENDPOINT", "/v1/responses"),
        help="Responses endpoint path (default: /v1/responses)",
    )
    _ = parser.add_argument(
        "--repeat-sentences",
        type=int,
        default=int(os.environ.get("REPEAT_SENTENCES", "300")),
        help="How many repeated sentences to use in the shared prefix",
    )
    _ = parser.add_argument(
        "--requests",
        type=int,
        default=int(os.environ.get("REQUEST_COUNT", "5")),
        help="How many requests to send in each variant run",
    )
    _ = parser.add_argument(
        "--variant",
        choices=("shared-prefix", "identical-full", "both"),
        default=os.environ.get("PROMPT_VARIANT", "shared-prefix"),
        help=(
            "Prompt variant to test: shared-prefix varies only the tail, identical-full "
            "repeats the exact same request, both runs both tests"
        ),
    )
    _ = parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("MAX_OUTPUT_TOKENS", "32")),
        help="max_output_tokens sent with each request",
    )
    _ = parser.add_argument(
        "--show-full-response",
        action="store_true",
        help="Print the full JSON response for each request",
    )
    namespace = parser.parse_args()
    return CliArgs(
        api_base=cast(str, namespace.api_base),
        api_key=cast(str, namespace.api_key),
        model=cast(str, namespace.model),
        endpoint=cast(str, namespace.endpoint),
        repeat_sentences=cast(int, namespace.repeat_sentences),
        requests=cast(int, namespace.requests),
        variant=cast(Variant, namespace.variant),
        max_output_tokens=cast(int, namespace.max_output_tokens),
        show_full_response=cast(bool, namespace.show_full_response),
    )


def require_value(name: str, value: str) -> str:
    if value.strip():
        return value.strip()
    raise SystemExit(f"Missing required value for {name}. Set it via flag or environment variable.")


def post_json(url: str, api_key: str, payload: JSONDict) -> JSONDict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with cast(addinfourl, urllib.request.urlopen(request, timeout=180)) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"HTTP {exc.code} from gateway\nURL: {url}\nResponse:\n{error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {url}: {exc}") from exc

    try:
        parsed_obj = cast(object, json.loads(body))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gateway did not return valid JSON:\n{body}") from exc
    if not isinstance(parsed_obj, dict):
        raise SystemExit(f"Expected JSON object from gateway, got: {type(parsed_obj).__name__}")
    return cast(JSONDict, parsed_obj)


def find_first_key(value: object, key: str) -> int | None:
    if isinstance(value, dict):
        mapping = cast(JSONDict, value)
        candidate = mapping.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        for nested in mapping.values():
            found = find_first_key(nested, key)
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        values = cast(list[object], value)
        for nested in values:
            found = find_first_key(nested, key)
            if found is not None:
                return found
    return None


def build_shared_prefix(repeat_sentences: int) -> str:
    sentence = "You are verifying provider prompt caching. Keep this exact prefix unchanged across requests. "
    return sentence * repeat_sentences


def build_input_text(index: int, variant: Variant) -> str:
    if variant == "identical-full":
        return "Reply with exactly: cache-check"
    return f"Reply with exactly: request-{index}"


def build_payload(
    model: str,
    shared_prefix: str,
    input_text: str,
    max_output_tokens: int,
) -> JSONDict:
    return {
        "model": model,
        "instructions": shared_prefix,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
    }


def extract_usage_stats(response_data: JSONDict) -> UsageStats:
    usage_obj = response_data.get("usage")
    usage = cast(JSONDict | None, usage_obj if isinstance(usage_obj, dict) else None)
    if usage is None:
        return UsageStats(None, None, None, None, None)
    input_tokens = find_first_key(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = find_first_key(usage, "prompt_tokens")
    output_tokens = find_first_key(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = find_first_key(usage, "completion_tokens")
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=find_first_key(usage, "cached_tokens"),
        cache_read_input_tokens=find_first_key(usage, "cache_read_input_tokens"),
        cache_creation_input_tokens=find_first_key(usage, "cache_creation_input_tokens"),
    )


def print_usage_summary(result: RequestResult, *, show_full_response: bool) -> None:
    usage = result.response_data.get("usage")
    print(f"\n=== {result.label} ===")
    if usage is None:
        print("usage: <missing>")
    else:
        print("usage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))

    print("summary:")
    print(f"  input_tokens: {result.usage.input_tokens}")
    print(f"  output_tokens: {result.usage.output_tokens}")
    print(f"  cached_tokens: {result.usage.cached_tokens}")
    print(f"  cache_read_input_tokens: {result.usage.cache_read_input_tokens}")
    print(f"  cache_creation_input_tokens: {result.usage.cache_creation_input_tokens}")

    if show_full_response:
        print(f"\n--- full response: {result.label} ---")
        print(json.dumps(result.response_data, indent=2, ensure_ascii=False))


def print_variant_interpretation(variant: Variant, results: list[RequestResult]) -> None:
    print("\nrollup:")
    for result in results:
        line = (
            f"  {result.label}: input={result.usage.input_tokens}, "
            + f"output={result.usage.output_tokens}, "
            + f"cached={result.usage.cached_tokens}, "
            + f"cache_read={result.usage.cache_read_input_tokens}, "
            + f"cache_creation={result.usage.cache_creation_input_tokens}"
        )
        print(line)

    cache_values = [
        value
        for result in results
        for value in (
            result.usage.cached_tokens,
            result.usage.cache_read_input_tokens,
            result.usage.cache_creation_input_tokens,
        )
    ]

    print("\nInterpretation:")
    if any(value is not None and value > 0 for value in cache_values):
        print(f"  {variant}: prompt caching appears to be working and visible.")
    elif any(value == 0 for value in cache_values):
        print(
            f"  {variant}: cache fields are present but zero across this run. "
            + "The provider/gateway exposes cache metrics, but this variant did not show a cache hit."
        )
    else:
        print(
            f"  {variant}: no cache-specific usage fields were found. "
            + "The provider may not expose prompt-cache stats, or the gateway may not be forwarding them."
        )


def run_variant(
    *,
    url: str,
    api_key: str,
    model: str,
    shared_prefix: str,
    max_output_tokens: int,
    request_count: int,
    variant: Variant,
    show_full_response: bool,
) -> None:
    print("\n" + "=" * 78)
    if variant == "identical-full":
        print("Variant: identical-full")
        print("  Sending the exact same full Responses API request for every call.")
    else:
        print("Variant: shared-prefix")
        print("  Keeping the large instructions prefix constant while varying only the input tail.")
    print(f"  Request count: {request_count}")

    results: list[RequestResult] = []
    for index in range(1, request_count + 1):
        input_text = build_input_text(index, variant)
        response_data = post_json(
            url,
            api_key,
            build_payload(model, shared_prefix, input_text, max_output_tokens),
        )
        result = RequestResult(
            label=f"request {index}",
            response_data=response_data,
            usage=extract_usage_stats(response_data),
        )
        results.append(result)
        print_usage_summary(result, show_full_response=show_full_response)

    print_variant_interpretation(variant, results)


def main() -> int:
    args = parse_args()
    api_base = require_value("API_BASE / --api-base", args.api_base).rstrip("/")
    api_key = require_value("API_KEY / --api-key", args.api_key)
    model = require_value("MODEL / --model", args.model)
    if args.requests < 2:
        raise SystemExit("--requests must be at least 2")
    if args.repeat_sentences < 1:
        raise SystemExit("--repeat-sentences must be at least 1")

    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{api_base}{endpoint}"
    shared_prefix = build_shared_prefix(args.repeat_sentences)

    print("Prompt-cache verification (Responses API)")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Prefix length (chars): {len(shared_prefix)}")
    print(f"  Repeat sentences: {args.repeat_sentences}")
    print(f"  Requests per variant: {args.requests}")
    print(f"  Variant selection: {args.variant}")

    variants: list[Variant]
    if args.variant == "both":
        variants = ["shared-prefix", "identical-full"]
    else:
        variants = [args.variant]

    for variant in variants:
        run_variant(
            url=url,
            api_key=api_key,
            model=model,
            shared_prefix=shared_prefix,
            max_output_tokens=args.max_output_tokens,
            request_count=args.requests,
            variant=variant,
            show_full_response=args.show_full_response,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
