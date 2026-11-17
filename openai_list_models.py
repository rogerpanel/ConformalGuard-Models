"""Print the model ids a key can use on the OpenAI API or on an OpenAI-compatible service (OpenAI SDK).

    export OPENAI_API_KEY=...
    python scripts/openai_list_models.py
    python scripts/openai_list_models.py --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --filter deepseek
"""

from __future__ import annotations

import argparse
import os

import openai
from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--filter", default="", help="only ids containing this text")
    args = ap.parse_args()
    client = OpenAI(max_retries=2, api_key=os.environ.get(args.api_key_env), **({"base_url": args.base_url} if args.base_url else {}))
    try:
        ids = sorted(m.id for m in client.models.list())
    except openai.AuthenticationError:
        print(f"the key in {args.api_key_env} was rejected (401)")
        return
    except openai.APIStatusError as e:
        print(f"could not list models: API error {e.status_code}: {e.message}")
        return
    except openai.APIConnectionError as e:
        print(f"could not list models: connection error: {e}")
        return
    if args.filter:
        ids = [i for i in ids if args.filter.lower() in i.lower()]
    print(f"{len(ids)} models" + (f" containing '{args.filter}'" if args.filter else "") + ":")
    print(", ".join(ids))


if __name__ == "__main__":
    main()
