"""Send one tiny chat request to the OpenAI API or an OpenAI-compatible service, to check that a key
and a model id work before a long run (OpenAI SDK).  Exit code 0 if the model answered, 1 otherwise.

    python scripts/openai_ping.py --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model deepseek/deepseek-v3.2

Listing models is not a test of the key: some services answer that request without authentication.
"""

from __future__ import annotations

import argparse
import os
import sys

import openai
from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = ap.parse_args()
    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"FAILED: the variable {args.api_key_env} is empty"); sys.exit(1)
    client = OpenAI(max_retries=1, api_key=key, **({"base_url": args.base_url} if args.base_url else {}))
    try:
        resp = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "Reply with the number 7."}])
    except openai.AuthenticationError as e:
        print(f"FAILED: the key in {args.api_key_env} was rejected (401): {e.message}\n"
              f"        The key has {len(key)} characters and starts with '{key[:6]}'. Check that it is a key of this service, "
              f"copied whole and without spaces, and that the account has credit.")
        sys.exit(1)
    except openai.NotFoundError as e:
        print(f"FAILED: the model id '{args.model}' was not found (404): {e.message}"); sys.exit(1)
    except openai.APIStatusError as e:
        print(f"FAILED: API error {e.status_code}: {e.message}"); sys.exit(1)
    except openai.APIConnectionError as e:
        print(f"FAILED: connection error: {e}"); sys.exit(1)
    text = (resp.choices[0].message.content or "").strip()
    print(f"OK: {getattr(resp, 'model', args.model)} answered {text[:40]!r}")


if __name__ == "__main__":
    main()
