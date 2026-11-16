"""Print the model ids an OpenAI key can use, to choose a value for OPENAI_MODEL (OpenAI SDK).

    export OPENAI_API_KEY=...
    python scripts/openai_list_models.py
"""

from __future__ import annotations

import openai
from openai import OpenAI


def main() -> None:
    try:
        ids = sorted(m.id for m in OpenAI(max_retries=2).models.list())
    except openai.AuthenticationError:
        print("the OpenAI key was rejected (401): check the OPENAI_API_KEY secret")
        return
    except openai.APIStatusError as e:
        print(f"could not list models: API error {e.status_code}: {e.message}")
        return
    except openai.APIConnectionError as e:
        print(f"could not list models: connection error: {e}")
        return
    print(f"{len(ids)} models available to this key:")
    print(", ".join(ids))


if __name__ == "__main__":
    main()
