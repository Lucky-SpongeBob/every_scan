#!/usr/bin/env python3
"""Shared VLM request parsing for the diagnostic shadow nodes."""

import base64
import json
import os
import re
from urllib.request import Request, urlopen


def load_dotenv(path):
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)


def query_json(args, prompt, image_jpeg):
    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("VLM_API_KEY is missing")
    if isinstance(image_jpeg, dict) or isinstance(image_jpeg, bytes):
        images = [image_jpeg]
    else:
        images = list(image_jpeg)
    content = [{"type": "text", "text": prompt}]
    for item in images:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            data = item.get("jpeg", item.get("data", b""))
        else:
            label = ""
            data = item
        if label:
            content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(data).decode("ascii")
                },
            }
        )
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 384,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if getattr(args, "enable_thinking", False):
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        reasoning_effort = getattr(args, "reasoning_effort", "low")
        if reasoning_effort in ("low", "medium", "high"):
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = Request(
        args.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=args.api_timeout) as response:
        body = json.load(response)
    choice = body["choices"][0]
    message = choice["message"].get("content", "")
    if not message:
        message = choice["message"].get("reasoning_content", "")
    match = re.search(r"\{.*\}", message, re.DOTALL)
    if match is None:
        raise RuntimeError(
            "VLM did not return JSON: finish=%s content_chars=%d "
            "reasoning_chars=%d"
            % (
                choice.get("finish_reason"),
                len(choice["message"].get("content", "")),
                len(choice["message"].get("reasoning_content", "")),
            )
        )
    return json.loads(match.group(0))
