import argparse
import base64
import json
import mimetypes
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def post(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def caption(path: Path, args, prompt: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    rel = path.relative_to(args.folder)
    author = rel.parts[0] if len(rel.parts) > 1 else ""  # first subfolder
    payload = dict(
        model=args.model,
        messages=[
            dict(
                role="user",
                content=[
                    dict(type="image_url", image_url=dict(url=f"data:{mime};base64,{b64}")),
                    dict(type="text", text=prompt.replace("{author}", author)),
                ],
            ),
        ],
        reasoning_effort="medium",
    )
    text = post(f"{args.base_url}/chat/completions", payload)["choices"][0]["message"]["content"]
    return " ".join(text.split())  # TSV-safe: collapse newlines/tabs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default="captions.tsv")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.model is None:
        args.model = post(f"{args.base_url}/models")["data"][0]["id"]
    prompt = args.prompt.read_text().strip()
    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Captioning {len(files)} images with {args.model}")

    with open(args.out, "x") as f, ThreadPoolExecutor(args.workers) as pool:
        f.write("filename\tprompt\n")
        futures = {pool.submit(caption, p, args, prompt): p for p in files}
        for i, fut in enumerate(as_completed(futures)):
            path, text = futures[fut], fut.result()
            f.write(f"{path.relative_to(args.folder)}\t{text}\n")
            f.flush()
            print(f"[{i + 1}/{len(files)}] {path.relative_to(args.folder)}")
