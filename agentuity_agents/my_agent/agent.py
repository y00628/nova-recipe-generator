# agentuity_pantrychef_agent.py
from __future__ import annotations

import os
import re
import json
import base64
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx  # pip install httpx
from agentuity import AgentRequest, AgentResponse, AgentContext
from openai import AsyncOpenAI

# Optional: image normalization if Pillow available (handles HEIC/WEBP/etc.)
try:
    from PIL import Image  # pip install pillow ; optionally pillow-heif to decode HEIC
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

client = AsyncOpenAI()

# =========================
# Helpers: normalization + expirations
# =========================

CANON = [
    "milk","eggs","butter","yogurt","chicken breast","ground beef","spinach","lettuce",
    "tomato","onion","garlic","carrot","potato","rice","pasta","bread","tortilla",
    "cheddar","mozzarella","salmon","tofu","mushroom","bell pepper","broccoli","zucchini",
    "cilantro","basil","lime","lemon","avocado","cucumber","olive oil","canola oil",
    "flour","sugar","brown sugar","yeast","baking powder","baking soda","salt","pepper",
    "cumin","paprika","chili powder","soy sauce","vinegar","oats","almond milk",
]

EXPIRY_RULES_DAYS: Dict[str, int] = {
    "milk": 7, "eggs": 21, "yogurt": 14, "butter": 60, "bread": 5, "tortilla": 14,
    "chicken breast": 2, "ground beef": 2, "salmon": 2, "tofu": 7,
    "spinach": 5, "lettuce": 7, "tomato": 5, "onion": 21, "garlic": 30, "carrot": 21, "potato": 30,
    "broccoli": 7, "mushroom": 5, "zucchini": 7, "bell pepper": 7, "cucumber": 7, "avocado": 5,
    "rice": 365, "pasta": 365, "flour": 180, "sugar": 365, "brown sugar": 180,
    "olive oil": 180, "canola oil": 180, "cheddar": 30, "mozzarella": 21,
}

def _ngrams(s: str, n: int) -> List[str]:
    s = f" {s} "
    return [s[i:i+n] for i in range(0, len(s)-n+1)]

def _sim(a: str, b: str) -> float:
    def grams(t: str):
        gs = set()
        for w in t.split():
            gs.update(_ngrams(w, 2))
        return gs
    A, B = grams(a), grams(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def normalize_name(raw: str) -> str:
    raw = re.sub(r"[^a-z\s]", " ", raw.lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""
    best, score = raw, 0.0
    for c in CANON:
        sc = _sim(raw, c)
        if sc > score:
            best, score = c, sc
    return best if score >= 0.8 else raw

def estimate_expiry(name: str, start: Optional[datetime] = None) -> Optional[str]:
    days = EXPIRY_RULES_DAYS.get(name)
    if not days:
        return None
    base = start or datetime.utcnow()
    return (base + timedelta(days=days)).isoformat()

# =========================
# Parsing helpers
# =========================

LIST_SPLIT = re.compile(r"[,;\n]|\band\b", re.I)
DATA_URL_RE = re.compile(r"^data:(?P<ctype>[^;]+);\s*base64,\s*(?P<b64>.+)$", re.I | re.S)

def parse_free_text(text: str) -> List[str]:
    parts = [p.strip() for p in LIST_SPLIT.split(text or "")]
    items = [re.sub(r"\s+\d+.*$", "", p).strip() for p in parts]
    items = [i for i in items if len(i) > 1]
    return items[:40]

async def _maybe_await(x):
    return (await x) if hasattr(x, "__await__") else x

def _maybe_bytes(x) -> Optional[bytes]:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    return None

def _try_decode_b64(s: str) -> Optional[bytes]:
    s = (s or "").strip()
    if not s:
        return None
    m = DATA_URL_RE.match(s)
    if m:
        try:
            return base64.b64decode(m.group("b64"))
        except Exception:
            return None
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return None

def _sniff_image_type(b: bytes) -> Optional[str]:
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
        return "image/gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    if b[4:8] == b"ftyp" and any(tag in b[8:16] for tag in (b"heic", b"heix", b"hevc", b"mif1", b"heif")):
        return "image/heic"
    return None

def _normalize_image_for_vision(b: bytes, ctx: AgentContext) -> Optional[bytes]:
    """
    Best-effort: convert any format to RGB JPEG (max side 1600px, quality 85).
    Returns JPEG bytes or None if Pillow not available/fails.
    """
    if not PIL_AVAILABLE:
        return None
    try:
        from io import BytesIO
        bio = BytesIO(b)
        img = Image.open(bio)
        # Some HEIC decoders are provided by pillow-heif; if not installed, this may fail earlier.
        img = img.convert("RGB")
        # Downscale if huge
        max_side = 1600
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)))
        out = BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()
    except Exception as e:
        ctx.logger.warn(f"Image normalize failed (non-fatal): {e}")
        return None

async def _get_first_file_bytes_from_request(request: AgentRequest, context: AgentContext) -> Optional[bytes]:
    containers = []
    for attr in ("files", "file", "attachments", "uploads"):
        if hasattr(request, attr):
            obj = getattr(request, attr)
            if callable(obj):
                obj = await _maybe_await(obj())
            if obj:
                containers.append(obj)

    candidates: List[Any] = []
    for obj in containers:
        if isinstance(obj, dict):
            candidates.extend(list(obj.values()))
        elif isinstance(obj, (list, tuple, set)):
            candidates.extend(list(obj))
        else:
            candidates.append(obj)

    async def _extract_from_obj(o) -> Optional[bytes]:
        b = _maybe_bytes(o)
        if b is not None:
            return b
        for name in ("bytes", "read", "content", "body", "value", "data", "buffer", "getvalue", "tobytes", "to_bytes", "toBytes"):
            if hasattr(o, name):
                attr = getattr(o, name)
                v = await _maybe_await(attr() if callable(attr) else attr)
                b = _maybe_bytes(v)
                if b is not None:
                    return b
        for name in ("file", "fp", "stream"):
            if hasattr(o, name):
                inner = getattr(o, name)
                if hasattr(inner, "read"):
                    v = await _maybe_await(inner.read())
                    b = _maybe_bytes(v)
                    if b is not None:
                        return b
        return None

    for idx, o in enumerate(candidates):
        try:
            v = await _extract_from_obj(o)
            if v is not None:
                return v
        except Exception as e:
            context.logger.error(f"_extract_from_obj[{idx}] failed: {e}")
            continue
    return None

async def _get_bytes_from_anywhere(request: AgentRequest, context: AgentContext) -> Optional[bytes]:
    data = getattr(request, "data", None)

    if data is not None:
        b = _maybe_bytes(data)
        if b is not None:
            return b
        for name in ("bytes", "read", "array_buffer", "content", "body", "value", "data", "buffer", "getvalue", "tobytes", "to_bytes", "toBytes"):
            if hasattr(data, name):
                attr = getattr(data, name)
                v = await _maybe_await(attr() if callable(attr) else attr)
                b = _maybe_bytes(v)
                if b is not None:
                    return b

    if hasattr(request, "json"):
        try:
            body = await _maybe_await(request.json())
            if isinstance(body, dict):
                for k in ("bytes", "binary", "content", "body", "image", "file", "payload", "data"):
                    if k in body:
                        b = _maybe_bytes(body[k])
                        if b is not None:
                            return b
                for k in ("dataUrl", "dataURL", "data_url", "base64", "b64"):
                    v = body.get(k)
                    if isinstance(v, str):
                        decoded = _try_decode_b64(v)
                        if decoded is not None:
                            return decoded
        except Exception:
            pass

    from_req = await _get_first_file_bytes_from_request(request, context)
    if from_req is not None:
        return from_req

    if data is not None and hasattr(data, "text"):
        try:
            t = await _maybe_await(data.text())
            if isinstance(t, str) and t.strip():
                decoded = _try_decode_b64(t)
                if decoded is not None:
                    return decoded
        except Exception:
            pass

    if isinstance(data, dict):
        for k in ("bytes", "binary", "content", "body", "image", "file", "payload", "data"):
            if k in data:
                b = _maybe_bytes(data[k])
                if b is not None:
                    return b
        for k in ("dataUrl", "dataURL", "data_url", "base64", "b64"):
            v = data.get(k)
            if isinstance(v, str):
                decoded = _try_decode_b64(v)
                if decoded is not None:
                    return decoded

    return None

# =========================
# LLM calls (Vision extraction + Recipe)
# =========================

async def extract_from_image(content_type: str, image_bytes: bytes, ctx: AgentContext, debug: bool = False, diag: Optional[dict] = None) -> List[str]:
    """
    Prefer Responses API (JSON schema). Fallback to Chat Completions (JSON mode).
    Optionally normalizes the image (RGB JPEG) if Pillow is available.
    """
    # Normalize image for maximum compatibility (handles HEIC/WEBP, big images, weird metadata)
    normalized = _normalize_image_for_vision(image_bytes, ctx)
    if normalized:
        image_bytes = normalized
        content_type = "image/jpeg"
        if debug and isinstance(diag, dict):
            diag.setdefault("notes", []).append("image-normalized->jpeg")

    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{content_type};base64,{b64}"

    def _to_items(raw: str) -> List[str]:
        try:
            obj = json.loads(raw)
            items = obj.get("items")
            if isinstance(items, list):
                return [str(s).strip() for s in items if str(s).strip()][:40]
        except Exception:
            pass
        return parse_free_text(raw)

    # 1) Responses API
    try:
        resp = await client.responses.create(
            model="gpt-4o-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract grocery ingredients/items visible in the image. Output ONLY JSON as {\"items\":[...]} with short generic names (no quantities)."},
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ingredients_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array", "items": {"type": "string"}, "maxItems": 40}
                        },
                        "required": ["items"],
                        "additionalProperties": False
                    }
                }
            },
        )
        raw = resp.output_text or ""
        items = _to_items(raw)
        if items:
            if debug and isinstance(diag, dict):
                diag.setdefault("notes", []).append(f"vision-ok(responses):{len(items)}")
            return items
    except Exception as e:
        ctx.logger.error(f"Responses extractor failed: {e}\n{traceback.format_exc()}")

    # 2) Chat Completions fallback
    try:
        comp = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract grocery items you can clearly see. Output ONLY JSON: {\"items\":[\"...\"]}. Short generic names; no quantities."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Return JSON only."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            temperature=0
        )
        raw = comp.choices[0].message.content or ""
        items = _to_items(raw)
        if items:
            if debug and isinstance(diag, dict):
                diag.setdefault("notes", []).append(f"vision-ok(chat):{len(items)}")
            return items
    except Exception as e:
        ctx.logger.error(f"Chat extractor failed: {e}\n{traceback.format_exc()}")

    return []

async def llm_recipe_text(items_text: str) -> str:
    comp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": ("You are a concise chef assistant. Suggest ONE recipe that uses ONLY the listed ingredients. "
                         "If a crucial item is missing, state it briefly instead of inventing substitutes. "
                         "No pasta unless 'pasta' is explicitly in the list.")},
            {"role": "user",
             "content": (f"My ingredients: {items_text}\n"
                         "Return 3 short parts:\n"
                         "1) Title\n2) Ingredients used\n3) Steps (3-6 bullets)\n")},
        ],
        temperature=0.2,
    )
    return comp.choices[0].message.content or "No recipe generated."

# Optional: YouTube tutorial lookup
async def youtube_search(query: str, max_results: int = 3) -> List[str]:
    key = os.getenv("YOUTUBE_KEY")
    if not key:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {"part": "snippet", "type": "video", "maxResults": str(max_results), "q": query, "key": key}
    async with httpx.AsyncClient(timeout=20) as hx:
        r = await hx.get(url, params=params)
        if r.status_code != 200:
            return []
        data = r.json()
        vids: List[str] = []
        for it in data.get("items", []):
            vid = (it.get("id") or {}).get("videoId")
            if vid:
                vids.append(f"https://www.youtube.com/watch?v={vid}")
        return vids

# =========================
# Agentuity entrypoints
# =========================

def welcome():
    return {
        "welcome": "PantryChef Agent ready. Send an image (receipt/fridge) or a text list of ingredients.",
        "prompts": [
            {"data": "eggs, spinach, mozzarella, tomato", "contentType": "text/plain"},
            {"data": "data:image/jpeg;base64,<BASE64_HERE>", "contentType": "image/jpeg"},
        ],
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    try:
        ctype = (getattr(request.data, "content_type", "") or "").lower()
        items_raw: List[str] = []

        # Debug flag
        debug = False
        try:
            if hasattr(request, "json"):
                jb = await _maybe_await(request.json())
                if isinstance(jb, dict) and jb.get("debug") is True:
                    debug = True
        except Exception:
            pass

        diag = {"debug": debug, "had_bytes": False, "byte_count": 0, "inferred_ctype": None, "initial_ctype": ctype, "notes": []}

        # Try to pull bytes from anywhere (covers JSON dataUrl/base64 too)
        image_bytes_any = await _get_bytes_from_anywhere(request, context)

        if image_bytes_any is not None and len(image_bytes_any) > 0:
            diag["had_bytes"] = True
            diag["byte_count"] = len(image_bytes_any)

            sniffed = _sniff_image_type(image_bytes_any)
            inferred_ctype = sniffed or ("image/jpeg" if not ctype or "image" not in ctype else ctype)
            diag["inferred_ctype"] = inferred_ctype
            if sniffed and ctype and sniffed != ctype:
                diag["notes"].append(f"Overrode content_type '{ctype}' with sniffed '{sniffed}'")

            # Vision extraction
            items_raw = await extract_from_image(inferred_ctype, image_bytes_any, context, debug=debug, diag=diag)
        else:
            if image_bytes_any is None:
                diag["notes"].append("No bytes returned by _get_bytes_from_anywhere()")
            else:
                diag["notes"].append("Zero-length bytes returned")

        # Text/JSON path if nothing extracted
        if not items_raw:
            txt = ""
            data = getattr(request, "data", None)

            if data is not None and hasattr(data, "text"):
                try:
                    t = data.text()
                    txt = await _maybe_await(t)
                except Exception:
                    txt = ""

            if hasattr(request, "json"):
                try:
                    body = await _maybe_await(request.json())
                    if isinstance(body, dict):
                        txt = body.get("text") or body.get("ingredients") or txt
                        for k in ("dataUrl", "dataURL", "data_url", "base64", "b64", "image"):
                            v = body.get(k)
                            if isinstance(v, str) and v.strip():
                                txt = v
                                break
                except Exception:
                    pass

            if debug and isinstance(txt, str):
                diag["notes"].append(f"text-len={len(txt)} (first 32 chars: {txt[:32]!r})")

            maybe_img = _try_decode_b64(txt) if isinstance(txt, str) else None
            if maybe_img is not None:
                if debug:
                    diag["notes"].append(f"Decoded base64/dataURL to {len(maybe_img)} bytes")
                inferred_ctype = "image/jpeg"
                m = DATA_URL_RE.match(txt.strip()) if isinstance(txt, str) else None
                if m:
                    inferred_ctype = m.group("ctype") or inferred_ctype
                items_raw = await extract_from_image(inferred_ctype, maybe_img, context, debug=debug, diag=diag)
            else:
                items_raw = parse_free_text(txt or "")

        # Normalize/dedupe + expirations
        now = datetime.utcnow()
        seen: set[str] = set()
        items_struct: List[Dict[str, Any]] = []
        for raw in items_raw:
            norm = normalize_name(raw)
            if norm in seen:
                continue
            seen.add(norm)
            items_struct.append({"name": raw, "normalized": norm, "expires_at": estimate_expiry(norm, now)})

        # Recipe + videos
        items_text = ", ".join([i["normalized"] for i in items_struct]).strip()
        if items_text:
            recipe_text = await llm_recipe_text(items_text)
            vids = await youtube_search(f"how to make {items_text.split(',')[0].strip()}") if items_text else []
        else:
            recipe_text = "I couldn’t detect any ingredients. Please send a clearer photo or type a short list (e.g., 'eggs, spinach, mozzarella, tomato')."
            vids = []

        # Optional video generation
        video_info: Optional[Dict[str, Any]] = None
        body = None
        try:
            if hasattr(request, "json"):
                body = await _maybe_await(request.json())
        except Exception:
            body = None

        meta = None
        try:
            if hasattr(request, "metadata"):
                meta = await _maybe_await(request.metadata())
        except Exception:
            meta = None

        gen_flag = False
        if isinstance(body, dict) and body.get("generate_video"):
            gen_flag = True
        if isinstance(meta, dict) and meta.get("generate_video"):
            gen_flag = True

        selected_recipe_text = None
        if isinstance(body, dict) and body.get("action") == "select_recipe":
            selected_recipe_text = body.get("recipe") or body.get("recipe_text")
            gen_flag = True
        elif isinstance(meta, dict) and meta.get("action") == "select_recipe":
            selected_recipe_text = meta.get("recipe")
            gen_flag = True

        if gen_flag:
            prompt_text = selected_recipe_text or recipe_text
            try:
                video_info = await generate_and_store_video(context, request, prompt_text)
            except Exception as e:
                context.logger.error(f"Video generation helper failed: {e}")
                video_info = {"error": "video_generation_exception"}

        payload = {
            "items": items_struct,
            "recipe": recipe_text,
            "videos": vids,
            "generated_video": video_info,
        }
        if debug:
            payload["diagnostics"] = diag
        return response.json(payload)

    except Exception as e:
        context.logger.error(f"Error running agent: {e}\n{traceback.format_exc()}")
        return response.json({
            "error": "agent_failed",
            "message": str(e),
            "hint": ("Verify OPENAI_API_KEY and payload shape. You can send an image as a data URL in JSON:\n"
                     '{"dataUrl":"data:image/jpeg;base64,...."}'),
        })

async def generate_and_store_video(context: AgentContext, request: AgentRequest, prompt: str) -> Dict[str, Any]:
    url = "https://ops-5--example-wan2-generate-video-http.modal.run"
    headers = {"Accept": "video/mp4"}
    params = {"prompt": prompt}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client_httpx:
            r = await client_httpx.get(url, params=params, headers=headers)
            if r.status_code != 200:
                context.logger.error(f"Video endpoint returned status {r.status_code}: {r.text}")
                return {"error": "video_endpoint_error", "status": r.status_code}
            content = r.content
    except Exception as e:
        context.logger.error(f"Error calling video endpoint: {e}")
        return {"error": "video_call_exception"}

    user_id = None
    try:
        user_id = getattr(request, "user_id", None)
        if user_id is None and hasattr(request, "json"):
            body = await _maybe_await(request.json())
            if isinstance(body, dict):
                user_id = body.get("user_id")
    except Exception:
        pass
    if not user_id:
        user_id = "owner"

    key = f"video:{int(datetime.utcnow().timestamp())}:{user_id}"
    try:
        await context.kv.set("videos", key, base64.b64encode(content).decode())
        return {"key": key, "size_bytes": len(content)}
    except Exception as e:
        context.logger.warn(f"Could not persist generated video to KV: {e}")
        return {"size_bytes": len(content)}
