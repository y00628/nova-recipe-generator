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
DATA_URL_RE = re.compile(r"^data:(?P<ctype>[^;]+);base64,(?P<b64>.+)$", re.I)

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

async def _get_first_file_bytes_from_request(request: AgentRequest, context: AgentContext) -> Optional[bytes]:
    """
    Probe common places Agentuity environments carry uploads:
    - request.files / request.file
    - request.attachments / request.uploads
    Each entry may expose .read(), .bytes(), .stream, .file, .content, .body, .getvalue(), etc.
    """
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
    """
    Aggressively try:
      A) request.data.* interfaces
      B) request.json() dict: image/file/payload/body/dataUrl/data_url/base64/b64
      C) request.files / request.attachments / request.uploads
      D) request.data.text() as data URL or raw base64
      E) request.data as dict with the same keys
    Returns bytes or None (caller may still choose to treat text as data URL path).
    """
    data = getattr(request, "data", None)

    # A) direct binary-like on request.data
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

    # B) request.json() – dict payloads carrying image data or data-url/base64
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

    # C) files/attachments
    from_req = await _get_first_file_bytes_from_request(request, context)
    if from_req is not None:
        return from_req

    # D) data.text() as data URL or raw base64
    if data is not None and hasattr(data, "text"):
        try:
            t = await _maybe_await(data.text())
            if isinstance(t, str) and t.strip():
                decoded = _try_decode_b64(t)
                if decoded is not None:
                    return decoded
        except Exception:
            pass

    # E) request.data as dict with common keys
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

    # Nothing found
    return None

# =========================
# LLM calls (no temperature params)
# =========================

async def extract_from_image(content_type: str, image_bytes: bytes, ctx: AgentContext) -> List[str]:
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
    try:
        out = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract grocery ingredients/items. Output only JSON: {\"items\":[\"...\"]}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "List all ingredients you see as short names (no quantities)."},
                    {"type": "image_url", "image_url": data_url},
                ]},
            ],
        )
        raw = out.choices[0].message.content or ""
        try:
            obj = json.loads(raw)
            items = obj.get("items")
            if isinstance(items, list):
                return [str(x).strip() for x in items if str(x).strip()][:40]
        except Exception:
            pass
        return parse_free_text(raw)
    except Exception as e:
        ctx.logger.error(f"Vision extract failed: {e}\n{traceback.format_exc()}")
        return []

async def llm_recipe_text(items_text: str) -> str:
    comp = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
            {"role": "user", "content": f"I have: {items_text}. Suggest a recipe and describe steps."}
        ],
    )
    return comp.choices[0].message.content or "No recipe generated."

# Optional: YouTube tutorial lookup
async def youtube_search(query: str, max_results: int = 3) -> List[str]:
    key = os.getenv("YOUTUBE_KEY")
    if not key:
        return []
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={max_results}&q={httpx.QueryParams({'q': query})['q']}&key={key}"
    )
    async with httpx.AsyncClient(timeout=20) as hx:
        r = await hx.get(url)
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
DATA_URL_RE = re.compile(r"^data:(?P<ctype>[^;]+);base64,(?P<b64>.+)$", re.I)

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

async def _get_first_file_bytes_from_request(request: AgentRequest, context: AgentContext) -> Optional[bytes]:
    """
    Probe common places Agentuity environments carry uploads:
    - request.files / request.file
    - request.attachments / request.uploads
    Each entry may expose .read(), .bytes(), .stream, .file, .content, .body, .getvalue(), etc.
    """
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
    """
    Aggressively try:
      A) request.data.* interfaces
      B) request.json() dict: image/file/payload/body/dataUrl/data_url/base64/b64
      C) request.files / request.attachments / request.uploads
      D) request.data.text() as data URL or raw base64
      E) request.data as dict with the same keys
    Returns bytes or None (caller may still choose to treat text as data URL path).
    """
    data = getattr(request, "data", None)

    # A) direct binary-like on request.data
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

    # B) request.json() – dict payloads carrying image data or data-url/base64
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

    # C) files/attachments
    from_req = await _get_first_file_bytes_from_request(request, context)
    if from_req is not None:
        return from_req

    # D) data.text() as data URL or raw base64
    if data is not None and hasattr(data, "text"):
        try:
            t = await _maybe_await(data.text())
            if isinstance(t, str) and t.strip():
                decoded = _try_decode_b64(t)
                if decoded is not None:
                    return decoded
        except Exception:
            pass

    # E) request.data as dict with common keys
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

    # Nothing found
    return None

# =========================
# LLM calls (no temperature params)
# =========================

async def extract_from_image(content_type: str, image_bytes: bytes, ctx: AgentContext) -> List[str]:
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
    try:
        out = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract grocery ingredients/items. Output only JSON: {\"items\":[\"...\"]}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "List all ingredients you see as short names (no quantities)."},
                    {"type": "image_url", "image_url": data_url},
                ]},
            ],
        )
        raw = out.choices[0].message.content or ""
        try:
            obj = json.loads(raw)
            items = obj.get("items")
            if isinstance(items, list):
                return [str(x).strip() for x in items if str(x).strip()][:40]
        except Exception:
            pass
        return parse_free_text(raw)
    except Exception as e:
        ctx.logger.error(f"Vision extract failed: {e}\n{traceback.format_exc()}")
        return []

async def llm_recipe_text(items_text: str) -> str:
    comp = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
            {"role": "user", "content": f"I have: {items_text}. Suggest a recipe and describe steps."}
        ],
    )
    return comp.choices[0].message.content or "No recipe generated."

# Optional: YouTube tutorial lookup
async def youtube_search(query: str, max_results: int = 3) -> List[str]:
    key = os.getenv("YOUTUBE_KEY")
    if not key:
        return []
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={max_results}&q={httpx.QueryParams({'q': query})['q']}&key={key}"
    )
    async with httpx.AsyncClient(timeout=20) as hx:
        r = await hx.get(url)
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

        # ========= IMAGE PATHS =========
        if "image" in ctype or ctype.startswith("multipart/"):
            # Try to fetch bytes from *any* supported path
            image_bytes = await _get_bytes_from_anywhere(request, context)
            if image_bytes is None:
                # If we cannot get bytes, we still try the TEXT path below, which now
                # can parse a data URL or base64 in the body and treat it as an image.
                pass
            else:
                items_raw = await extract_from_image(ctype or "image/jpeg", image_bytes, context)

        # ========= TEXT / JSON (also handles data URLs) =========
        if not items_raw:
            txt = ""
            data = getattr(request, "data", None)

            # (1) data.text()
            if data is not None and hasattr(data, "text"):
                try:
                    t = data.text()
                    txt = await _maybe_await(t)
                except Exception:
                    txt = ""

            # (2) request.json()
            if hasattr(request, "json"):
                try:
                    body = await _maybe_await(request.json())
                    if isinstance(body, dict):
                        # Accept plain text under "text"/"ingredients" AND
                        # image data via dataUrl/data_url/base64/b64/image
                        txt = body.get("text") or body.get("ingredients") or txt
                        for k in ("dataUrl", "dataURL", "data_url", "base64", "b64", "image"):
                            v = body.get(k)
                            if isinstance(v, str) and v.strip():
                                txt = v  # let data URL/base64 override plain text
                                break
                except Exception:
                    pass

            # (3) If txt is a data URL or raw base64, treat as image now
            maybe_img = _try_decode_b64(txt) if isinstance(txt, str) else None
            if maybe_img is not None:
                inferred_ctype = "image/jpeg"
                m = DATA_URL_RE.match(txt.strip()) if isinstance(txt, str) else None
                if m:
                    inferred_ctype = m.group("ctype") or inferred_ctype
                items_raw = await extract_from_image(inferred_ctype, maybe_img, context)
            else:
                # Otherwise treat as a plain ingredient list
                items_raw = parse_free_text(txt or "")

        # ========= Normalize + expirations (dedupe) =========
        now = datetime.utcnow()
        seen: set[str] = set()
        items_struct: List[Dict[str, Any]] = []
        for raw in items_raw:
            norm = normalize_name(raw)
            if norm in seen:
                continue
            seen.add(norm)
            items_struct.append({
                "name": raw,
                "normalized": norm,
                "expires_at": estimate_expiry(norm, now),
            })

        # ========= Recipe + Tutorials =========
        items_text = ", ".join([i["normalized"] for i in items_struct]) or ", ".join(items_raw) or "nothing specified"
        recipe_text = await llm_recipe_text(items_text)
        vids = await youtube_search(f"how to make {items_text.split(',')[0].strip()}") if items_text else []

        # === Optional video generation ===
        video_info: Optional[Dict[str, Any]] = None

        # Try to read JSON/body or metadata for generation flags
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

        # If the caller selected a specific recipe to generate, use that
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

        return response.json({
            "items": items_struct,
            "recipe": recipe_text,
            "videos": vids,
            "generated_video": video_info,
        })

    except Exception as e:
        context.logger.error(f"Error running agent: {e}\n{traceback.format_exc()}")
        return response.json({
            "error": "agent_failed",
            "message": str(e),
            "hint": (
                "Verify OPENAI_API_KEY and payload shape. "
                "You can always send an image as a data URL in JSON:\n"
                '{"dataUrl":"data:image/jpeg;base64,...."}'
            ),
        })


async def generate_and_store_video(context: AgentContext, request: AgentRequest, prompt: str) -> Dict[str, Any]:
    """Generate a video via the ScottyLabs endpoint and store it in KV.

    Returns metadata dict with key/size or error.
    """
    url = "https://ops-5--example-wan2-generate-video-http.modal.run"
    headers = {"Accept": "video/mp4"}
    params = {"prompt": prompt}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code != 200:
                context.logger.error(f"Video endpoint returned status {r.status_code}: {r.text}")
                return {"error": "video_endpoint_error", "status": r.status_code}
            content = r.content
    except Exception as e:
        context.logger.error(f"Error calling video endpoint: {e}")
        return {"error": "video_call_exception"}

    # store in KV as base64 so it's JSON-serializable
    key = f"video:{int(datetime.utcnow().timestamp())}:{getattr(request, 'user_id', None) or (await _maybe_await(request.json())).get('user_id') if hasattr(request, 'json') else 'owner'}"
    try:
        await context.kv.set("videos", key, base64.b64encode(content).decode())
        return {"key": key, "size_bytes": len(content)}
    except Exception as e:
        context.logger.warn(f"Could not persist generated video to KV: {e}")
        return {"size_bytes": len(content)}
