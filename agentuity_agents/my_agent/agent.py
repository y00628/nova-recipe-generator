from __future__ import annotations

"""
Agentuity PantryChef Agent — drop-in replacement for your snippet
- Accepts an image receipt or plain-text ingredients
- Extracts/normalizes ingredients (via GPT-4o-mini for vision or text parsing)
- Estimates expiration dates (heuristic)
- Generates concise recipe ideas with steps (LLM)
- Optionally fetches relevant YouTube tutorial videos

Requires
  pip install httpx
  (You already have: agentuity, openai)

Env
  OPENAI_API_KEY      – for LLM & vision
  YOUTUBE_KEY         – optional, for tutorial lookup
  SPOONACULAR_KEY     – optional, for richer matches (not used by default here)
"""
import os
import re
import json
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from agentuity import AgentRequest, AgentResponse, AgentContext
from openai import AsyncOpenAI

client = AsyncOpenAI()

# --------------------
# Normalization + expiry
# --------------------
CANON = [
    "milk","eggs","butter","yogurt","chicken breast","ground beef","spinach","lettuce",
    "tomato","onion","garlic","carrot","potato","rice","pasta","bread","tortilla",
    "cheddar","mozzarella","salmon","tofu","mushroom","bell pepper","broccoli","zucchini",
    "cilantro","basil","lime","lemon","avocado","cucumber","olive oil","canola oil",
    "flour","sugar","brown sugar","yeast","baking powder","baking soda","salt","pepper",
    "cumin","paprika","chili powder","soy sauce","vinegar","oats","almond milk",
]

EXPIRY_RULES_DAYS = {
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

# --------------------
# Parsers
# --------------------
LIST_SPLIT = re.compile(r"[,\n]|\band\b", re.I)


def parse_free_text(text: str) -> List[str]:
    parts = [p.strip() for p in LIST_SPLIT.split(text or "")]
    items = [re.sub(r"\s+\d+.*$", "", p).strip() for p in parts]
    items = [i for i in items if len(i) > 1]
    return items

async def extract_from_image(content_type: str, image_bytes: bytes) -> List[str]:
    # GPT-4o-mini vision extraction → returns raw list in JSON
    try:
        data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
        out = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract grocery ingredients/items. Output only JSON: {\"items\":[\"...\"]}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "List all ingredients you see as short names (no quantities)."},
                    {"type": "image_url", "image_url": data_url},
                ]},
            ],
            temperature=0.0,
        )
        raw = out.choices[0].message.content or "{}"
        data = json.loads(raw)
        items = data.get("items") or parse_free_text(raw)  # fallback if model returns text list
        return [str(x) for x in items][:40]
    except Exception:
        # ultra-fallback: ask for plain text and split
        try:
            out = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract grocery ingredients/items, comma-separated."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Return a comma-separated list of items."},
                        {"type": "image_url", "image_url": data_url},
                    ]},
                ],
            )
            return parse_free_text(out.choices[0].message.content or "")
        except Exception:
            return []

# --------------------
# Videos (optional)
# --------------------
async def youtube_search(query: str, max_results: int = 3) -> List[str]:
    key = os.getenv("YOUTUBE_KEY")
    if not key:
        return []
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={max_results}&q={httpx.QueryParams({'q': query})['q']}&key={key}"
    )
    async with httpx.AsyncClient(timeout=20) as client_hx:
        r = await client_hx.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        vids = []
        for it in data.get("items", []):
            vid = (it.get("id") or {}).get("videoId")
            if vid:
                vids.append(f"https://www.youtube.com/watch?v={vid}")
        return vids

# --------------------
# Recipe generation (LLM-first; can swap to Spoonacular)
# --------------------
async def llm_recipes(ingredients: List[str], n: int = 3) -> List[Dict[str, Any]]:
    prompt = (
        "You are a chef. Given ingredients, propose concise recipes.\n"
        "Return ONLY JSON as {\"recipes\":[{\"title\":str,\"steps\":[str,...]}]}.\n"
        f"Ingredients: {', '.join(ingredients)}"
    )
    out = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": "Output only valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.6,
    )
    raw = out.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return (data.get("recipes") or [])[:n]
    except Exception:
        # fallback: create a simple single recipe
        return [{"title": "Pantry Stir-Fry", "steps": [
            "Heat pan and oil.",
            "Add chopped aromatics and vegetables.",
            "Add protein; season.",
            "Toss with sauce; serve with rice or noodles.",
        ]}]

# --------------------
# Agent entrypoints
# --------------------

def welcome():
    return {
        "welcome": "PantryChef Agent ready. Send an image (receipt/fridge) or a text list of ingredients.",
        "prompts": [
            {"data": "Upload a grocery receipt image.", "contentType": "image/jpeg"},
            {"data": "eggs, spinach, mozzarella, tomato", "contentType": "text/plain"},
        ],
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    try:
        ctype = request.data.content_type or "text/plain"

        # 1) Collect raw items
        if "image" in ctype:
            img = await request.data.read()
            raw_items = await extract_from_image(ctype, img)
        else:
            txt = await request.data.text()
            raw_items = parse_free_text(txt)

        # 2) Normalize + attach expirations
        now = datetime.utcnow()
        items_struct = []
        seen = set()
        for raw in raw_items:
            norm = normalize_name(raw)
            if norm in seen:
                continue
            seen.add(norm)
            items_struct.append({
                "name": raw,
                "normalized": norm,
                "expires_at": estimate_expiry(norm, now),
            })

        # 3) Recipes
        recipes = await llm_recipes([it["normalized"] for it in items_struct], n=3)

        # 4) Videos (best-effort)
        videos: Dict[str, List[str]] = {}
        for r in recipes:
            vids = await youtube_search(f"how to make {r.get('title','recipe')}")
            videos[r.get("title", "recipe")] = vids

        payload = {
            "items": items_struct,
            "recipes": recipes,
            "videos": videos,
        }
        return response.json(payload)

    except Exception as e:
        context.logger.error(f"PantryChef error: {e}")
        return response.text("Sorry, there was an error processing your request.")
