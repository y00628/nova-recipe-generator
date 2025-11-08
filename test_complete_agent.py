"""Complete agent test - mimics full PantryChef functionality"""
import streamlit as st
import asyncio
import base64
import json
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Configure OpenAI client to use OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)

# Ingredient normalization
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

def _ngrams(s: str, n: int):
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

def estimate_expiry(name: str, start=None):
    days = EXPIRY_RULES_DAYS.get(name)
    if not days:
        return None
    base = start or datetime.utcnow()
    return (base + timedelta(days=days)).isoformat()

async def extract_from_image(image_bytes: bytes):
    """Extract ingredients from image using vision API"""
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract grocery ingredients/items. Output only JSON: {\"items\":[\"...\"]}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "List all ingredients you see as short names (no quantities)."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )

        raw = response.choices[0].message.content or ""
        obj = json.loads(raw.strip().replace("```json", "").replace("```", ""))
        items = obj.get("items", [])
        return items

    except Exception as e:
        st.error(f"Vision API failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

async def generate_recipe(items_text: str):
    """Generate recipe from ingredients"""
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
                {"role": "user", "content": f"I have: {items_text}. Suggest a recipe and describe steps."}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No recipe generated."
    except Exception as e:
        st.error(f"Recipe generation failed: {e}")
        return f"Error: {e}"

st.title("🍳 PantryChef Complete Test")
st.write("Test the complete agent functionality: image extraction + recipe generation")

uploaded_file = st.file_uploader("Choose a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Receipt", use_container_width=True)

    # Read image bytes
    image_bytes = uploaded_file.read()
    st.write(f"**Image size:** {len(image_bytes)} bytes")

    if st.button("🚀 Process Complete Agent Flow"):
        with st.spinner("Step 1/3: Extracting ingredients from image..."):
            items_raw = asyncio.run(extract_from_image(image_bytes))

        if items_raw:
            st.success(f"✅ Extracted {len(items_raw)} items from receipt")

            # Normalize and dedupe
            with st.spinner("Step 2/3: Normalizing ingredients..."):
                now = datetime.utcnow()
                seen = set()
                items_struct = []

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

            st.success(f"✅ Normalized to {len(items_struct)} unique ingredients")

            # Show items
            st.subheader("📦 Extracted Items:")
            cols = st.columns(3)
            for i, item in enumerate(items_struct):
                with cols[i % 3]:
                    expiry = item['expires_at']
                    expiry_str = f" (expires: {expiry[:10]})" if expiry else ""
                    st.write(f"• **{item['name']}**")
                    st.caption(f"normalized: {item['normalized']}{expiry_str}")

            # Generate recipe
            items_text = ", ".join([i["normalized"] for i in items_struct])

            with st.spinner("Step 3/3: Generating recipe..."):
                recipe = asyncio.run(generate_recipe(items_text))

            st.success("✅ Recipe generated!")

            st.subheader("👨‍🍳 Recipe Suggestion:")
            st.markdown(recipe)

            # Show full response
            with st.expander("📄 View Complete Agent Response (JSON)"):
                full_response = {
                    "items": items_struct,
                    "recipe": recipe,
                    "videos": [],
                    "generated_video": None
                }
                st.json(full_response)
        else:
            st.error("❌ No items extracted from image")
