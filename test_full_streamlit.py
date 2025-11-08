"""Complete Streamlit app: Receipt -> Ingredients -> Recipe -> Video"""
import streamlit as st
import asyncio
import base64
import json
import os
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Configure OpenAI client to use OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)

async def extract_from_image(image_bytes: bytes):
    """Extract ingredients from receipt image"""
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

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
    return obj.get("items", [])

async def generate_recipe(items: list):
    """Generate recipe from ingredients"""
    items_text = ", ".join(items[:15])

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
            {"role": "user", "content": f"I have: {items_text}. Suggest ONE simple recipe with steps. Keep it under 150 words."}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or "No recipe generated."

async def generate_video(recipe_text: str):
    """Generate video from recipe"""
    url = "https://ops-5--example-wan2-generate-video-http.modal.run"
    prompt = f"A chef cooking: {recipe_text[:150]}"

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as http_client:
        response = await http_client.get(
            url,
            params={"prompt": prompt},
            headers={"Accept": "video/mp4"}
        )

        if response.status_code == 200:
            return response.content
        return None

st.set_page_config(page_title="PantryChef Full Demo", page_icon="🍳", layout="wide")

st.title("🍳 PantryChef Complete Demo")
st.write("Upload a receipt → Extract ingredients → Generate recipe → Create video")

uploaded_file = st.file_uploader("📸 Upload Receipt Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Receipt")
        st.image(uploaded_file, use_container_width=True)

    image_bytes = uploaded_file.read()

    if st.button("🚀 Process Complete Pipeline", type="primary", use_container_width=True):

        # Step 1: Extract Ingredients
        with st.spinner("Step 1/3: Extracting ingredients from receipt..."):
            items = asyncio.run(extract_from_image(image_bytes))

        if items:
            st.success(f"✅ Extracted {len(items)} ingredients!")

            with col2:
                st.subheader("🛒 Extracted Ingredients")

                # Show in columns
                cols = st.columns(3)
                for i, item in enumerate(items):
                    with cols[i % 3]:
                        st.write(f"• {item}")

            # Step 2: Generate Recipe
            with st.spinner("Step 2/3: Generating recipe..."):
                recipe = asyncio.run(generate_recipe(items))

            st.success("✅ Recipe generated!")

            st.subheader("👨‍🍳 Recipe")
            st.markdown(recipe)

            # YouTube Tutorial Links
            st.subheader("📺 YouTube Tutorial Videos")
            # Extract first ingredient for search
            first_ingredient = items[0] if items else "recipe"
            search_query = f"how to cook {first_ingredient}"
            youtube_search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"

            st.markdown(f"[🔍 Search YouTube for '{search_query}']({youtube_search_url})")

            # Embed a sample tutorial (you can customize this)
            st.info("💡 Tip: The agent can also fetch specific YouTube tutorial links with a YouTube API key!")

            # Step 3: Generate Video (Optional)
            with st.spinner("Step 3/3: Generating video... (this may take 1-2 minutes)"):
                video_bytes = asyncio.run(generate_video(recipe))

            if video_bytes:
                st.success("✅ Video generated!")

                st.subheader("🎬 Generated Video")
                st.video(video_bytes)

                # Download button
                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name="pantrychef_recipe.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

                st.balloons()
            else:
                st.error("❌ Video generation failed")
        else:
            st.error("❌ No ingredients extracted from receipt")
