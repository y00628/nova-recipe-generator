"""Complete Streamlit app: Receipt -> Ingredients -> Recipe -> Step Images"""
import streamlit as st
import asyncio
import base64
import json
import os
import re
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
            {"role": "system", "content": "You are a chef assistant. Create a recipe with clear numbered steps."},
            {"role": "user", "content": f"I have: {items_text}. Create a simple recipe with 4-5 numbered steps. Format: Step 1: ..., Step 2: ..., etc."}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or "No recipe generated."

async def generate_image_prompts(recipe_text: str):
    """Use LLM to intelligently create image prompts for each cooking step"""

    prompt = f"""Given this recipe, extract the main cooking steps and create visual prompts for animated cooking illustrations.

Recipe:
{recipe_text}

For each major cooking step, generate a prompt for a simple, friendly animated illustration in cooking animation style.
Return ONLY a JSON array with this format:
{{"steps": [{{"step_number": 1, "step_description": "brief step", "image_prompt": "Simple animated illustration: [describe the action in a fun, cartoon style]"}}]}}

Limit to 4-5 key steps. Use simple, cheerful animation style."""

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("steps", [])

async def generate_step_images(recipe_text: str):
    """Generate step-by-step cooking images using LLM-generated prompts"""

    # Use LLM to intelligently create image prompts
    steps = await generate_image_prompts(recipe_text)

    images = []

    for step_data in steps:
        try:
            # Get the animated-style prompt from LLM
            image_prompt = step_data.get("image_prompt", "")

            # Use OpenRouter's image generation via chat/completions
            response = await client.chat.completions.create(
                model="google/gemini-2.5-flash-image-preview",
                messages=[
                    {"role": "user", "content": image_prompt}
                ],
                modalities=["image", "text"],
                extra_body={
                    "image_config": {
                        "aspect_ratio": "16:9"
                    }
                }
            )

            # Extract base64 image from response
            message = response.choices[0].message
            if hasattr(message, 'images') and message.images:
                image_data = message.images[0]
                image_url = image_data.image_url.url if hasattr(image_data, 'image_url') else image_data.get('image_url', {}).get('url')

                images.append({
                    "step_number": step_data.get("step_number", 0),
                    "step_text": step_data.get("step_description", ""),
                    "url": image_url,
                })
        except Exception as e:
            st.warning(f"Could not generate image for step {step_data.get('step_number', '?')}: {str(e)}")
            continue

    return images

st.set_page_config(page_title="PantryChef with Step Images", page_icon="🍳", layout="wide")

st.title("🍳 PantryChef - Recipe with Visual Steps")
st.write("Upload a receipt → Extract ingredients → Generate recipe → See cooking steps!")

uploaded_file = st.file_uploader("📸 Upload Receipt Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Receipt")
        st.image(uploaded_file, use_container_width=True)

    image_bytes = uploaded_file.read()

    if st.button("🚀 Generate Recipe with Visual Steps", type="primary", use_container_width=True):

        # Step 1: Extract Ingredients
        with st.spinner("Step 1/3: Extracting ingredients from receipt..."):
            items = asyncio.run(extract_from_image(image_bytes))

        if items:
            st.success(f"✅ Extracted {len(items)} ingredients!")

            with col2:
                st.subheader("🛒 Extracted Ingredients")
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

            # Step 3: Generate Step Images
            with st.spinner("Step 3/3: Generating cooking step images... (may take 1-2 minutes)"):
                step_images = asyncio.run(generate_step_images(recipe))

            if step_images:
                st.success(f"✅ Generated {len(step_images)} step images!")

                st.subheader("📸 Step-by-Step Visual Guide")

                # Display each step with its image
                for img_data in step_images:
                    st.markdown(f"### Step {img_data['step_number']}: {img_data['step_text']}")
                    st.image(img_data['url'], use_container_width=True)
                    st.markdown("---")

                st.balloons()
            else:
                st.warning("⚠️ Could not generate step images (API limit or error)")
        else:
            st.error("❌ No ingredients extracted from receipt")

st.sidebar.markdown("### 💡 About")
st.sidebar.info("""
This demo:
1. Reads receipt images
2. Extracts ingredients
3. Generates recipes
4. Creates step-by-step cooking images

Powered by OpenRouter + OpenAI
""")
