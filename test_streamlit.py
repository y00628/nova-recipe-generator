"""Simple Streamlit app to test receipt image reading"""
import streamlit as st
import asyncio
import base64
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Configure OpenAI client to use OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)

async def extract_from_image(image_bytes: bytes):
    """Extract ingredients from image using vision API"""
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract grocery ingredients/items. Output only JSON: {\"items\":[\"...\"]}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "List all ingredients you see as short names (no quantities)."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )

        raw = response.choices[0].message.content or ""

        # Try to parse JSON
        try:
            obj = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            items = obj.get("items", [])
            return items, raw
        except:
            return [], raw

    except Exception as e:
        st.error(f"Vision API failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], str(e)

st.title("🛒 Receipt Image Tester")
st.write("Upload a receipt image to test if the vision API can read it")

uploaded_file = st.file_uploader("Choose a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)

    # Read image bytes
    image_bytes = uploaded_file.read()
    st.write(f"**Image size:** {len(image_bytes)} bytes")

    if st.button("Extract Ingredients"):
        with st.spinner("Calling Vision API..."):
            items, raw_response = asyncio.run(extract_from_image(image_bytes))

        st.success("Vision API completed!")

        st.subheader("Extracted Items:")
        if items:
            for item in items:
                st.write(f"- {item}")
        else:
            st.warning("No items extracted")

        st.subheader("Raw API Response:")
        st.code(raw_response, language="json")
