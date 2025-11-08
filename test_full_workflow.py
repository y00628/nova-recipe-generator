"""Test complete workflow: Receipt -> Ingredients -> Recipe -> Video"""
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

async def extract_from_image(image_path: str):
    """Extract ingredients from receipt image"""
    print("\n[STEP 1] Extracting ingredients from receipt...")

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

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
    items = obj.get("items", [])

    print(f"[SUCCESS] Extracted {len(items)} items:")
    for item in items[:10]:  # Show first 10
        print(f"  - {item}")
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more")

    return items

async def generate_recipe(items: list):
    """Generate recipe from ingredients"""
    print("\n[STEP 2] Generating recipe...")

    items_text = ", ".join(items[:15])  # Use first 15 items

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
            {"role": "user", "content": f"I have: {items_text}. Suggest ONE simple recipe with steps. Keep it under 100 words."}
        ],
        temperature=0.2,
    )

    recipe = response.choices[0].message.content or "No recipe generated."

    print(f"[SUCCESS] Recipe generated:")
    print(f"{recipe}\n")

    return recipe

async def generate_video(recipe_text: str):
    """Generate video from recipe using ScottyLabs endpoint"""
    print("[STEP 3] Generating video from recipe...")
    print("(This may take 1-2 minutes...)")

    url = "https://ops-5--example-wan2-generate-video-http.modal.run"

    # Create a short video prompt from the recipe
    prompt = f"A chef cooking: {recipe_text[:150]}"  # Limit prompt length

    try:
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as http_client:
            response = await http_client.get(
                url,
                params={"prompt": prompt},
                headers={"Accept": "video/mp4"}
            )

            if response.status_code == 200:
                # Save the video
                output_file = "full_workflow_video.mp4"
                with open(output_file, "wb") as f:
                    f.write(response.content)

                print(f"[SUCCESS] Video saved to: {output_file}")
                print(f"Video size: {len(response.content)} bytes ({len(response.content)/1024/1024:.2f} MB)")
                return output_file
            else:
                print(f"[FAILED] Status {response.status_code}")
                return None

    except Exception as e:
        print(f"[ERROR] Video generation failed: {e}")
        return None

async def main():
    print("=" * 60)
    print("COMPLETE WORKFLOW TEST: Receipt -> Recipe -> Video")
    print("=" * 60)

    # Path to your receipt
    receipt_path = "C:\\Users\\Lenovo\\Downloads\\grocery_receipt.jpg"

    # Step 1: Extract ingredients
    items = await extract_from_image(receipt_path)

    if not items:
        print("[ERROR] No items extracted!")
        return

    # Step 2: Generate recipe
    recipe = await generate_recipe(items)

    # Step 3: Generate video
    video_file = await generate_video(recipe)

    print("\n" + "=" * 60)
    if video_file:
        print("[SUCCESS] Complete workflow finished!")
        print(f"Your video is ready: {video_file}")
    else:
        print("[PARTIAL] Recipe generated but video failed")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
