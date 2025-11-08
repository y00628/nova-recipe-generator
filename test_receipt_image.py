"""Test receipt image reading locally - no agentuity needed"""
import asyncio
import base64
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)

async def test_image(image_path: str):
    print(f"\n{'='*60}")
    print(f"Testing receipt image: {image_path}")
    print(f"{'='*60}\n")

    # Read image
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        print(f"[OK] Image loaded: {len(image_bytes)} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to read image: {e}")
        return

    # Convert to base64
    b64_image = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64_image}"
    print(f"[OK] Converted to base64")

    # Test vision API
    print(f"\n{'='*60}")
    print("Calling OpenRouter Vision API...")
    print(f"{'='*60}\n")

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

        result = response.choices[0].message.content
        print("[SUCCESS] Vision API returned:")
        print(f"\n{'='*60}")
        print(result)
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[ERROR] Vision API failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_receipt_image.py <path_to_receipt_image>")
        print("\nExample:")
        print('  python test_receipt_image.py "C:\\Users\\Lenovo\\Downloads\\receipt.jpg"')
        print("  python test_receipt_image.py receipt.png")
        sys.exit(1)

    image_path = sys.argv[1]
    asyncio.run(test_image(image_path))
