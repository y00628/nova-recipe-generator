"""Test ScottyLabs video generation endpoint"""
import httpx
import asyncio

async def test_video_generation():
    url = "https://ops-5--example-wan2-generate-video-http.modal.run"

    # Simple recipe prompt
    prompt = "A chef cooking stir fry with chicken, rice, and vegetables in a wok"

    print(f"Testing video generation endpoint...")
    print(f"Prompt: {prompt}")
    print(f"\nCalling endpoint (this may take 1-2 minutes)...")

    try:
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                params={"prompt": prompt},
                headers={"Accept": "video/mp4"}
            )

            print(f"\nStatus Code: {response.status_code}")

            if response.status_code == 200:
                # Save the video
                output_file = "test_generated_video.mp4"
                with open(output_file, "wb") as f:
                    f.write(response.content)

                print(f"[SUCCESS] Video saved to: {output_file}")
                print(f"Video size: {len(response.content)} bytes ({len(response.content)/1024/1024:.2f} MB)")
                return True
            else:
                print(f"[FAILED] Status {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return False

    except httpx.TimeoutException:
        print("[ERROR] Request timed out (took longer than 3 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_video_generation())
    if success:
        print("\n[SUCCESS] Video generation endpoint is working!")
    else:
        print("\n[FAILED] Video generation endpoint failed")
