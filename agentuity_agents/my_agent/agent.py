from typing import Optional
from agentuity import AgentRequest, AgentResponse, AgentContext
from openai import AsyncOpenAI
import io
import base64
import os

# Configure OpenAI client to use OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)


def welcome():
    return {
        "welcome": "Welcome to the OpenAI Python Agent! I can help you build AI-powered applications using OpenAI models.",
        "prompts": [
            {
                "data": "How do I implement streaming responses with OpenAI models?",
                "contentType": "text/plain"
            },
            {
                "data": "What are the best practices for prompt engineering with OpenAI?",
                "contentType": "text/plain"
            }
        ]
    }


async def _get_content_type(request: AgentRequest) -> Optional[str]:
    """Try multiple ways to get the incoming content type.

    The Agentuity runtime provides `request.data.contentType()` in some SDKs; fall back
    to known attributes if necessary so the code is robust across environments.
    """
    try:
        # preferred async accessor
        ct = await request.data.contentType()
        return ct
    except Exception:
        # fallback attribute names that some runtimes use
        for attr in ("content_type", "contentType"):
            ct = getattr(request.data, attr, None)
            if ct:
                return ct
        # final fallback: top-level request attribute
        return getattr(request, "content_type", None)


async def _read_image_bytes(request: AgentRequest) -> Optional[bytes]:
    """Try to read image bytes from request using common interfaces.

    Returns None if reading fails.
    """
    # Try the documented binary accessor first
    try:
        return await request.data.binary()
    except Exception:
        pass

    # Some runtimes provide a generic read() coroutine
    try:
        return await request.data.read()
    except Exception:
        pass

    # Some runtimes might expose .body or .bytes attributes
    try:
        b = getattr(request.data, "body", None) or getattr(request.data, "bytes", None)
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
    except Exception:
        pass

    return None


async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """Agent handler: accepts text or image input and returns ingredients + a recipe.

    When an image is provided, the agent will try to read bytes from the request and
    then either run OCR (if `pytesseract` is installed) or attempt to call a vision-capable
    model. If both fail, it returns a helpful hint to the caller explaining how to enable OCR.
    """
    try:
        content_type = await _get_content_type(request)
        user_input = None

        # If we see an image content-type, try to read the binary payload
        if content_type and content_type.startswith("image"):
            image_bytes = await _read_image_bytes(request)
            if not image_bytes:
                context.logger.error("Failed to read image bytes from request")
                return response.json({
                    "ok": False,
                    "error": "Could not read image payload. Ensure the request sends binary image data and that the runtime supports request.data.binary()/read().",
                    "hint": "If you can't change the client, try sending the ingredients as text instead, or install pytesseract and enable OCR in the agent."
                })

            # Try OCR locally if available
            try:
                import pytesseract
                from PIL import Image

                img = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(img)
                user_input = text
                context.logger.info("Extracted text from image using pytesseract for request")
            except Exception as e:
                context.logger.info("Local OCR not available or failed: %s", e)

                # As a fallback, try sending the base64 data to a vision-capable model.
                # Note: this depends on the model and client supporting data URIs in messages.
                try:
                    b64 = base64.b64encode(image_bytes).decode()
                    vision_result = await client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Extract a list of ingredients from the image. List only the ingredients you can identify, separated by commas."},
                            {"role": "user", "content": [
                                {
                                    "type": "text",
                                    "text": "Please extract the ingredients from this receipt image."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{b64}"
                                    }
                                }
                            ]}
                        ],
                    )
                    user_input = vision_result.choices[0].message.content
                except Exception as e2:
                    context.logger.warn("Vision model call failed: %s", e2)
                    return response.json({
                        "ok": False,
                        "error": "OCR is not available and the vision model call failed.",
                        "hint": "To enable OCR locally: pip install pytesseract Pillow and install the Tesseract binary; or send ingredients as text."
                    })

        else:
            # Not an image: read text payload
            try:
                # prefer documented text accessor
                user_input = await request.data.text()
            except Exception:
                # fallback: try .json() then cast to string
                try:
                    j = await request.data.json()
                    user_input = str(j)
                except Exception:
                    user_input = None

        if not user_input:
            return response.json({"ok": False, "error": "No input detected. Send a text list of ingredients or an image containing a receipt/photo."})

        # Use the ingredients (user_input) to ask the recipe model
        recipe_result = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
                {"role": "user", "content": f"I have: {user_input}. Suggest one or two simple recipes and describe steps."}
            ],
        )

        recipe_text = recipe_result.choices[0].message.content

        return response.json({
            "ok": True,
            "ingredients": user_input,
            "recipe": recipe_text,
        })

    except Exception as e:
        context.logger.error("Error running agent: %s", e)
        return response.text("Sorry, there was an error processing your request.")
