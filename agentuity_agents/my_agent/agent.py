from agentuity import AgentRequest, AgentResponse, AgentContext
from openai import AsyncOpenAI
import base64

client = AsyncOpenAI()

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

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    try:
        content_type = request.data.content_type
        user_input = None

        # 🧾 Case 1: Receipt or photo input
        if "image" in content_type:
            image_bytes = await request.data.read()
            # Use GPT-4o (multimodal) to read ingredients
            vision_result = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract a list of ingredients from receipt or fridge photo."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "List all ingredients you see."},
                        {"type": "image_url", "image_url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"}
                    ]}
                ]
            )
            user_input = vision_result.choices[0].message.content

        # 🧠 Case 2: Plain text list
        else:
            user_input = await request.data.text()

        # 🍳 Generate recipe recommendation
        recipe_result = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a chef assistant that creates concise recipes from given ingredients."},
                {"role": "user", "content": f"I have: {user_input}. Suggest a recipe and describe steps."}
            ]
        )

        recipe_text = recipe_result.choices[0].message.content

        # 🎥 Optional: Generate tutorial video using text-to-video API
        # (pseudo-code — integrate with Pika, Runway, or OpenAI video model)
        # video_url = await generate_video_from_text(recipe_text)

        return response.json({
            "ingredients": user_input,
            "recipe": recipe_text,
            # "video_url": video_url
        })

    except Exception as e:
        context.logger.error(f"Error running agent: {e}")
        return response.text("Sorry, there was an error processing your request.")