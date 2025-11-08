"""Simple test script to validate the agent without Agentuity infrastructure"""
import asyncio
import os
from typing import Optional

# Load env variables
from dotenv import load_dotenv
load_dotenv()

# Mock the agentuity classes for testing
class MockAgentRequest:
    class MockData:
        def __init__(self, text_content=None, binary_content=None, content_type="text/plain"):
            self._text = text_content
            self._binary = binary_content
            self.contentType = content_type

        async def text(self):
            return self._text

        async def binary(self):
            return self._binary

        async def json(self):
            return {}

    def __init__(self, text_content=None):
        self.data = self.MockData(text_content=text_content)

    def get(self, key, default=None):
        return default


class MockAgentResponse:
    def json(self, data, metadata=None):
        return data

    def text(self, data, metadata=None):
        return data


class MockAgentContext:
    class MockLogger:
        def info(self, msg, *args):
            print(f"[INFO] {msg % args if args else msg}")

        def error(self, msg, *args):
            print(f"[ERROR] {msg % args if args else msg}")

        def warn(self, msg, *args):
            print(f"[WARN] {msg % args if args else msg}")

    def __init__(self):
        self.logger = self.MockLogger()


async def test_text_input():
    """Test the agent with text input"""
    print("\n" + "="*60)
    print("Testing agent with text input (ingredient list)")
    print("="*60 + "\n")

    # Import the agent
    from agentuity_agents.my_agent.agent import run

    # Create mock request with text
    request = MockAgentRequest(text_content="chicken, rice, tomatoes, garlic, onions")
    response = MockAgentResponse()
    context = MockAgentContext()

    # Run the agent
    result = await run(request, response, context)

    print("\n" + "-"*60)
    print("RESULT:")
    print("-"*60)
    print(result)
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_text_input())
