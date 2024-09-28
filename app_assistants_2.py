
import os
from io import BytesIO
from pathlib import Path
from typing import List

from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

from literalai.helper import utc_now

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element


async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ASSISTANT_INSTRUCTIONS = """\
You are a helpful assistant in providing movie recommendations and helping users select movies by answering their questions and providing 
necessary information. You are able to provide reviews, showtimes and also confirm and complete movie ticket purchases.

"""

async def create_assistant():
    assistant = await async_openai_client.beta.assistants.create(
    instructions = ASSISTANT_INSTRUCTIONS,
    model="gpt-4o",
    tools=[
        {
        "type": "function",
        "function": {
            "name": "get_movies",
            "description": "Get a list of movies currently playing. For each movie, it also returns a movie ID.",
        }
        },
        {
        "type": "function",
        "function": {
            "name": "get_showtimes",
            "description": "Get the showtimes for a specific movie ID and a specific location.",
            "parameters": {
            "type": "object",
            "properties": {
                "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA"
                },
                "movie_id": {
                "type": "string",
                "description": "The ID of the movie. The ID is obtained from the get_movies() function call."
                }
            },
            "required": ["location", "movie_id"]
            }
        }
        },
        {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "Get the reviews for a specific movie ID.",
            "parameters": {
            "type": "object",
            "properties": {
                "movie_id": {
                "type": "string",
                "description": "The ID of the movie. The ID is obtained from the get_movies() function call."
                }
            },
            "required": ["movie_id"]
            }
        }
        },
        {
        "type": "function",
        "function": {
            "name": "confirm_ticket_purchase",
            "description": "Confirm purchase of a movie ticket for a specific movie, theater and showtime.",
            "parameters": {
            "type": "object",
            "properties": {
                "theater": {
                "type": "string",
                "description": "The name/location of the movie theater."
                },
                "movie": {
                "type": "string",
                "description": "Title of the movie."
                },
                "showtime": {
                "type": "string",
                "description": "Showtime for the movie at the given theater."
                }
            },
            "required": ["theater", "movie", "showtime"]
            }
        }
        },
        {
        "type": "function",
        "function": {
            "name": "buy_ticket",
            "description": "Purchase of a movie ticket for a specific movie, theater and showtime.",
            "parameters": {
            "type": "object",
            "properties": {
                "theater": {
                "type": "string",
                "description": "The name/location of the movie theater."
                },
                "movie": {
                "type": "string",
                "description": "Title of the movie."
                },
                "showtime": {
                "type": "string",
                "description": "Showtime for the movie at the given theater."
                }
            },
            "required": ["theater", "movie", "showtime"]
            }
        }
        }
    ]
    )
    return assistant

async def get_or_create_assistant():
    assistant = sync_openai_client.beta.assistants.retrieve(
        os.environ.get("OPENAI_ASSISTANT_ID")
    )
    if not assistant:
        assistant = await create_assistant()
    return assistant



class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot): 
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()  
                 
        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool"
                        )
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)


    async def on_tool_call_done(self, tool_call):
        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]


@cl.on_chat_start
async def start_chat():
    assistant = await get_or_create_assistant()
    cl.user_session.set("assistant", assistant)
    config.ui.name = assistant.name
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    await cl.Message(content=f"Hello, I'm {assistant.name}!", disable_feedback=True).send()
    

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()
