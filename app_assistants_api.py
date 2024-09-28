from dotenv import load_dotenv
import chainlit as cl
import json
from movie_functions import get_now_playing_movies, get_showtimes, get_reviews, buy_ticket
import re
from typing_extensions import override
from openai import AssistantEventHandler
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta

load_dotenv()

class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == 'thread.run.requires_action':
            print("Got event that requires action")
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)

    @override
    def on_text_created(self, text: Text):
        print("on_text_created: ", text)

    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text):
        print("on_text_delta: ", delta)

    @override
    def on_text_done(self, text: Text):
        print("on_text_done: ", text)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []
        
        for tool in data.required_action.submit_tool_outputs.tool_calls:
            print(f"tool function name = {tool.function_name}")
            if tool.function.name == "get_movies":
                tool_outputs.append({"tool_call_id": tool.id, "output": "57"})
            elif tool.function.name == "get_reviews":
                tool_outputs.append({"tool_call_id": tool.id, "output": "0.06"})
            
      # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)
 
    def submit_tool_outputs(self, tool_outputs, run_id):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                print(text, end="", flush=True)
            print()
    

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 1500
}

ASSISTANT_INSTRUCTIONS = """\
You are a helpful assistant in providing movie recommendations and helping users select movies by answering their questions and providing 
necessary information. You are able to provide reviews, showtimes and also confirm and complete movie ticket purchases.

"""
 
async def create_assistant():
    assistant = await client.beta.assistants.create(
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

@observe
@cl.on_chat_start
async def on_chat_start():    
    message_history = [{"role": "system", "content": ASSISTANT_INSTRUCTIONS}]
    cl.user_session.set("message_history", message_history)
    current_message_thread = await client.beta.threads.create()
    cl.user_session.set("current_message_thread", current_message_thread)

async def generate_assistant_response(client, gen_kwargs):
    thread = cl.user_session.get("current_message_thread")
    assistant = await create_assistant()
    stream = await client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant.id)
    await stream.until_done()
    #async with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant.id, event_handler=EventHandler()) as stream:
    #    await stream.until_done()


@cl.on_message
@observe
async def on_message_assistant(message: cl.Message):
    current_thread = cl.user_session.get("current_message_thread")

    # Add message to current thread.
    message_oai = await client.beta.threads.messages.create(thread_id=current_thread.id, role="user", content=message.content)
    
    await generate_assistant_response(client, gen_kwargs)

def extract_json(text):
    # Regular expression to capture JSON-like objects
    json_regex = r'(\{[^{}]*\})'

    # Find the first match
    matches = re.search(json_regex, text)

    if matches:
            # Get the prefix, json string, and postfix
        json_str = matches.group(1)
        prefix = text[:matches.start()]
        postfix = text[matches.end():]
        
        try:
            # Parse the matched JSON string into a Python dictionary
            json_obj = json.loads(json_str)
            print("Prefix:", prefix)
            prefix = prefix.replace("```json", "")
            print("Extracted JSON string:", json_str)
            print("Postfix:", postfix)
            print("Parsed JSON object:", json_obj)
            return (prefix, json_obj, postfix)
        except json.JSONDecodeError:
            print("Matched string is not a valid JSON object")
    else:
        print("No JSON object found")
    return (None, None, None)

# Extract function call parsing into a separate function
def parse_function_call(content):
    try:
        function_call = json.loads(content)
        if "function_name" in function_call:
            return function_call
    except json.JSONDecodeError:
        print ("Error parsing function call", content)
        pass
    return None

async def confirm_ticket_purchase(theater, movie, showtime):
    res = await cl.AskActionMessage(
        content=f"Confirm purchase of ticket for {movie} at {theater} for showtime {showtime} ",
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
        ], ).send()

    if res and res.get("value") == "continue":
        #await cl.Message(
        #    content="Continue!",
        #).send()
        return "Confirmed"
    return None

if __name__ == "__main__":
    cl.main()