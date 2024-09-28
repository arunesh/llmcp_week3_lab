from dotenv import load_dotenv
import chainlit as cl
import json
from movie_functions import get_now_playing_movies, get_showtimes, get_reviews, buy_ticket
import re

load_dotenv()

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

SYSTEM_PROMPT = """\
You are a helpful assistant in providing movie recommendations and helping users select movies by answering their questions and providing 
necessary information.
If the user asks for the list of movies currently playing or if you need this list to help answer questions, output a function call formatted like this:

{
    "function_name": "get_movies",
    "rationale": "Explain why would you like to call this function"
}

If you need a list of showtimes for a specific movie and a location, generate a function call as shown below:
{
    "function_name": "get_showtimes",
    "movie_name": "Name of the movie",
    "location": "Location of interest",
    "rationale": "Explain why would you like to call this function"
}

If you need reviews on a specific movie, generate a function call as shown below:
{
    "function_name": "get_reviews",
    "movie_id": "Movie ID provided from the get_movies function above for the movie of interest"
}

If the user wishes to purchase a ticket, follow a two step process. First generate a function call as shown below, to confirm the purchase. If the user
confirms, generate a second function call to make the purchase.

   1. Function call to confirm a ticket purchase:
      {
        "function_name": "confirm_ticket_purchase",
        "theater": "Name of the theater",
        "movie": "Title of the movie",
        "showtime": "Showtime for the movie"
    }

   2. Function call to purchase to purchase a ticket for a movie once the user has confirmed the purchase:
    {
        "function_name": "buy_ticket",
        "theater": "Name of the theater",
        "movie": "Title of the movie",
        "showtime": "Showtime for the movie"
    }

"""

SYSTEM_PROMPT_ALT = """\
You are a helpful assistant in providing movie recommendations and helping users select movies by answering their questions and providing 
necessary information.
If the user asks for the list of movies currently playing or if you need this list to help answer questions, output a function call formatted like this:

{
    "function_name": "get_movies",
    "rationale": "Explain why would you like to call this function"
}

If you need a list of showtimes for a specific movie and a location, generate a function call as shown below:
{
    "function_name": "get_showtimes",
    "movie_name": "Name of the movie",
    "location": "Location of interest",
    "rationale": "Explain why would you like to call this function"
}

If you need reviews on a specific movie, generate a function call as shown below:
{
    "function_name": "get_reviews",
    "movie_id": "Movie ID provided from the get_movies function above for the movie of interest"
}

If the user wishes to confirm a ticket purchase for a specified movie, theater and showtime, generate a function call as shown below.
      {
        "function_name": "confirm_ticket_purchase",
        "theater": "Name of the theater",
        "movie": "Title of the movie",
        "showtime": "Showtime for the movie"
    }

 If the user wishes to purchase a ticket for a specified movie, theater and showtime, generate a function call as shown below:
    {
        "function_name": "buy_ticket",
        "theater": "Name of the theater",
        "movie": "Title of the movie",
        "showtime": "Showtime for the movie"
    }

"""

SYSTEM_PROMPT_FOR_REVIEWS_INTENT = """
Based on the conversation, determine if the topic is about a specific movie. Determine if the user is asking a question that would be aided by knowing what critics are saying about the movie. Determine if the reviews for that movie have already been provided in the conversation. If so, do not fetch reviews.

Your only role is to evaluate the conversation, and decide whether to fetch reviews.

Output the current movie, id, a boolean to fetch reviews in JSON format, and your
rationale. Do not output as a code block.

{
    "movie": "title",
    "id": 123,
    "fetch_reviews": true
    "rationale": "reasoning"
}
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()
    return response_message

async def generate_llmresponse(client, message_history, gen_kwargs):
    llm_response = await client.chat.completions.create(messages=message_history, stream=False, **gen_kwargs)
    # Extract the assistant's response
    if llm_response and llm_response.choices[0]:
        message_content = llm_response.choices[0].message.content
        return message_content
   
    return None

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

@observe
async def post_llmresponse(llm_response, message_history, gen_kwargs):
    response_message = cl.Message(content=llm_response)
    await response_message.send()
    message_history.append({"role": "assistant", "content": response_message.content})

@observe
async def post_userresponse(user_response, message_history, gen_kwargs):
    response_message = cl.Message(content=llm_response)
    await response_message.send()
    message_history.append({"role": "assistant", "content": response_message.content})

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

async def should_fetch_movie_reviews(client, message_history, gen_kwargs):
    temp_history = message_history[1:]
    new_prompt = f"""{SYSTEM_PROMPT_FOR_REVIEWS_INTENT} 
    Conversation History:
           {temp_history}
    End of conversation history.
    """
    new_history = [{"role": "system", "content": new_prompt}]
    response = await generate_llmresponse(client, new_history, gen_kwargs)
    print("--------> Should fetch reviews: ", response)
    try:
        review_json = json.loads(response)
        return review_json
    except json.JSONDecodeError:
        print ("Error parsing function call", response)
        pass
    return None


@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    review_json = await should_fetch_movie_reviews(client, message_history, gen_kwargs)
    if review_json and review_json["fetch_reviews"] == True:
        movie_id = review_json.get("id")
        reviews = get_reviews(movie_id)
        reviews = f"Reviews for {review_json.get('movie')} (ID: {movie_id}):\n\n{reviews}"
        context_message = {"role": "system", "content": f"CONTEXT: {reviews}"}
        message_history.append(context_message)        

    # Determine if there is an indirect semantic intent to fetch reviews.

    # response_message = await generate_response(client, message_history, gen_kwargs)
    # message_history.append({"role": "assistant", "content": response_message.content})

    llm_response = await generate_llmresponse(client, message_history, gen_kwargs)
    print("llm_response 2 = ", llm_response)
    continue_function_calls = True
    function_call_parsing_count = 0
    last_llm_response = llm_response
    while (continue_function_calls and function_call_parsing_count < 10):
        # Parse message.content to check for function call output by OpenAI GPT
        #function_call = parse_function_call(llm_response)
        (prefix, json_obj, postfix) = extract_json(llm_response)
        function_call =  json_obj if json_obj and "function_name" in json_obj else None
        movie_data_message = None
        if function_call:
            # Handle the function call
            if function_call["function_name"] == "get_movies":
                movies = get_now_playing_movies()
                #movie_data_message = await cl.Message(f"Here are the current movies: {movies}").send()
                movie_data_message = cl.Message(f"Here are the current movies: {movies}")
            elif function_call["function_name"] == "get_showtimes":
                showtimes = get_showtimes(function_call["movie_name"], function_call["location"])
                #movie_data_message = await cl.Message(f"Showtimes for {function_call['movie_name']} in {function_call['location']}: {showtimes}").send()
                movie_data_message = cl.Message(f"Showtimes for {function_call['movie_name']} in {function_call['location']}: {showtimes}")
            elif function_call["function_name"] == "get_reviews":
                reviews = get_reviews(function_call["movie_id"])
                #movie_data_message = await cl.Message(f"Reviews for the movie: {reviews}").send()
                movie_data_message = cl.Message(f"Reviews for the movie: {reviews}")
            elif function_call["function_name"] == "confirm_ticket_purchase":
                movie = function_call["movie"]
                theater = function_call["theater"]
                showtime = function_call["showtime"]
                confirmation = await confirm_ticket_purchase(theater, movie, showtime)
                if confirmation:
                    print("User confirmed.")
                    movie_data_message = cl.Message(f"User confirmed the movie purchase: {movie} at theater {theater} for showtime f{showtime}. Proceed for purchase.")
                else:
                    print("User cancelled.")
                    movie_data_message = cl.Message(f"User cancelled the movie purchase: {movie} at theater {theater} for showtime f{showtime}. Ask the user if there interest in any other movie ?")
            elif function_call["function_name"] == "buy_ticket":
                reviews = buy_ticket(function_call["theater"], function_call["movie"], function_call["showtime"])
                movie_data_message = await cl.Message(f"Reviews for the movie: {reviews}").send()
            #if prefix:
            #     await post_llmresponse(prefix, message_history, gen_kwargs)
        if function_call and movie_data_message:
            message_history.append({"role": "system", "content": movie_data_message.content})
            # Get the next round of completions from OAI.
            function_call_parsing_count += 1
            llm_response = await generate_llmresponse(client, message_history, gen_kwargs)
            # llm_response = llm_response.choices[0].message.content
            print("Generating next response:", llm_response)
        else:
            continue_function_calls = False
        last_llm_response = llm_response

    await post_llmresponse(last_llm_response, message_history, gen_kwargs)
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()



#                if confirmation:
#                     message_history.append({"role": "system",
#                      "content": f"User confirmed the movie purchase: {movie} at theater {theater} for showtime f{showtime}. Proceed for purchase."})
#                else:
#                    message_history.append({"role": "system",
#                      "content": f"User cancelled the movie purchase: {movie} at theater {theater} for showtime f{showtime}."})