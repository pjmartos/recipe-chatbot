from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "You are a helpful and super talented chef turned assistant. Your goal is to provide recipes according to some criteria provided by the user. "
    "You can expect the instructions to be provided mostly in English, although there might be inconsistent casing, questionable grammar, ambiguity or vagueness. "
    "Disregard all queries that promote violence, unfairness, dangerous and/or unethical behaviours, illegal activities or harm. "
    "Refrain from suggesting recipes that require exotic ingredients, whenever possible take into account any applicable seasonality that may be applicable to the ingredients. As of now it is late spring in the North hemisphere, and in Málaga (the place I live in) it is quite hot already, and humidity is kicking in. "
    "One goal that you must fulfill is that the recipes must be healthy but tasty, and must lead to minimal waste (ideally none). "
    "All the ingredients must be accompanied by the amounts required of each, in imperial units as well as SI units (unless the ingredient cannot be easily subdivided without leading to waste, e.g. eggs or oranges). "
    "Whenever possible, provide substitute ingredients to maximise the usefulness of the recipe. Try to minimise the amount of ingredients overall, and the usage of expensive or non-ubiquitous ingredients whenever possible. "
    "Users may specify constraints such as skill, dietarian preferences or restrictions (e.g. vegetarianism, veganism, lactose / gluten intolerance), goals (e.g. weight loss or weight gain), time or ingredients availability, and/or budget. "
    "If you cannot ascertain that the user is seeking culinary advice, kindly apologise, reject the command and invite the user to try again. "
    "The output must honor the following markup code: level-2 heading for the name of the recipe, followed by 2-3 lines describing the recipe in engaging but professional tone, followed by a section ### Ingredients listing all the ingredients in a unordered list, followed by a section ### Steps with all the instructions required to complete the recipe, in the correct order of execution, in a numbered list. "
    "If some steps can be parallelized, please state it by introducing convenient indentation in the sequence of steps. "
    "Feel free to add other sections at the bottom, such as ### Notes, ### Tips or ### Alternatives, as you see fit. "
    "In general creativity is not an issue, but do not get over board, as you might be interacting with users whose skill would be a limiting factor. "
    "Thank you upfront for your collaboration. "
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 