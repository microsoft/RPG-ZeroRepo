import re
from typing import Dict, List
import tiktoken
from .envs import ANSWER_END_TAG, ANSWER_START_TAG, THINKING


def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)  # get the encoder for the given model
    tokens = encoding.encode(text)  # encode text into tokens
    return len(tokens)  # return token count


def parse_thinking_output(output: str) -> str:
    """Extract the <solution> part for thinking models"""
    if THINKING:
        output = output.split(ANSWER_START_TAG, 1)[-1]
        output = output.split(ANSWER_END_TAG, 1)[0]
    return output.strip()


def parse_solution_output(output: str) -> str:
    """Extract the <solution> part for thinking models"""
    
    output = output.split("<solution>", 1)[-1]
    output = output.split("</solution>", 1)[0]
    return output.strip()


def parse_thinking(output: str) -> str:
    """Extract the <solution> part for thinking models"""
    if THINKING:
        output = output.split("<think>", 1)[-1]
        output = output.split("</think>", 1)[0]
    return output.strip()


def truncate_by_token(
    text: str,
    max_tokens: int = 50000,
    model: str = "gpt-4o",
) -> str:
    """
    Truncate text by token count:
    - If the token count does not exceed max_tokens, return the text as-is.
    - If it exceeds max_tokens, keep the head and tail tokens and remove a middle segment.
    """

    model_to_encoding: Dict[str, str] = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "code-davinci-002": "p50k_base",
    }

    encoding_name = model_to_encoding.get(model, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    tokens = enc.encode(text)
    total = len(tokens)

    if total <= max_tokens:
        return text

    # Total number of tokens to keep
    keep = max_tokens

    # Simple strategy: keep half at the head and half at the tail;
    # if odd, give the extra token to the head
    head_keep = keep // 2 + keep % 2
    tail_keep = keep // 2

    # Extreme-case safeguard: ensure at least 1 token on each side (if max_tokens >= 2)
    if keep >= 2:
        head_keep = max(1, head_keep)
        tail_keep = max(1, tail_keep)

    # Number of tokens removed from the middle
    removed = total - (head_keep + tail_keep)
    if removed <= 0:
        # Should not happen in theory, but keep as a safeguard
        return text

    head_tokens = tokens[:head_keep]
    tail_tokens = tokens[-tail_keep:] if tail_keep > 0 else []

    head_str = enc.decode(head_tokens)
    tail_str = enc.decode(tail_tokens)

    # Insert a marker in the middle
    marker = (
        f"\n\n... [stderr output truncated: {removed} tokens omitted in the middle] ...\n\n"
    )

    return head_str + marker + tail_str


def calculate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Calculate the number of tokens in the text."""
    model_to_encoding: Dict[str, str] = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "code-davinci-002": "p50k_base",
    }

    encoding_name = model_to_encoding.get(model, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    # Build a regex pattern matching all special tokens
    specials = enc.special_tokens_set
    pattern = re.compile("|".join(re.escape(s) for s in specials))

    # Remove all special tokens
    cleaned_text = pattern.sub("", text)

    # Encode with disallowed_special disabled
    tokens = enc.encode(cleaned_text, disallowed_special=())

    return len(tokens)


def parse_code_blocks(output: str, type: str = "general") -> List[str]:
    """
    Parse code blocks of a given type from a string.

    Args:
        output (str): The text containing code blocks.
        type (str): The type of code block to parse.
                    Supported types:
                        - "general": ```...```
                        - "python": ```python ... ```
                        - "javascript", "html", etc.

    Returns:
        List[str]: A list of extracted code block contents.
    """
    if type == "general":
        pattern = r"```(?:\n)?(.*?)```"   # Capture general code blocks
    else:
        pattern = rf"```{type}\s+(.*?)```"  # Capture specific language code blocks

    # DOTALL lets '.' match newlines
    matches = re.findall(pattern, output, re.DOTALL)
    return [m.strip() for m in matches]