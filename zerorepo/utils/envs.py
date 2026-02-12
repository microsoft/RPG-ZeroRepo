import os
from typing import cast

# Whether or not to enable the thinking mode (yes or no)
THINKING = os.environ.get("THINKING", "yes") == "yes"
# If enabled, what's the enclosing tag for fetching the answer from output
ANSWER_START_TAG = os.environ.get("ANSWER_START_TAG", "<tool_call>")
ANSWER_END_TAG = os.environ.get("ANSWER_END_TAG", "</tool_call>")
# Where to put temporary generated files during processing & reranking
PLAYGROUND_DIR = os.getenv("PLAYGROUND_DIR", "playground")
# Preprocessed structure information for each SWE-Bench problem
# Please download it from the original Agentless repository

# The path to the HF tokenizer model for counting context tokens
# Or tiktoken model name
# TOKENIZER_MODEL = cast(str, os.getenv("TOKENIZER_MODEL", None))
# assert TOKENIZER_MODEL is not None
# The tokenizer type to use for counting tokens
# hf or tiktoken
# TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "hf")
# assert TOKENIZER_TYPE in ["hf", "tiktoken"], f"Invalid TOKENIZER_TYPE: {TOKENIZER_TYPE}
TOKENIZER_MODEL = cast(str, os.getenv("TOKENIZER_MODEL", "gpt-4o"))
assert TOKENIZER_MODEL is not None
# The tokenizer type to use for counting tokens
# hf or tiktoken
CODE_OMITE = "\n# ...(code omitted)...\n"
TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "tiktoken")
assert TOKENIZER_TYPE in ["hf", "tiktoken"], f"Invalid TOKENIZER_TYPE: {TOKENIZER_TYPE}"
