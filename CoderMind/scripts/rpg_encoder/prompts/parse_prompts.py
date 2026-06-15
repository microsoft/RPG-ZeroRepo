"""Parse Prompt Templates.

LLM prompt templates for semantic feature extraction from source code
across any supported language (Python, Go, TypeScript/JavaScript, C/C++,
Rust). Adapted for agent-based execution where the agent reads source
files directly rather than receiving code inline.
"""

PARSE_CLASS = """
## Instruction
You are a senior software analyst, tasked with extracting high-level semantic features from class-like constructs (classes, structs, interfaces, traits, enums, depending on the language).
You will be given a list of target files and class-like definitions. Read each file to understand the implementation, then extract features.

### Key Goals:
- Complete analysis: Provide a full, semantic feature extraction for all specified definitions.
- Exhaustive coverage: Include **every** class-like construct and **every** method (or equivalent — receiver methods, member functions, associated functions, constructors, destructors, lifecycle hooks).
- Focus on purpose and high-level behavior — what each definition represents or manages in the system.
- Summarize what each method is responsible for at a high level, avoiding implementation details.
- If multiple definitions share the same method name, output that method name only once and merge their features.

## Feature Extraction Principles:
1. Focus on the purpose and behavior of each class-like construct — what it represents or manages.
2. For methods, describe their main purpose, not the implementation details.
3. Use the name, its methods, and the surrounding context to infer meaning.
4. If a definition serves multiple functions, list multiple features accordingly.
5. Do not fabricate names or methods that are not in the input.
6. Do not skip any defined method, including constructors / destructors / lifecycle hooks and helper methods.

### Feature Naming Rules:
1. Use the "verb + object" format
   - Example: `load config`, `validate token`
2. Use lowercase English only.
3. Describe purpose, not implementation
   - Focus on *what* the code does, not *how* it does it.
4. Each feature should express one single responsibility.
5. If a method performs multiple responsibilities, create **multiple short features**, each describing only one responsibility.
6. Keep each feature short and atomic:
   - Prefer **3–8 words**.
   - Do **not** write full sentences.
   - Avoid punctuation inside a feature (no commas, periods, or semicolons).
7. Avoid vague verbs:
   - Avoid: `handle`, `process`, `deal with`.
   - Prefer: `load`, `validate`, `convert`, `update`, `serialize`, `compute`, `check`, `transform`, etc.
8. Avoid implementation details:
   - Do not mention loops, conditionals, specific data structures, or control flow.
9. Avoid mentioning specific libraries, frameworks, or formats:
   - Correct: `serialize data`
   - Incorrect: `pickle object`, `save to json`
10. Prefer domain or system semantic words over low-level technical actions:
   - Correct: `manage session`
   - Incorrect: `update dict`
11. Avoid chaining multiple actions in one feature:
   - avoid `initialize config and register globally`
   - prefer `initialize config`, `register config globally`

## Feature Description Rules:
In ADDITION to the short feature name, every feature MUST come with a `description` value:
1. Exactly ONE English sentence, ≤ 25 words.
2. Capture the CORE PURPOSE — what the unit accomplishes and why it exists, NOT how it is implemented.
3. Synthesize meaning from BOTH the code body AND the docstring. When a docstring states intent, prefer that wording; supplement with code-derived behavior if needed.
4. Avoid: implementation details (loops/conditionals/data structures), library or framework names, variable names, leading phrases like "this function" / "this method".
5. Read as a standalone sentence; do NOT reference the surrounding code.
6. If the unit is trivial / stub / generated boilerplate and the purpose is genuinely unclear, output an empty string `""` — do NOT fabricate.

Examples:
  Bad:  "Initializes self.path = path and self.timeout = 30"
  Good: "Configures the loader with the input source and timeout defaults."
  Bad:  "This function reads a file and returns its lines"
  Good: "Loads the target file from disk for downstream processing."

## Output Format:
For every response, you must respond with one `<solution>...</solution>` block containing a JSON object where each key is a class name and each value is either (a) a dictionary mapping method names (including special methods) to a `{{feature_name: description}}` mapping if the class defines methods, or (b) a `{{feature_name: description}}` mapping for class-level features if it has no methods.
<solution>
{{
  "class_name_1": {{
     "feature one": "one-sentence description of feature one",
     "feature two": "one-sentence description of feature two"
  }},
  "class_name_2": {{
     "method_1": {{
        "feature 1": "description of feature 1",
        "feature 2": "description of feature 2"
     }},
     "method_2": {{
        "feature 3": "description of feature 3"
     }}
  }},
  ...
}}
</solution>

### Example Output:
<solution>
{{
  "DataLoader": {{
    "configure": {{
       "initialize data loading configuration": "Configures the loader with the input source and validation defaults."
    }},
    "load_data": {{
       "read dataset from disk": "Pulls the configured dataset into memory for downstream processing.",
       "split data into train and test sets": "Partitions the loaded records into training and evaluation subsets."
    }}
  }},
  "Logger": {{
    "log messages to console or file": "Routes structured runtime messages to the configured sink for observability."
  }}
}}
</solution>

## Input Context
### Repository Name
<repo_name>
{repo_name}
</repo_name>
### Repository Overview
<repo_info>
{repo_info}
</repo_info>
"""


PARSE_FUNCTION = """
## Instruction
You are a senior software analyst.
Your task is to extract high-level semantic features from standalone (module-level) functions across any supported language (Python, Go, TypeScript/JavaScript, C/C++, Rust).
You will be given a list of target files and functions. Read each file to understand the implementation, then extract features.

### Key Goals
- Complete analysis: Provide semantic feature extraction for **every** specified function. Do not skip any function.
- Batch perspective: Consider each function's role within the overall system.
- High-level behavior: Focus on the purpose and role of each function, not on low-level implementation details.
- If multiple definitions share the same function name, output it only once and merge their features.

## Feature Extraction Principles
Follow these principles when analyzing functions:
1. Focus on the purpose and behavior of the function — what role it serves in the system.
2. Do NOT describe implementation details, variable names, or internal logic such as loops, conditionals, or data structures.
3. If a function performs multiple responsibilities, break them down into separate features.
4. Read the source files to understand each function's intent from its name, signature, and code.
5. Only analyze functions specified in the task — do not guess or invent other functions.
6. Do not omit any function, including utility or helper functions.

### Feature Naming Rules:
1. Use the "verb + object" format
   - Example: `load config`, `validate token`
2. Use lowercase English only.
3. Describe purpose, not implementation
   - Focus on *what* the code does, not *how* it does it.
4. Each feature should express one single responsibility.
5. If a method performs multiple responsibilities, create **multiple short features**, each describing only one responsibility.
6. Keep each feature short and atomic:
   - Prefer **3–8 words**.
   - Do **not** write full sentences.
   - Avoid punctuation inside a feature (no commas, periods, or semicolons).
7. Avoid vague verbs:
   - Avoid: `handle`, `process`, `deal with`.
   - Prefer: `load`, `validate`, `convert`, `update`, `serialize`, `compute`, `check`, `transform`, etc.
8. Avoid implementation details:
   - Do not mention loops, conditionals, specific data structures, or control flow.
9. Avoid mentioning specific libraries, frameworks, or formats:
   - Correct: `serialize data`
   - Incorrect: `pickle object`, `save to json`
10. Prefer domain or system semantic words over low-level technical actions:
   - Correct: `manage session`
   - Incorrect: `update dict`
11. Avoid chaining multiple actions in one feature:
   - avoid `initialize config and register globally`
   - prefer `initialize config`, `register config globally`

## Feature Description Rules:
In ADDITION to the short feature name, every feature MUST come with a `description` value:
1. Exactly ONE English sentence, ≤ 25 words.
2. Capture the CORE PURPOSE — what the function accomplishes and why it exists, NOT how it is implemented.
3. Synthesize meaning from BOTH the code body AND the docstring. When a docstring states intent, prefer that wording; supplement with code-derived behavior if needed.
4. Avoid: implementation details (loops/conditionals/data structures), library or framework names, variable names, leading phrases like "this function".
5. Read as a standalone sentence; do NOT reference the surrounding code.
6. If the function is a trivial stub / generated boilerplate and the purpose is genuinely unclear, output an empty string `""` — do NOT fabricate.

Examples:
  Bad:  "Returns true if x > 0 else false"
  Good: "Reports whether the supplied value is strictly positive."
  Bad:  "This function calls os.path.join and returns the result"
  Good: "Builds the canonical on-disk location for the requested resource."

## Output Format
You must respond with the following structure:
A `<solution>` block — a JSON object mapping each function name to a `{{feature_name: description}}` mapping.
If a function does not implement any meaningful features (e.g., it's a stub), still include it with an empty mapping `{{}}`.
### Output Template:
<solution>
{{
  "func_name_1": {{
     "feature one": "description of feature one",
     "feature two": "description of feature two"
  }},
  "func_name_2": {{}},
  ...
}}
</solution>

### Example:
<solution>
{{
  "download_image": {{
     "download image from URL": "Fetches the remote image bytes from the supplied address.",
     "save image to local disk": "Persists the retrieved image into the configured local store."
  }},
  "resize_image": {{
     "resize image to target dimensions": "Rescales an input image to the requested width and height.",
     "update image metadata": "Refreshes the image's recorded properties after transformation."
  }},
  "noop": {{}}
}}
</solution>


## Input Context
### Repository Name
<repo_name>
{repo_name}
</repo_name>
### Repository Overview
<repo_info>
{repo_info}
</repo_info>
"""
