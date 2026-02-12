PARSE_CLASS = """
## Instruction
You are a senior software analyst, tasked with extracting high-level semantic features from one or more Python classes in a complete and comprehensive manner.
Your task is to analyze all the provided classes at once and summarize their purpose and behavior. You should focus on understanding what each class and its methods are responsible for within the system as a whole, not on low-level implementation details.

You are given:
- The repository skeleton (for context)
- The full code of one or more classes (each chunk may contain multiple classes)
- The repository's purpose and background information

### Key Goals:
- Complete analysis: Provide a full, semantic feature extraction for all classes in the given input. Do not process classes separately; treat them as one task and analyze them together.
- Exhaustive coverage: Include **every** class and **every** method defined in the input, including special methods and lifecycle methods (e.g., `__init__`, `__new__`, `__enter__`, `__exit__`), as well as class methods and static methods.
- Focus on the purpose and high-level behavior of the classes — what they represent or manage in the system.
- Summarize what each method is responsible for at a high level, avoiding any implementation details.
- If multiple definitions share the same method name (e.g., property getter and setter for the same attribute), you may output that method name only once and merge their semantic features; you do not need to distinguish decorator variants.

## Feature Extraction Principles:
1. Focus on the purpose and behavior of each class — what it represents or manages.
2. For methods, describe their main purpose, not the implementation details.
3. Use the class name, its methods, and the surrounding context to infer meaning.
4. If a class serves multiple functions, list multiple features accordingly.
5. Do not fabricate class names or methods that are not in the input.
6. Do not skip any defined method, including special methods (e.g., `__init__`, `__new__`, `__repr__`) and helper methods.

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

## Output Format:
For every response, you must respond with one `<solution>...</solution>` block containing a JSON object where each key is a class name and each value is either (a) a dictionary mapping method names (including special methods) to their semantic feature lists if the class defines methods, or (b) a list of class-level features if it has no methods.
<solution>
{{
  "class_name_1": ["feature one", "feature two"],
  "class_name_2": {{
     "method_1": ["feature 1", "feature 2"],
     "method_2": ["feature 3, "feature 3"],
     ...
   }},
  ...
}}
</solution>

### Example Output:
<solution>
{{
  "DataLoader": {{
    "__init__": ["initialize data loading configuration"],
    "load_data": ["read dataset from disk", "split data into train and test sets"]
  }},
  "Logger": ["log messages to console or file"]
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
Your task is to extract high-level semantic features from a group of standalone Python functions.

You are given:
- The skeleton of the full repository (for context)
- The function names and their code blocks (each chunk may contain multiple functions)
- The repository's purpose and background information

Your goal is to analyze **all** functions in the current input and return their key semantic features — what each function does, not how it’s implemented.

### Key Goals
- Complete analysis: Provide semantic feature extraction for **every** function in the given input. Do not skip any function.
- Batch perspective: Analyze all functions in the chunk together, considering their roles within the overall system.
- High-level behavior: Focus on the purpose and role of each function, not on low-level implementation details.
- If multiple definitions share the same method name (e.g., property getter and setter for the same attribute), you may output that method name only once and merge their semantic features; you do not need to distinguish decorator variants.

## Feature Extraction Principles
Follow these principles when analyzing functions:
1. Focus on the purpose and behavior of the function — what role it serves in the system.
2. Do NOT describe implementation details, variable names, or internal logic such as loops, conditionals, or data structures.
3. If a function performs multiple responsibilities, break them down into separate features.
4. Use your understanding of each function’s name, signature, and code to infer its intent.
5. Only analyze functions included in the current input — do not guess or invent other functions.
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

## Output Format
You must respond with the following structure:
A `<solution>` block — a JSON object mapping each function name to a list of its semantic features
If a function does not implement any meaningful features (e.g., it's a stub), still include it with an empty list.
### Output Template:
<solution>
{{
  "func_name_1": ["feature one", "feature two"],
  "func_name_2": [],
  ...
}}
</solution>

### Example:
<solution>
{{
  "download_image": ["download image from URL", "save image to local disk"],
  "resize_image": ["resize image to target dimensions", "update image metadata"],
  "noop": []
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