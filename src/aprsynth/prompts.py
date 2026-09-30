from __future__ import annotations

PROMPT_VERSION = "tagged-generation-v1"
EVALUATION_PROMPT_VERSION = "patch-evaluation-v1"

GENERATION_PROMPT = """You are an expert software developer and secure coding specialist. Your task is to generate pairs of code snippets that demonstrate common programming bugs and their corresponding fixes. Each pair should include:

- **Error Description:** A brief explanation of the specific bug or vulnerability in the code.
- **Buggy Code:** A detailed code snippet that contains the bug. Ensure the code realistically reflects common development mistakes or vulnerabilities.
- **Fixed Code:** The corrected version of the buggy code with the identified issue resolved.

## Requirements:

- **Bug or Vulnerability Type:** The buggy code should contain a realistic programming bug or vulnerability of the following type: {bug}.
- **Programming Languages:** The code should be written in {language}.
- **Realism and Complexity:** The code should be at least 50 lines long and resemble real-world scenarios, including realistic functions, classes, or modules that might appear in software projects.
- **Output Format:** Ensure your response strictly follows this format:

<error_description>
Brief description of the bug or vulnerability
</error_description>
<buggy_code>
Code snippet containing the bug
</buggy_code>
<fixed_code>
Corrected code snippet
</fixed_code>

## Example Output (in java)

<error_description>
Race condition in a multi-threaded application leading to inconsistent data being written to the database.
</error_description>
<buggy_code>
class DataProcessor {{\n    private int count = 0;\n    public void increment() {{\n        for (int i = 0; i < 1000; i++) {{\n            Thread t = new Thread(() -> {{\n                count++;\n            }});\n            t.start();\n        }}\n    }}\n    public int getCount() {{\n        return count;\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) throws InterruptedException {{\n        DataProcessor dp = new DataProcessor();\n        dp.increment();\n        Thread.sleep(1000);\n        System.out.println("Count: " + dp.getCount());\n    }}\n}}
</buggy_code>
<fixed_code>
class DataProcessor {{\n    private int count = 0;\n    public synchronized void increment() {{\n        for (int i = 0; i < 1000; i++) {{\n            Thread t = new Thread(() -> {{\n                count++;\n            }});\n            t.start();\n        }}\n    }}\n    public synchronized int getCount() {{\n        return count;\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) throws InterruptedException {{\n        DataProcessor dp = new DataProcessor();\n        dp.increment();\n        Thread.sleep(1000);\n        System.out.println("Count: " + dp.getCount());\n    }}\n}}
</fixed_code>

Do not include backticks to represent the source code. Do not generate any additional text or information. Only the specified output format is accepted. A single pair of code snippets is required.
"""

PATCH_EVALUATION_PROMPT = """You are an expert software developer and secure coding specialist. Your task is to evaluate pairs of code snippets that demonstrate common programming bugs and their corresponding fixes. Each pair includes:

- **Error Description:** A brief explanation of the specific bug or vulnerability in the code.
- **Buggy Code:** A detailed code snippet that contains the bug.
- **Patch:** The generated patch that fixes the identified issue in the buggy code.

Your evaluation should include:

- **Correctness:** Whether the fixed code correctly resolves the issue described in the error description.
- **Code Quality:** Whether the fixed code maintains or improves the overall code quality, including readability, maintainability, and adherence to best practices.
- **Security:** Whether the fixed code addresses any security vulnerabilities present in the buggy code.
- **Performance:** Whether the fixed code maintains or improves the performance of the original code.
- **Completeness:** Whether the fixed code fully addresses all aspects of the bug or vulnerability described.

## Requirements:

- **Rating:** Assign a score from 0 to 10 for each criterion.
- **Output Format:** Ensure your response strictly follows this JSON format:

{{
  "correctness": x,
  "code_quality": x,
  "security": x,
  "performance": x,
  "completeness": x
}}

Provide only a JSON object with ratings, no additional text or information.

This is the sample you need to evaluate:

<error_description>
{error_description}
</error_description>
<buggy_code>
{buggy_code}
</buggy_code>
<patch>
{patch}
</patch>
"""


def render_generation_messages(
    language: str,
    bug: str,
) -> list[dict[str, str]]:
    prompt = render_generation_prompt(language, bug)
    return [{"role": "user", "content": prompt}]


def render_generation_prompt(
    language: str,
    bug: str,
) -> str:
    return GENERATION_PROMPT.format(language=language, bug=bug)


def render_evaluation_messages(
    *,
    buggy_code: str,
    patch: str,
    error_description: str,
) -> list[dict[str, str]]:
    prompt = render_evaluation_prompt(
        buggy_code=buggy_code,
        patch=patch,
        error_description=error_description,
    )
    return [{"role": "user", "content": prompt}]


def render_evaluation_prompt(
    *,
    buggy_code: str,
    patch: str,
    error_description: str,
) -> str:
    return PATCH_EVALUATION_PROMPT.format(
        buggy_code=buggy_code,
        patch=patch,
        error_description=error_description,
    )
