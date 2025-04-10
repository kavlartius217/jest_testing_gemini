#API KEYS
import os
os.environ['EXA_API_KEY']
os.environ['OPENAI_API_KEY']





#TOOLS
from crewai_tools import DirectoryReadTool
from crewai_tools import FileReadTool
from crewai_tools import EXASearchTool
from crewai_tools import CodeDocsSearchTool
from exa_py import Exa
from crewai.tools import tool





#TOOLS SET 1
directory_read_tool=DirectoryReadTool('gcc-national-registry-dashboard-Dev_Branch/server/src/controller')
file_read_tool=FileReadTool()
docs_search_tool=CodeDocsSearchTool('https://archive.jestjs.io/docs/en/22.x/getting-started.html')





#TOOLS SET 2
#Exa Search Tool
exa_api_key = os.getenv("EXA_API_KEY")

@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""

    exa = Exa(exa_api_key)

    response = exa.search_and_contents(
        question,
        type="neural",
        use_autoprompt=True,
        num_results=3,
        highlights=True
    )

    parsedResult = ''.join([
    f'<Title id="{idx}">{eachResult.title}</Title>'
    f'<URL id="{idx}">{eachResult.url}</URL>'
    f'<Highlight id="{idx}">{"".join(eachResult.highlights)}</Highlight>'
    for idx, eachResult in enumerate(response.results)
])

    return parsedResult

exa_tool=search_and_get_contents_tool





#LLMS
#Testing all LLM combinations
from crewai import LLM
llm_gemma_1=LLM(model="groq/gemma2-9b-it",temperature=0) #chaitany_api_key
llm_gemma_2=LLM(model="groq/gemma2-9b-it",temperature=0) #hardik_api_key
llm_openai_4o=LLM(model="openai/gpt-4o-mini",temperature=0)
llm_openai_o3=LLM(model="openai/o1-mini-2024-09-12", temperature=0)





#Agents and Tasks
from crewai import Agent, Task, Crew, Process

#Agent 1
#directory_listing_agent= This agent will retrieve the path of all the files in the directories.

directory_listing_agent=Agent(
    role="Directory Listing Agent",
    goal="Identify and return all file paths within the specified directory, including subdirectories.",
    backstory=(
            "This agent scans the entire directory and retrieves a structured list of file paths, "
            "which will be used for further content extraction and analysis."
        ),
    tools=[directory_read_tool],
    llm=llm_gemma_1,
    memory=True,
    verbose=True
    )

directory_listing_task=Task(
    description="Scan the specified directory and list all file paths, including those in subdirectories.",
    expected_output="A structured list of absolute file paths, saved in 'file_paths.md' with one file path per line.",
    agent= directory_listing_agent,
    output_file="file_paths_2.md"
)





#Agent 2
#file_extraction_agent= This agent will extract all the code

file_extraction_agent=Agent(
    role="Code File Consolidator",
    goal="Extract complete source code from multiple files and combine them into a single consolidated output while preserving file boundaries and organizational structure.",
    backstory="A specialized assistant that efficiently processes multiple source code files, extracts their full content without modification, and arranges them in a single organized document for comprehensive analysis.",
    tools=[file_read_tool],
    llm=llm_openai_4o,
    function_calling_llm=llm_gemma_2,
    memory=True,
    verbose=True,
)

file_extraction_task=Task(
    description="Extract all JavaScript code from the file paths provided by the directory_listing_task and save it as a single consolidated JavaScript file.",
    expected_output="""
    A JavaScript file (`extracted_code.js`) that contains all extracted JavaScript code from the file paths listed in 'file_paths.md', with the following requirements:

    1. IMPORTANT: Not a single line of code should be missed. Ensure 100% extraction accuracy from all listed files.
    2. Read and process each JavaScript file path provided in the output of directory_listing_task
    3. Each extracted file must begin with a clear file header showing the original filename
    4. All code, comments, and whitespace must be preserved exactly as found in the original files
    5. Files must be separated by clear visual delimiters for easy navigation
    6. The entire content of each file must be included with no truncation
    7. No explanations, summaries, or analysis of the code should be included
    8. The code must not be modified, optimized, or reformatted in any way
    9. If a file cannot be read, a comment documenting the specific error must be included
    10. Return only the consolidated file content - no additional commentary
    11. The entire content of each file should be accurately extracted
    """,
    agent=file_extraction_agent,
    context=[directory_listing_task],
    output_file="extracted_code.js"
)





#Agent 3- This agent will divide the code into units

code_segmentation_agent = Agent(
    role="JavaScript Code Unit Segmentation Agent",
    goal="Break down the extracted JavaScript code into individual units (functions, classes, modules, and components) for structured analysis.",
    backstory="This agent specializes in parsing JavaScript code structure to identify discrete functional units such as functions, classes, ES6 modules, React components, and other JavaScript-specific patterns, enabling effective dependency mapping and test generation.",
    llm=llm_openai_4o,
    function_calling_llm=llm_gemma_2,
    memory=True,
    verbose=True
)

code_segmentation_task = Task(
    description="Segment the extracted JavaScript code into logical units such as functions, classes, components, and modules. Each segment should be properly formatted and structured for easy analysis. Ensure EVERY code unit is identified and documented with no omissions.",
    expected_output="""
    A structured JavaScript file (`segments.js`) that contains each JavaScript code unit as an independent, clearly separated segment, with the following requirements:

    1. IMPORTANT: Every segment must include actual code from the original file. Do not generate placeholders, metadata-only entries, or empty sections. The code must be directly extracted, and no segment should be documented without its real source code block.

    2. Identify and separate ALL JavaScript-specific code units, including but not limited to:
      - ES6 classes and their methods
      - Function declarations and function expressions
      - Arrow functions that define components or utilities
      - ES6 modules (import/export statements)
      - React components (functional and class-based)
      - Object literals that define structured interfaces
      - Prototype-based inheritance patterns
      - IIFE (Immediately Invoked Function Expressions)
      - Event handlers and callbacks
      - Async functions and Promises
      - Anonymous functions and nested functions
      - Higher-order functions
      
    3. For each code unit, include:
      - Original file name
      - Line numbers (start-end) from the original file
      - Complete implementation with preserved formatting
      - Type of unit (function, class, component, etc.)
      - Dependencies and imports used by the unit

    4. Preserve JavaScript-specific patterns like closures and lexical scoping

    5. Track both CommonJS and ES6 module syntax appropriately

    6. Include module exports to document public interfaces

    7. Properly document framework-specific components (React, Vue, Angular, etc.)

    8. Organize segments hierarchically when appropriate (e.g., methods under their classes)

    9. Ensure COMPLETE coverage - every single line of code must be accounted for in at least one segment

    10. Do not omit any code segment

    11. For overlapping segments (e.g., a function within a class), include the code in both relevant sections with appropriate cross-references
    """,
    agent=code_segmentation_agent,
    context=[file_extraction_task],
    output_file="segments.js"
)






#Agent 4- This agent will generate the jest test cases

test_case_generator=Agent(
        role="Automated JavaScript Test Case Generator",
        goal="Analyze the segmented JavaScript code and generate comprehensive Jest test cases for each function, class, component, and module to ensure correctness, edge-case handling, and robustness.",
        backstory="An AI-powered JavaScript testing expert designed to deeply understand modern JS code structure, dependencies, and functionality. It generates precise and efficient Jest test cases covering various scenarios, including positive, negative, and edge cases, with special focus on asynchronous patterns and React components.",
        llm=llm_openai_o3,
        tools=[exa_tool, docs_search_tool],
        function_calling_llm=llm_gemma_2,
        memory=True,
        verbose=True,
)

test_case_generation_task = Task(
    description="Generate comprehensive Jest test cases for all JavaScript functions, classes, components, and modules based on the segmented code structure.",
    expected_output="""
    IMPORTANT: Provide complete Jest test code for EVERY identified JavaScript unit without any omissions. No code unit should be left without corresponding test cases.

    A structured test case file (`test_cases.test.js`), containing Jest unit tests and integration tests for all identified JavaScript code units, with the following requirements:

    1. Test Coverage Requirement:
     - Ensure that the generated tests provide **at least 90% code coverage** across all JavaScript units.
     - This includes **statements, branches, functions, and lines** as measured by `jest --coverage`.
     - Include:
       - Valid input (happy path) scenarios
       - Edge cases such as `null`, `undefined`, empty arrays/objects, boundary values
       - Failure scenarios and exception handling
       - Asynchronous logic including Promises and `async/await` functions

    2. Test coverage should include:
       - Happy path tests with valid inputs and expected outputs
       - Edge cases (empty arrays, null values, boundary conditions)
       - Error handling scenarios and expected exceptions
       - Asynchronous behavior for Promises, async/await functions

    3. When dependencies are not available:
       - Create Jest mocks (`jest.mock()`) for external dependencies
       - Use mock functions (`jest.fn()`) with appropriate return values
       - Mock API calls and external services
       - Provide mock implementations for complex dependencies

    4. For React components:
       - Use React Testing Library or Enzyme patterns
       - Test both rendering and component behavior
       - Mock context providers and Redux stores as needed

    5. Test organization:
       - Group related tests in `describe` blocks
       - Use `beforeEach` / `afterEach` for common setup and teardown
       - Separate unit tests from integration tests

    6. Each test file should:
       - Reference the original file and function names
       - Include necessary import statements
       - Follow Jest naming conventions (`*.test.js` or `*.spec.js`)

    7. Include appropriate test data fixtures or factory functions

    8. For object-oriented code, test both class instantiation and methods

    9. Use appropriate Jest matchers (`toEqual`, `toBe`, `toHaveBeenCalled`, etc.)

    10. Actively use the provided tools:
       - Use `exa_tool` for internet searches to find examples, patterns, and best practices for testing specific JavaScript patterns or frameworks
       - Use `docs_search_tool` to access and reference the Jest documentation for correct syntax, available matchers, and recommended approaches
       - When encountering unfamiliar patterns, use the tools to research appropriate testing strategies
       - Reference specific Jest documentation when implementing complex testing patterns
       - Search for examples of testing similar components or functions when needed
       - Use tools to verify correct Jest usage for mocking, spying, and asynchronous testing
    """,
    agent=test_case_generator,
    context=[code_segmentation_task],
    output_file="test_cases.test.js"
)





#Testing the Agents

testing_crew=Crew(
    agents=[directory_listing_agent,file_extraction_agent,code_segmentation_agent, test_case_generator],
    tasks=[directory_listing_task,file_extraction_task,code_segmentation_task, test_case_generation_task],
    process=Process.sequential
)
testing_crew.kickoff()
