"""
Code Scan agent using OpenAI Agents SDK.
"""

from agents import Agent, ComputerTool, FileSearchTool, FunctionTool, WebSearchTool
from pydantic import BaseModel, Field

from src.tools.github_function_tools import (
    create_issue,
    get_repository_file_content,
    get_repository_info,
    get_repository_stats,
    list_repository_files,
    search_code,
)


class CodeIssue(BaseModel):
    """Code issue found during scanning."""

    file: str = Field(description="File path where the issue was found")
    line: int | None = Field(description="Line number where the issue was found", default=None)
    severity: str = Field(description="Severity of the issue (critical, high, medium, low)")
    description: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Suggestion for fixing the issue")


class CodeScanResponse(BaseModel):
    """Response model for Code Scan agent."""

    overview: str = Field(description="Overview of the code scan results")
    errors: list[CodeIssue] = Field(description="List of errors and bugs found")
    gaps: list[CodeIssue] = Field(
        description="List of gaps found (missing functionality, tests, documentation)"
    )
    misconfigurations: list[CodeIssue] = Field(
        description="List of misconfigurations found (wrong settings, environment issues)"
    )
    missing_data: list[CodeIssue] = Field(
        description=(
            "List of missing data issues (missing required fields, incomplete configurations)"
        )
    )
    issues: list[CodeIssue] = Field(description="List of other issues found")
    good_practices: list[str] = Field(description="Good practices observed in the code")
    recommendations: list[str] = Field(description="Overall recommendations for code improvements")


def create_code_scan_agent(model: str = "gpt-4o-mini", custom_prompt: str | None = None) -> Agent:
    """Create a Code Scan agent for analyzing repository code.

    Args:
        model: Model name to use
        custom_prompt: Custom prompt override

    Returns:
        Configured Agent instance
    """

    instructions = """
    You are a code scan agent that analyzes code in GitHub repositories.
    
    Your task is to scan repository files and produce a detailed report that includes:
    1. An overview of the code scan results
    2. Errors: bugs, runtime errors, and logic defects
    3. Gaps: missing functionality, missing tests, missing documentation
    4. Misconfigurations: wrong settings, misconfigured environments, incorrect parameters
    5. Missing data: missing required fields, incomplete configurations, absent required files
    6. Other issues: remaining code quality problems, anti-patterns, performance concerns
    7. Good practices observed in the code
    8. Overall recommendations for improvement
    
    Look for:
    - Security vulnerabilities
    - Performance issues
    - Code quality problems
    - Potential bugs
    - Anti-patterns
    
    Be thorough and specific in your analysis, categorizing each finding accurately.
    Focus on the most important issues first within each category.

    IMPORTANT: If you find any issues, create an issue in the repository using the create_issue tool
    """

    if custom_prompt:
        instructions = custom_prompt

    tools: list[FunctionTool | FileSearchTool | WebSearchTool | ComputerTool] = [
        get_repository_info,
        get_repository_file_content,
        get_repository_stats,
        create_issue,
        search_code,
        list_repository_files,
    ]

    return Agent(
        name="Code Scan Agent",
        instructions=instructions,
        tools=tools,
        model=model,
        output_type=CodeScanResponse,
    )
