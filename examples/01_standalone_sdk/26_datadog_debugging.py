#!/usr/bin/env python3
"""
Datadog Debugging Example

This example demonstrates how to use the OpenHands agent to debug errors
logged in Datadog.
The agent will:
1. Query Datadog logs to understand the error using curl commands
2. Clone relevant GitHub repositories using git commands
3. Analyze the codebase to identify potential causes
4. Attempt to reproduce the error
5. Optionally create a draft PR with a fix

Usage:
    python 26_datadog_debugging.py --query "status:error service:deploy" \\
        --repos "All-Hands-AI/OpenHands,All-Hands-AI/deploy"

Environment Variables Required:
    - DD_API_KEY: Your Datadog API key
    - DD_APP_KEY: Your Datadog application key
    - DD_SITE: (optional) Datadog site (e.g., datadoghq.com, datadoghq.eu)
    - GITHUB_TOKEN: Your GitHub personal access token
    - LLM_API_KEY: API key for the LLM service
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool


logger = get_logger(__name__)


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "DD_API_KEY",
        "DD_APP_KEY",
        "GITHUB_TOKEN",
        "LLM_API_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_key_here")
        return False

    return True


def fetch_datadog_errors(query: str, working_dir: Path, limit: int = 5) -> Path:
    """
    Fetch error examples from Datadog Error Tracking and save to a JSON file.

    Args:
        query: Datadog query string (can be error tracking query or issue ID)
        working_dir: Directory to save the error examples
        limit: Maximum number of error examples to fetch (default: 5)

    Returns:
        Path to the JSON file containing error examples
    """
    dd_api_key = os.getenv("DD_API_KEY")
    dd_app_key = os.getenv("DD_APP_KEY")
    dd_site = os.getenv("DD_SITE", "datadoghq.com")

    # Use Error Tracking Search API
    api_url = f"https://api.{dd_site}/api/v2/error-tracking/issues/search"

    # Calculate timestamps (30 days back)
    now = int(datetime.now().timestamp() * 1000)
    thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

    # Build the request body for Error Tracking API
    request_body = {
        "data": {
            "attributes": {
                "query": query,
                "from": thirty_days_ago,
                "to": now,
                "track": "logs",  # Track errors from logs
            },
            "type": "search_request",
        }
    }

    print(f"📡 Fetching up to {limit} error tracking issues from Datadog...")
    print(f"   Query: {query}")
    print(f"   API: {api_url}")

    # Add include parameter to get full issue details
    api_url_with_include = f"{api_url}?include=issue"

    # Use curl to fetch data
    curl_cmd = [
        "curl",
        "-X",
        "POST",
        api_url_with_include,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"DD-API-KEY: {dd_api_key}",
        "-H",
        f"DD-APPLICATION-KEY: {dd_app_key}",
        "-d",
        json.dumps(request_body),
        "-s",  # Silent mode
    ]

    try:
        result = subprocess.run(
            curl_cmd, capture_output=True, text=True, check=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        print("❌ Error: Request to Datadog API timed out")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching from Datadog API: {e}")
        print(f"   stderr: {e.stderr}")
        sys.exit(1)

    try:
        response_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Datadog API response: {e}")
        print(f"   Response: {result.stdout[:500]}")
        sys.exit(1)

    # Check for API errors
    if "errors" in response_data:
        print(f"❌ Datadog API error: {response_data['errors']}")
        sys.exit(1)

    # Extract and format error tracking issues
    error_examples = []
    search_results = response_data.get("data", [])
    included_details = {item["id"]: item for item in response_data.get("included", [])}

    if search_results:
        for idx, search_result in enumerate(search_results[:limit], 1):
            issue_id = search_result.get("id")
            search_attrs = search_result.get("attributes", {})

            # Get detailed issue info from included section
            issue_details = included_details.get(issue_id, {})
            issue_attrs = issue_details.get("attributes", {})

            error_example = {
                "example_number": idx,
                "issue_id": issue_id,
                "total_count": search_attrs.get("total_count"),
                "impacted_users": search_attrs.get("impacted_users"),
                "impacted_sessions": search_attrs.get("impacted_sessions"),
                "service": issue_attrs.get("service"),
                "error_type": issue_attrs.get("error_type"),
                "error_message": issue_attrs.get("error_message", ""),
                "file_path": issue_attrs.get("file_path"),
                "function_name": issue_attrs.get("function_name"),
                "first_seen": issue_attrs.get("first_seen"),
                "last_seen": issue_attrs.get("last_seen"),
                "state": issue_attrs.get("state"),
                "platform": issue_attrs.get("platform"),
                "languages": issue_attrs.get("languages", []),
            }
            error_examples.append(error_example)

    # Save to file
    errors_file = working_dir / "datadog_errors.json"
    with open(errors_file, "w") as f:
        json.dump(
            {
                "query": query,
                "fetch_time": "now",
                "total_examples": len(error_examples),
                "examples": error_examples,
            },
            f,
            indent=2,
        )

    print(f"✅ Fetched {len(error_examples)} error examples")
    print(f"📄 Saved to: {errors_file}")
    return errors_file


def create_unique_identifier(query: str, errors_data: dict) -> str:
    """
    Create a unique identifier for the error based on query or issue ID.

    Args:
        query: The Datadog query string
        errors_data: The parsed error data from datadog_errors.json

    Returns:
        Unique identifier string
    """
    # Check if we have a specific issue ID
    examples = errors_data.get("examples", [])
    if examples and examples[0].get("issue_id"):
        issue_id = examples[0]["issue_id"]
        return f"error-id: {issue_id}"
    else:
        # Use query as identifier
        return f"query: {query}"


def search_existing_issue(
    issue_repo: str, identifier: str, github_token: str
) -> int | None:
    """
    Search for existing GitHub issues containing the identifier.

    Args:
        issue_repo: Repository in format 'owner/repo'
        identifier: Unique identifier to search for
        github_token: GitHub API token

    Returns:
        Issue number if found, None otherwise
    """
    print(f"🔍 Searching for existing issue with identifier: {identifier}")

    # Search issues in the repository
    search_query = f'repo:{issue_repo} is:issue "{identifier}"'
    encoded_query = quote(search_query)
    url = "https://api.github.com/search/issues"

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-H",
            f"Authorization: Bearer {github_token}",
            "-H",
            "Accept: application/vnd.github+json",
            f"{url}?q={encoded_query}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    try:
        data = json.loads(result.stdout)
        items = data.get("items", [])
        if items:
            # Sort by created_at to get the oldest issue (first created)
            items_sorted = sorted(items, key=lambda x: x["created_at"])
            issue_number = items_sorted[0]["number"]
            print(f"✅ Found existing issue #{issue_number} (oldest of {len(items)})")
            return issue_number
        else:
            print("❌ No existing issue found")
            return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Error searching for issues: {e}")
        return None


def create_github_issue(
    issue_repo: str,
    title: str,
    body: str,
    github_token: str,
) -> int:
    """
    Create a new GitHub issue.

    Args:
        issue_repo: Repository in format 'owner/repo'
        title: Issue title
        body: Issue body content
        github_token: GitHub API token

    Returns:
        Created issue number
    """
    print(f"📝 Creating new issue: {title}")

    url = f"https://api.github.com/repos/{issue_repo}/issues"

    payload = {"title": title, "body": body}

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            url,
            "-H",
            f"Authorization: Bearer {github_token}",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    try:
        data = json.loads(result.stdout)
        issue_number = data["number"]
        issue_url = data["html_url"]
        print(f"✅ Created issue #{issue_number}: {issue_url}")
        return issue_number
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error creating issue: {e}")
        print(f"Response: {result.stdout[:500]}")
        sys.exit(1)


def format_issue_body(
    errors_data: dict,
    identifier: str,
    parent_issue_url: str | None,
) -> str:
    """
    Format the GitHub issue body with error details.

    Args:
        errors_data: The parsed error data
        identifier: Unique identifier
        parent_issue_url: Optional parent issue URL

    Returns:
        Formatted issue body
    """
    examples = errors_data.get("examples", [])
    query = errors_data.get("query", "")

    body_parts = []

    # Add parent issue reference if provided
    if parent_issue_url:
        body_parts.append(f"**Parent Issue:** {parent_issue_url}\n")

    # Add identifier for searchability
    body_parts.append(f"**Identifier:** `{identifier}`\n")

    # Add query info
    body_parts.append(f"**Query:** `{query}`\n")

    # Add error summary
    if examples:
        first_example = examples[0]
        body_parts.append("## Error Summary\n")

        if first_example.get("issue_id"):
            body_parts.append(f"- **Issue ID:** `{first_example['issue_id']}`")
        if first_example.get("total_count"):
            body_parts.append(
                f"- **Total Occurrences:** {first_example['total_count']}"
            )
        if first_example.get("error_type"):
            body_parts.append(f"- **Error Type:** `{first_example['error_type']}`")
        if first_example.get("service"):
            body_parts.append(f"- **Service:** `{first_example['service']}`")
        if first_example.get("file_path"):
            body_parts.append(f"- **File:** `{first_example['file_path']}`")
        if first_example.get("function_name"):
            body_parts.append(f"- **Function:** `{first_example['function_name']}`")
        if first_example.get("state"):
            body_parts.append(f"- **State:** {first_example['state']}")

        body_parts.append("")

        # Add error message if available
        if first_example.get("error_message"):
            body_parts.append("## Error Message\n")
            body_parts.append("```")
            body_parts.append(first_example["error_message"])
            body_parts.append("```\n")

    # Add note about full data
    body_parts.append("## Full Error Data\n")
    body_parts.append(
        "The complete error tracking data has been saved and will be analyzed "
        "by the debugging agent.\n"
    )

    # Add JSON data as collapsible section
    body_parts.append("<details>")
    body_parts.append("<summary>View Full Error Data (JSON)</summary>\n")
    body_parts.append("```json")
    body_parts.append(json.dumps(errors_data, indent=2))
    body_parts.append("```")
    body_parts.append("</details>\n")

    body_parts.append("---")
    body_parts.append(
        "*This issue is being tracked by an automated debugging agent. "
        "Analysis findings will be posted as comments below.*"
    )

    return "\n".join(body_parts)


def setup_github_issue(
    query: str,
    errors_file: Path,
    issue_repo: str,
    issue_prefix: str,
    issue_parent: str | None,
) -> tuple[int, str]:
    """
    Create or find GitHub issue for tracking debugging progress.

    Args:
        query: The Datadog query
        errors_file: Path to the errors JSON file
        issue_repo: GitHub repository for issues
        issue_prefix: Prefix for issue titles
        issue_parent: Optional parent issue URL

    Returns:
        Tuple of (issue_number, issue_url)
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    # Load error data
    with open(errors_file) as f:
        errors_data = json.load(f)

    # Create unique identifier
    identifier = create_unique_identifier(query, errors_data)

    # Search for existing issue
    issue_number = search_existing_issue(issue_repo, identifier, github_token)

    if issue_number:
        # Return existing issue
        issue_url = f"https://github.com/{issue_repo}/issues/{issue_number}"
        return issue_number, issue_url

    # Create new issue
    # Determine title from error data
    examples = errors_data.get("examples", [])
    if examples and examples[0].get("error_type"):
        error_name = examples[0]["error_type"]
    else:
        # Use query as fallback
        error_name = query[:50]  # Limit length

    title = f"{issue_prefix}{error_name}"

    # Format issue body
    body = format_issue_body(errors_data, identifier, issue_parent)

    # Create issue
    issue_number = create_github_issue(issue_repo, title, body, github_token)
    issue_url = f"https://github.com/{issue_repo}/issues/{issue_number}"

    return issue_number, issue_url


def create_debugging_prompt(
    query: str, repos: list[str], errors_file: Path, issue_url: str
) -> str:
    """Create the debugging prompt for the agent."""
    repos_list = "\n".join(f"- {repo}" for repo in repos)
    dd_site = os.getenv("DD_SITE", "datadoghq.com")
    error_tracking_url = f"https://api.{dd_site}/api/v2/error-tracking/issues/search"
    logs_url = f"https://api.{dd_site}/api/v2/logs/events/search"

    prompt = (
        "Your task is to debug an error from Datadog Error Tracking to find "
        "out why it is happening.\n\n"
        "## GitHub Issue for Tracking\n\n"
        f"A GitHub issue has been created to track this investigation: {issue_url}\n\n"
        "**IMPORTANT**: As you make progress in your investigation, post your "
        "findings as comments on this GitHub issue using curl commands:\n\n"
        "```bash\n"
        "curl -X POST \\\n"
        f"  'https://api.github.com/repos/{{REPO}}/issues/{{NUMBER}}/comments' \\\n"
        "  -H 'Authorization: Bearer $GITHUB_TOKEN' \\\n"
        "  -H 'Accept: application/vnd.github+json' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        '  -d \'{"body": "Your finding here"}\'\n'
        "```\n\n"
        "Post updates when you:\n"
        "- Complete analyzing the error data\n"
        "- Find relevant code in the repositories\n"
        "- Identify the root cause\n"
        "- Attempt a reproduction\n"
        "- Make any significant discovery\n\n"
        "## Error Tracking Issues\n\n"
        f"I have already fetched error tracking issues and saved them to: "
        f"`{errors_file}`\n\n"
        "This JSON file contains:\n"
        "- `query`: The Datadog query used to fetch these errors\n"
        "- `total_examples`: Number of error tracking issues in the file\n"
        "- `examples`: Array of error tracking issues, where each has:\n"
        "  - `issue_id`: Unique identifier for the aggregated error issue\n"
        "  - `total_count`: Total number of error occurrences\n"
        "  - `impacted_users`: Number of users affected\n"
        "  - `service`: Service name where errors occurred\n"
        "  - `error_type`: Type of error (e.g., exception class)\n"
        "  - `error_message`: Error message text\n"
        "  - `file_path`: Source file where error occurred\n"
        "  - `function_name`: Function where error occurred\n"
        "  - `first_seen`: Timestamp when first seen (milliseconds)\n"
        "  - `last_seen`: Timestamp when last seen (milliseconds)\n"
        "  - `state`: Issue state (OPEN, ACKNOWLEDGED, RESOLVED, etc.)\n\n"
        "**First, read the GitHub issue** to see the error summary, then read "
        f"`{errors_file}` to understand the error patterns. Error Tracking "
        "aggregates similar errors together, so each issue may represent many "
        "occurrences.\n\n"
        "## Additional Context\n\n"
        f"The original Datadog query was: `{query}`\n\n"
        "If you need more details, you can use Datadog APIs via curl commands "
        "with $DD_API_KEY and $DD_APP_KEY environment variables.\n\n"
        "To search for more error tracking issues:\n"
        "```bash\n"
        f"curl -X POST '{error_tracking_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H 'DD-API-KEY: $DD_API_KEY' \\\n"
        "  -H 'DD-APPLICATION-KEY: $DD_APP_KEY' \\\n"
        '  -d \'{"data": {"attributes": {"query": "service:YOUR_SERVICE", '
        '"from": <timestamp_ms>, "to": <timestamp_ms>, "track": "logs"}, '
        '"type": "search_request"}}\'\n'
        "```\n\n"
        "To query individual log entries, use the Logs API:\n"
        "```bash\n"
        f"curl -X POST '{logs_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H 'DD-API-KEY: $DD_API_KEY' \\\n"
        "  -H 'DD-APPLICATION-KEY: $DD_APP_KEY' \\\n"
        "  -d '{\n"
        '    "filter": {\n'
        '      "query": "YOUR_QUERY_HERE",\n'
        '      "from": "now-1d",\n'
        '      "to": "now"\n'
        "    },\n"
        '    "sort": "timestamp",\n'
        '    "page": {\n'
        '      "limit": 10\n'
        "    }\n"
        "  }'\n"
        "```\n\n"
        "The Datadog query syntax supports:\n"
        "- status:error - Find error logs\n"
        "- service:my-service - Filter by service\n"
        '- "exact phrase" - Search for exact text\n'
        "- -(status:info OR status:debug) - Exclude certain statuses\n"
        "- Use time ranges to focus on recent issues\n\n"
        "The error class that I would like you to debug is characterized "
        f"by this datadog query:\n{query}\n\n"
        "To clone the GitHub repositories, use git with authentication:\n"
        "```bash\n"
        "git clone https://$GITHUB_TOKEN@github.com/OWNER/REPO.git\n"
        "```\n\n"
        "The github repos that you should clone (using GITHUB_TOKEN) are "
        f"the following:\n{repos_list}\n\n"
        "## Debugging Steps\n\n"
        "Follow these steps systematically:\n\n"
        "1. **Read the error file** - Start by reading "
        f"`{errors_file}` to understand the error patterns. "
        "Examine all examples to identify:\n"
        "   - Common error messages\n"
        "   - Stack traces and their origins\n"
        "   - Affected services\n"
        "   - Timestamps (when did errors start?)\n\n"
        "2. **Analyze the timeline** - Check when the error class started "
        "occurring/becoming frequent. Look at the timestamps in the error "
        "examples. This helps identify what code changes or deployment may "
        "have caused the issue. Code changed during the release cycle before "
        "the error occurred will be most suspicious.\n\n"
        "3. **Clone repositories** - Clone the relevant repositories using:\n"
        "   ```bash\n"
        "   git clone https://$GITHUB_TOKEN@github.com/OWNER/REPO.git\n"
        "   ```\n\n"
        "4. **Investigate the codebase** - Carefully read the code related "
        "to the error. Look at:\n"
        "   - Files mentioned in stack traces\n"
        "   - Recent commits (use git log)\n"
        "   - Related code paths\n\n"
        "5. **Develop hypotheses** - Think of 5 possible root causes and "
        "write sample code to test each hypothesis. Try to reproduce the "
        "error.\n\n"
        "6. **Create fix or summarize** - Based on your findings:\n"
        "   - If reproducible: Create a fix and optionally open a draft PR\n"
        "   - If not reproducible: Summarize your investigation, findings, "
        "and recommendations\n\n"
        "**Important**: Use the task_tracker tool to organize your work and "
        "keep track of your progress through these steps."
    )

    return prompt


def main():
    """Main function to run the Datadog debugging example."""
    parser = argparse.ArgumentParser(
        description="Debug errors from Datadog logs using OpenHands agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Datadog query to search for error logs "
        "(e.g., 'status:error service:deploy')",
    )
    parser.add_argument(
        "--repos",
        required=True,
        help="Comma-separated list of GitHub repositories to analyze "
        "(e.g., 'All-Hands-AI/OpenHands,All-Hands-AI/deploy')",
    )
    parser.add_argument(
        "--working-dir",
        default="./datadog_debug_workspace",
        help="Working directory for cloning repos and analysis "
        "(default: ./datadog_debug_workspace)",
    )
    parser.add_argument(
        "--issue-repo",
        required=True,
        help="GitHub repository for creating/updating issues "
        "(e.g., 'All-Hands-AI/infra')",
    )
    parser.add_argument(
        "--issue-parent",
        help="Parent issue URL to reference (e.g., "
        "'https://github.com/All-Hands-AI/infra/issues/672')",
    )
    parser.add_argument(
        "--issue-prefix",
        default="",
        help="Prefix to add to issue titles (e.g., 'DataDog Error Bash: ')",
    )

    args = parser.parse_args()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Parse repositories
    repos = [repo.strip() for repo in args.repos.split(",")]

    # Create working directory
    working_dir = Path(args.working_dir).resolve()
    working_dir.mkdir(exist_ok=True)

    print("🔍 Starting Datadog debugging session")
    print(f"📊 Query: {args.query}")
    print(f"📁 Repositories: {', '.join(repos)}")
    print(f"🌍 Datadog site: {os.getenv('DD_SITE', 'datadoghq.com')}")
    print(f"💼 Working directory: {working_dir}")
    print()

    # Fetch error examples from Datadog
    errors_file = fetch_datadog_errors(args.query, working_dir)
    print()

    # Setup GitHub issue for tracking
    print("📋 Setting up GitHub issue for tracking...")
    issue_number, issue_url = setup_github_issue(
        args.query,
        errors_file,
        args.issue_repo,
        args.issue_prefix,
        args.issue_parent,
    )
    print(f"📌 Tracking issue: {issue_url}")
    print()

    # Configure LLM
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("❌ LLM_API_KEY environment variable is required")
        sys.exit(1)

    # Get LLM configuration from environment
    model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

    # Run debugging session
    run_debugging_session(llm, working_dir, args.query, repos, errors_file, issue_url)


def run_debugging_session(
    llm: LLM,
    working_dir: Path,
    query: str,
    repos: list[str],
    errors_file: Path,
    issue_url: str,
):
    """Run the debugging session with the given configuration."""
    # Register and set up tools
    register_tool("BashTool", BashTool)
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("TaskTrackerTool", TaskTrackerTool)

    tools = [
        Tool(name="BashTool"),
        Tool(name="FileEditorTool"),
        Tool(name="TaskTrackerTool"),
    ]

    # Create agent
    agent = Agent(llm=llm, tools=tools)

    # Collect LLM messages for debugging
    llm_messages = []

    def conversation_callback(event: Event):
        if isinstance(event, LLMConvertibleEvent):
            llm_messages.append(event.to_llm_message())

    # Start conversation with local workspace
    conversation = Conversation(
        agent=agent, workspace=str(working_dir), callbacks=[conversation_callback]
    )

    # Send the debugging task
    debugging_prompt = create_debugging_prompt(query, repos, errors_file, issue_url)

    conversation.send_message(
        message=Message(
            role="user",
            content=[TextContent(text=debugging_prompt)],
        )
    )

    print("🤖 Starting debugging analysis...")
    try:
        conversation.run()

        print("\n" + "=" * 80)
        print("🎯 Debugging session completed!")
        print(f"📁 Results saved in: {working_dir}")
        print(f"💬 Total LLM messages: {len(llm_messages)}")

        # Show summary of what was accomplished
        print("\n📋 Session Summary:")
        print("- Queried Datadog logs for error analysis")
        print("- Cloned and analyzed relevant repositories")
        print("- Investigated potential root causes")
        print("- Attempted error reproduction")

        # Check for cloned repositories
        if working_dir.exists():
            cloned_repos = [
                d for d in working_dir.iterdir() if d.is_dir() and (d / ".git").exists()
            ]
            if cloned_repos:
                print(
                    f"- Cloned repositories: {', '.join(d.name for d in cloned_repos)}"
                )
    finally:
        # Clean up conversation
        logger.info("Closing conversation...")
        conversation.close()


if __name__ == "__main__":
    main()
