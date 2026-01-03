"""Test condensation with thinking blocks.

Verifies that conversation continues successfully after condensation when
thinking blocks are forgotten. Tests that signature verification works correctly
when earlier thinking blocks are condensed away while later ones remain.
"""

from openhands.sdk import get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.terminal import TerminalTool
from tests.integration.base import BaseIntegrationTest, SkipTest, TestResult


# Multi-step task requiring multiple tool calls with reasoning
INSTRUCTION = """
Solve this multi-step problem using bash commands. Show your reasoning for each step:

1. Calculate compound interest for $10,000 at 5% annually, compounded quarterly for 3
years:
   A = P(1 + r/n)^(nt) where P=10000, r=0.05, n=4, t=3

2. Calculate simple interest for the same amount:
   A = P(1 + rt) where P=10000, r=0.05, t=3

3. Calculate the difference between compound and simple interest

4. Write each intermediate result to separate files:
   - compound.txt: compound interest result
   - simple.txt: simple interest result
   - difference.txt: the difference

Use the 'bc' calculator command for each calculation. You can use echo and pipes to
calculate.

For example: echo "scale=2; 10000 * 1.05" | bc
"""

logger = get_logger(__name__)


class CondensationThinkingTest(BaseIntegrationTest):
    """Test condensation with thinking blocks."""

    INSTRUCTION: str = INSTRUCTION

    def __init__(self, *args, **kwargs):
        self.tool_loop_count = 0
        self.condensation_count = 0
        self.thinking_block_event_ids = []
        self.condensations = []
        self.post_condensation_errors = []
        super().__init__(*args, **kwargs)

        from openhands.sdk.llm.utils.model_features import get_features

        model_name = self.llm.model
        canonical = self.llm.model_info.get("model") if self.llm.model_info else None
        name = canonical or model_name or "unknown"
        features = get_features(name)

        supports_thinking = (
            features.supports_extended_thinking or features.supports_reasoning_effort
        )
        if not supports_thinking:
            raise SkipTest(
                f"Model '{name}' does not support extended thinking or reasoning effort"
            )

    @property
    def tools(self) -> list[Tool]:
        register_tool("TerminalTool", TerminalTool)
        return [Tool(name="TerminalTool")]

    @property
    def condenser(self) -> LLMSummarizingCondenser:
        """Configure condenser for manual triggering only.

        High limits prevent auto-condensation. keep_first=1 ensures first
        thinking block is forgotten while second is kept.
        """
        condenser_llm = self.llm.model_copy(update={"usage_id": "test-condenser-llm"})
        return LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=10000,
            max_tokens=100000,
            keep_first=1,
        )

    @property
    def max_iteration_per_run(self) -> int:
        return 30

    def conversation_callback(self, event):
        super().conversation_callback(event)

        if isinstance(event, ActionEvent) and event.thinking_blocks:
            self.tool_loop_count += 1
            self.thinking_block_event_ids.append(event.id)

        if isinstance(event, Condensation):
            self.condensation_count += 1
            self.condensations.append(event)

    def setup(self) -> None:
        pass

    def verify_result(self) -> TestResult:
        if self.tool_loop_count < 2:
            return TestResult(
                success=False,
                reason=f"Expected 2+ thinking blocks, got {self.tool_loop_count}",
            )

        if self.condensation_count == 0:
            return TestResult(
                success=False,
                reason="Manual condensation was not triggered",
            )

        # Check that first thinking block was forgotten
        first_thinking_forgotten = False
        second_thinking_kept = True

        if len(self.thinking_block_event_ids) >= 2:
            first_thinking_id = self.thinking_block_event_ids[0]
            second_thinking_id = self.thinking_block_event_ids[1]

            for condensation in self.condensations:
                if first_thinking_id in condensation.forgotten_event_ids:
                    first_thinking_forgotten = True
                if second_thinking_id in condensation.forgotten_event_ids:
                    second_thinking_kept = False

        if not first_thinking_forgotten:
            return TestResult(
                success=False,
                reason="First thinking block was not forgotten during condensation",
            )

        if not second_thinking_kept:
            return TestResult(
                success=False,
                reason=(
                    "Second thinking block was incorrectly forgotten during "
                    "condensation"
                ),
            )

        if self.post_condensation_errors:
            return TestResult(
                success=False,
                reason=f"Post-condensation errors: {self.post_condensation_errors}",
            )

        return TestResult(
            success=True,
            reason=(
                f"Condensation successful with {self.tool_loop_count} thinking blocks"
            ),
        )

    def execute_conversation(self) -> None:
        """Custom conversation flow with follow-up messages and manual condensation."""
        # Send the instruction and run until completion
        self.conversation.send_message(self.INSTRUCTION)
        self.conversation.run()

        # The agent should resolve the task with a single tool loop and just one
        # thinking block

        self.conversation.send_message(
            "Now verify your calculations are correct by re-running the commands "
            "and comparing the results. Show your reasoning about whether the "
            "calculations match."
        )
        self.conversation.run()

        # Manually trigger condensation
        self.conversation.condense()
        self.conversation.run()
        
        # Send one more message to see if conversation can continue
        self.conversation.send_message("What was the final compound interest result?")
        self.conversation.run()