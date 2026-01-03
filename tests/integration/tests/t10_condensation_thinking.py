"""Test condensation with thinking blocks.

Verifies that conversation continues successfully after condensation when
thinking blocks are forgotten. Tests that signature verification works correctly
when earlier thinking blocks are condensed away while later ones remain.
"""

from itertools import chain

from openhands.sdk import get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm.utils.model_features import get_features
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
        self.thinking_block_events = []
        self.condensations = []
        
        super().__init__(*args, **kwargs)

        # Grab the model name so we can check features
        model_name = self.llm.model
        canonical = self.llm.model_info.get("model") if self.llm.model_info else None
        name = canonical or model_name or "unknown"
        
        features = get_features(name)

        # This test only works for models that generate thinking blocks, so skip
        # otherwise
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
            keep_first=1,
        )

    @property
    def max_iteration_per_run(self) -> int:
        return 30

    def conversation_callback(self, event):
        super().conversation_callback(event)

        if isinstance(event, ActionEvent) and event.thinking_blocks:
            self.thinking_block_events.append(event)
            
        if isinstance(event, Condensation):
            self.condensations.append(event)

    def setup(self) -> None:
        pass

    def verify_result(self) -> TestResult:
        # Sanity checks to ensure the flow worked the way we expected: at least three
        # thinking blocks and one condensation
        if len(self.thinking_block_events) < 3:
            return TestResult(
                success=False,
                reason=(
                    f"Expected at least 3 thinking blocks, got "
                    f"{len(self.thinking_block_events)}"
                ),
            )
        
        if len(self.condensations) != 1:
            return TestResult(
                success=False,
                reason=(
                    f"Expected exactly 1 condensation event, got "
                    f"{len(self.condensations)}"
                ),
            )

        # Check the conditions of the actual test. The condensation should forget the
        # first thinking block but keep the second.
        forgotten_event_ids = list(chain.from_iterable(
            c.forgotten_event_ids for c in self.condensations
        ))

        if self.thinking_block_events[0].id not in forgotten_event_ids:
            return TestResult(
                success=False,
                reason="First thinking block was not forgotten during condensation",
            )
        
        if self.thinking_block_events[1].id in forgotten_event_ids:
            return TestResult(
                success=False,
                reason=(
                    "Second thinking block was incorrectly forgotten during "
                    "condensation"
                ),
            )

        # In all other cases, if the test was successfully run without errors then it's
        # a pass
        return TestResult(
            success=True,
            reason=(
                f"Condensation successful with {len(self.thinking_block_events)} "
                "thinking blocks"
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
        self.conversation.send_message(
            "Now run the commands again with 10% interest rate compounded annually."
            " Show your reasoning and explain which method will yield a larger return."
        )
        self.conversation.run()