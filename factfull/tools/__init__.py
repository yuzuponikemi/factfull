"""factfull-side tools exposed to the test-smith ReAct agent.

The Knowledge Layer (factfull) provides primitives. When multi-step
judgement is needed, the Agent Layer (test-smith) drives the loop and
calls these tools. Subclass ``test_smith.agents.react.Tool`` here and
register the resulting tool instances in the agent's tool list at the
pipeline entry point.

Import direction is one-way: factfull → test_smith.
"""

from .chapter import SubmitChaptersTool
from .factcheck import ExtractClaimsTool, VerifyClaimTool

__all__ = [
    "SubmitChaptersTool",
    "ExtractClaimsTool",
    "VerifyClaimTool",
]
