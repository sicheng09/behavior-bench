from typing import Dict, Protocol


class OpponentSampler(Protocol):
    def sample(self, scene_context: Dict, history: Dict) -> str:
        """Return a registered behavior-strategy name for a future episode."""
        ...
