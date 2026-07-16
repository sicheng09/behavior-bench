from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    policy_index: int
    reward_mode: str
    trainable: bool


class StrategyRegistry:
    def __init__(self):
        self._strategies = {}

    def register(self, definition: StrategyDefinition) -> None:
        if definition.name in self._strategies:
            raise ValueError(f"Strategy already registered: {definition.name}")
        self._strategies[definition.name] = definition

    def resolve(self, name: str) -> StrategyDefinition:
        try:
            return self._strategies[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown or unimplemented adversarial strategy: {name}"
            ) from exc


DEFAULT_STRATEGY_REGISTRY = StrategyRegistry()
DEFAULT_STRATEGY_REGISTRY.register(
    StrategyDefinition("ego_drive_recurrent", 0, "base", True)
)
DEFAULT_STRATEGY_REGISTRY.register(
    StrategyDefinition(
        "primary_opponent_drive_recurrent", 1, "adversarial", True
    )
)
