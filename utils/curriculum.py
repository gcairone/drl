from typing import Any, Dict, List, Optional


class DifficultyCurriculum:
    """
    Adaptive curriculum over the randomization level in [0, 1].

    Every finished episode is recorded as a success (it survived until the time limit) or a failure
    (the humanoid fell). Once `min_episodes` results have accumulated, the success rate decides:
    at or above `up` the level rises by `step`, below `down` it falls by `step`, otherwise it holds.
    Each decision uses only the episodes finished since the previous one.
    """

    def __init__(self, start: float = 0.0, step: float = 0.02, up: float = 0.7, down: float = 0.3,
                 min_episodes: int = 20, max_level: float = 1.0) -> None:
        assert 0.0 <= down <= up <= 1.0, "need 0 <= down <= up <= 1"
        self.level = float(min(max(start, 0.0), max_level))
        self.step = step
        self.up = up
        self.down = down
        self.min_episodes = min_episodes
        self.max_level = max_level
        self.last_rate: Optional[float] = None  # success rate behind the latest decision
        self._results: List[bool] = []

    def record(self, success: bool) -> None:
        self._results.append(bool(success))

    def update(self) -> bool:
        """Takes a decision if enough episodes were recorded. Returns True if the level changed."""
        if len(self._results) < self.min_episodes:
            return False
        rate = sum(self._results) / len(self._results)
        self._results.clear()
        self.last_rate = rate
        old = self.level
        if rate >= self.up:
            self.level = min(self.max_level, self.level + self.step)
        elif rate < self.down:
            self.level = max(0.0, self.level - self.step)
        return self.level != old

    def state_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "last_rate": self.last_rate, "results": list(self._results)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.level = float(min(max(state["level"], 0.0), self.max_level))
        self.last_rate = state.get("last_rate")
        self._results = list(state.get("results", []))
