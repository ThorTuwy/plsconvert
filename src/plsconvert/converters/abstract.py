from abc import ABC, abstractmethod
from pathlib import Path


class Converter(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def exist(self) -> bool:
        return True

    def adj(self) -> dict[str, list[list[str]]]:
        old_adj = self.adjConverter()
        adj = {}
        for source in old_adj:
            new_targets = []
            for target in old_adj[source]:
                if target not in adj:
                    adj[target] = []
                if source != target:  # Avoid self-loops
                    new_targets.append([target, self.name])
            adj[source] = new_targets
        return adj

    @abstractmethod
    def adjConverter(self) -> dict[str, list[list[str]]]:
        pass

    @abstractmethod
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        pass

    @abstractmethod
    def metDependencies(self) -> bool:
        pass
