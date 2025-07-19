import copy
from collections import deque
from typing import Tuple, Deque, Union
from plsconvert.converters.abstract import Converter
from plsconvert.utils.files import fileType
from typing import TypeAlias

Format: TypeAlias = str

class Conversion(tuple[Format, Converter]):
    @property
    def format(self) -> Format:
        return self[0]
    
    @property
    def converter(self) -> Converter:
        return self[1]

class ConversionAdj(dict[Format, list[Conversion]]):
    def filter(self, formats: list[Format]) -> "ConversionAdj":
        return ConversionAdj({key: value for key, value in self.items() if key in formats})
    
    def countConverters(self) -> tuple[dict[str, int], int]:
        """
        Count the number of connections for each converter.
        Returns a dictionary of converters and the number of connections.
        """
        totalConversions = 0
        counts: dict[str, int] = {}
        for _, value in self.items():
            for conversion in value:
                if str(conversion.converter) not in counts:
                    counts[str(conversion.converter)] = 0
                counts[str(conversion.converter)] += 1
                totalConversions += 1

        # Remove duplicates cause some converters appear multiple times
        counts = {converter: count for converter, count in counts.items() if count > 0}

        return counts, totalConversions
    
    def countFormats(self) -> dict[Format, int]:
        """
        Count the number of connections for each format.
        Returns a dictionary of formats and the number of connections.
        """
        return {key: len(value) for key, value in self.items()}
    
    def __add__(self, other: "ConversionAdj") -> "ConversionAdj":
        return mergeAdj(self, other)

def conversionFromToAdj(
    conversionFrom: list[Union[str, Format]], conversionTo: list[str]
) -> ConversionAdj:
    """
    Create a dictionary mapping from conversionFrom to conversionTo.
    """
    adj: ConversionAdj = ConversionAdj()

    for source in conversionFrom:
        adj[source] = [(target, source) for target in conversionTo] # type: ignore

    return adj


def mergeAdj(adj1: ConversionAdj, adj2: ConversionAdj) -> ConversionAdj:
    """
    Merge two adjacency dictionaries.
    """
    for key, value in adj2.items():
        if key not in adj1:
            adj1[key] = copy.deepcopy(value)
        else:
            adj1[key].extend(value)

    return adj1


def bfs(start: Union[str, Format], end: Union[str, Format], adj: ConversionAdj) -> list[Conversion]:
    visited = []
    queue: Deque[Tuple[Format, list[Conversion]]] = deque([(Format(start), [])])

    while queue:
        current, path = queue.popleft()

        if current == Format(end):
            return path
        visited.append(current)

        # Never do things after audio=>video
        if (
            len(path) == 1
            and fileType(start) == "audio"
            and fileType(path[0][0]) == "video"
        ):
            continue

        for neighbor, converter in adj.get(current, []):
            if neighbor not in visited:
                path_copy = path.copy()
                path_copy.append(Conversion((neighbor, converter)))
                queue.append((neighbor, path_copy))

    return []
    