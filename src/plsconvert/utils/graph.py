from collections import deque
from typing import Tuple, Deque, TYPE_CHECKING
from plsconvert.utils.files import fileType
from typing import TypeAlias

if TYPE_CHECKING:
    from plsconvert.converters.abstract import Converter

Format: TypeAlias = str

Pair: TypeAlias = tuple[Format, Format]

class PairList(list[Pair]):
    """
    A list of pairs of formats.
    """
    def __init__(self, *args: Pair):
        super().__init__(args)

    def findByInput(self, input: Format) -> "PairList":
        return PairList(*[pair for pair in self if pair[0] == input])

    def findByOutput(self, output: Format) -> "PairList":
        return PairList(*[pair for pair in self if pair[1] == output])

    def findByInputAndOutput(self, input: Format, output: Format) -> "PairList":
        return PairList(
            *[pair for pair in self if pair[0] == input and pair[1] == output]
        )

    def __add__(self, other: "PairList | list[Pair]") -> "PairList":  # type: ignore
        return PairList(*super().__add__(other))

    @classmethod
    def all2all(
        cls, inputList: list[Format], outputList: list[Format], excludeSelf: bool = True
    ) -> "PairList":
        """
        Create a PairList of all possible pairs of inputs and outputs.
        If excludeSelf is True, the pair (input, input) is excluded.
        """
        return PairList(
            *[(input, output) for input in inputList for output in outputList if input != output if excludeSelf]
        )


class ConversionData(dict[str, "str | Converter | bool"]):
    """
    A dictionary of conversion data.
    """
    def __init__(self, converter: "Converter", methodName: str, hasProgressBar: bool = False, **kwargs):
        super().__init__(**kwargs)
        self["converter"] = converter
        self["methodName"] = methodName
        self["hasProgressBar"] = hasProgressBar
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def converter(self) -> "Converter":
        return self["converter"]  # type: ignore

    @converter.setter
    def converter(self, converter: "Converter"):
        self["converter"] = converter

    @property
    def hasProgressBar(self) -> bool:
        return self["hasProgressBar"]  # type: ignore

    @hasProgressBar.setter
    def hasProgressBar(self, hasProgressBar: bool):
        self["hasProgressBar"] = hasProgressBar

    @property
    def methodName(self) -> str:
        return self["methodName"]  # type: ignore
    
    @methodName.setter
    def methodName(self, methodName: str):
        self["methodName"] = methodName


class Conversion(tuple[Pair, ConversionData]):
    """
    A conversion is a pair of a format and a conversion data.
    """
    def __new__(cls, pair: Pair, conversionData: ConversionData):
        return super().__new__(cls, (pair, conversionData))

    @property
    def pair(self) -> Pair:
        return self[0]

    @property
    def input(self) -> Format:
        return self.pair[0]

    @property
    def output(self) -> Format:
        return self.pair[1]

    @property
    def converter(self) -> "Converter":
        return self[1].converter
    
    @property
    def methodName(self) -> str:
        return self[1].methodName


class ConversionList(list[Conversion]):
    """
    A list of conversions.
    """
    def __init__(self, *args: list[Conversion] | Conversion):
        super().__init__()
        for arg in args:
            if isinstance(arg, list):
                for conversion in arg:
                    self.append(conversion)
            else:
                self.append(arg)

    def findByPair(self, pair: Pair) -> "ConversionList":
        return ConversionList(
            *[conversion for conversion in self if conversion.pair == pair]
        )

    def findByInput(self, input: Format) -> "ConversionList":
        return ConversionList(
            *[conversion for conversion in self if conversion.input == input]
        )

    def findByOutput(self, output: Format) -> "ConversionList":
        return ConversionList(
            *[conversion for conversion in self if conversion.output == output]
        )

    def findByConverter(self, converter: "Converter") -> "ConversionList":
        return ConversionList(
            *[conversion for conversion in self if conversion.converter == converter]
        )

class Graph(dict[Format, ConversionList]):
    """
    A graph is a dictionary of formats and their conversion lists.
    """
    def __init__(self, *args: dict[Format, ConversionList]):
        for arg in args:
            for key, value in arg.items():
                if key not in self:
                    self[key] = value
                else:
                    self[key].extend(value)

    def filter(self, formats: list[Format]) -> "Graph":
        """
        Filter the graph to only include the given input formats.
        """
        return Graph({key: value for key, value in self.items() if key in formats})
    
    def hardFilter(self, formats: list[Format]) -> "Graph":
        """
        Filter the graph to only include the given input and output formats.
        """
        copy = self.copy()
        listOfKeys = list(copy.keys())
        for key in listOfKeys:
            if key not in formats:
                del copy[key]
            else:
                copy[key] = ConversionList(*[conversion for conversion in copy[key] if conversion.output in formats])
        return Graph(copy)
    
    def getAllConverters(self) -> list["Converter"]:
        """
        Get all converters in the graph.
        """
        return list(set([conversion.converter for conversionList in self.values() for conversion in conversionList]))
    
    def getAllUniqueFormats(self) -> list[Format]:
        """
        Get all unique formats in the graph.
        """
        return list(set(self.getAllSourceFormats()) | set(self.getAllTargetFormats()))
    
    def getAllSourceFormats(self) -> list[Format]:
        """
        Get all source formats in the graph.
        """
        return list(self.keys())
    
    def getAllTargetFormats(self) -> list[Format]:
        """
        Get all target formats in the graph.
        """
        return list(set([conversion.output for conversionList in self.values() for conversion in conversionList]))
    
    def getAllConversions(self) -> "ConversionList":
        """
        Get all conversions in the graph.
        """
        return ConversionList([conversion for conversionList in self.values() for conversion in conversionList])

    def numConnectionsPerConverter(self) -> dict[str, int]:
        """
        Count the number of connections for each converter.
        Returns a dictionary of converters and the number of connections.
        """
        counts: dict[str, int] = {}
        for _, value in self.items():
            for conversion in value:
                if str(conversion.converter) not in counts:
                    counts[str(conversion.converter)] = 0
                counts[str(conversion.converter)] += 1

        # Remove duplicates cause some converters appear multiple times
        counts = {converter: count for converter, count in counts.items() if count > 0}

        return counts

    def numConnectionsPerFormat(self) -> dict[Format, int]:
        """
        Count the number of connections for each format.
        Returns a dictionary of formats and the number of connections.
        """
        return {key: len(value) for key, value in self.items()}

    def __add__(self, other: "Graph | ConversionList | Conversion") -> "Graph":
        if isinstance(other, Graph):
            return Graph(self, other)
        elif isinstance(other, ConversionList):
            for conversion in other:
                if conversion.input not in self:
                    self[conversion.input] = ConversionList()
                self[conversion.input].append(conversion)
            return self
        elif isinstance(other, Conversion):
            if other.input not in self:
                self[other.input] = ConversionList()
            self[other.input].append(other)
            return self
        else:
            raise TypeError(f"Cannot add {type(other)} to Graph")

    def bfs(self, start: Format, end: Format) -> list[Conversion]:
        """
        Breadth-first search for the shortest path between two formats.
        Returns a list of conversions to be done to convert from start to end.
        """
        visited = []
        queue: Deque[Tuple[Format, list[Conversion]]] = deque([(start, [])])

        while queue:
            current, path = queue.popleft()

            if current == end:
                return path
            visited.append(current)

            # Never do things after audio=>video
            if (
                len(path) == 1
                and fileType(path[0].input) == "audio"
                and fileType(path[0].output) == "video"
            ):
                continue

            for conversion in self.get(current, []):
                if conversion.output not in visited:
                    path_copy = path.copy()
                    path_copy.append(conversion)
                    queue.append((conversion.output, path_copy))

        return []
