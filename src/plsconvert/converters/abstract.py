from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import copy


class Converter(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def exist(self) -> bool:
        return True

    def adj(self) -> dict[str, list[list[str]]]:
        adj = self.adjConverter()
        for key in adj:
            adj[key] = [[ext, self.name] for ext in adj[key]]
        return adj

    @abstractmethod
    def adjConverter(self) -> dict[str, list[list[str]]]:
        pass

    @abstractmethod
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        pass


def conversionFromToAdj(
    conversion_from: list[str], conversion_to: list[str]
) -> dict[str, (str, str)]:
    """
    Create a dictionary mapping from conversion_from to conversion_to.
    """

    adj = {}

    for ext in conversion_from:
        adj[ext] = conversion_to

    return adj


def mergeAdj(adj1, adj2):
    """
    Merge two adjacency dictionaries.
    """
    for key, value in adj2.items():
        if key not in adj1:
            adj1[key] = copy.deepcopy(value)
        else:
            adj1[key].extend(value)

    return adj1


def runCommand(command: list[str]) -> None:
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def fileType(extension: str) -> str:
    return {
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "gif": "image",
        "bmp": "image",
        "svg": "image",
        "webp": "image",
        "tiff": "image",
        "mp4": "video",
        "mov": "video",
        "avi": "video",
        "mkv": "video",
        "flv": "video",
        "wmv": "video",
        "webm": "video",
        "mpeg": "video",
        "mp3": "audio",
        "wav": "audio",
        "ogg": "audio",
        "aac": "audio",
        "flac": "audio",
        "m4a": "audio",
        "wma": "audio",
        "alac": "audio",
    }.get(extension, "other")
