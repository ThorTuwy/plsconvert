import copy
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Deque
from plsconvert.utils.files import fileType




def conversionFromToAdj(
    conversionFrom: List[str], conversionTo: List[str]
) -> Dict[str, List[str]]:
    """
    Create a dictionary mapping from conversionFrom to conversionTo.
    """
    adj = {}

    for source in conversionFrom:
        adj[source] = conversionTo

    return adj


def mergeAdj(adj1: Dict[str, List[str]], adj2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge two adjacency dictionaries.
    """
    for key, value in adj2.items():
        if key not in adj1:
            adj1[key] = copy.deepcopy(value)
        else:
            adj1[key].extend(value)

    return adj1


def bfs(start: str, end: str, adj: Dict[str, List[List[str]]]) -> List[List[str]]:
    visited = []
    queue: Deque[Tuple[str, List[List[str]]]] = deque([(start, [])])

    while queue:
        current, path = queue.popleft()

        if current == end:
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
                path_copy.append([neighbor, converter])
                queue.append((neighbor, path_copy))

    return []
    