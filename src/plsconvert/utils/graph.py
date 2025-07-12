import copy
from collections import deque
from plsconvert.utils.files import fileType


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


def bfs(start: str, end: str, adj: dict[str, (str, str)]) -> list[str]:
    visited = []
    queue = deque([(start, [])])

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
