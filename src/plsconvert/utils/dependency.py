import importlib.util
from plsconvert.utils.files import runCommand


def checkLibsDependencies(dependencies: list[str]) -> bool:
    for dependency in dependencies:
        if not importlib.util.find_spec(dependency):
            return False

    return True


def checkToolsDependencies(dependencies: list[str]) -> bool:
    try:
        for dependency in dependencies:
            runCommand([dependency, "--help"])
    except:
        return False

    return True
