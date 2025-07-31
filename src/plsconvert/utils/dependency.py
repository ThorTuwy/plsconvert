import importlib.util
import platform
from pathlib import Path
from plsconvert.utils.files import runCommand

class LibDependency(str):
    @property
    def check(self) -> bool:
        return importlib.util.find_spec(self) is not None

class ToolDependency(str):
    @property
    def check(self) -> bool:
        if "7z" in self and platform.system() == "Windows":
            return getSevenZipPath() is not None
        # DEV ONLY
        # else:
        #     if LibDependency(self).check:
        #         print(f"{self} found as library, should be changed from ToolDependency to LibDependency")

        #         return False

        try:
            runCommand([self, "--help"])
        except Exception:
            return False

        return True
        
class Dependencies:
    def __init__(self, dependencies: list[LibDependency | ToolDependency]):
        self.dependencies = dependencies

    @classmethod
    def empty(cls) -> "Dependencies":
        return cls([])

    @property
    def check(self) -> bool:
        for dependency in self.dependencies:
            if isinstance(dependency, LibDependency):
                if not dependency.check:
                    return False
            elif isinstance(dependency, ToolDependency):
                if not dependency.check:
                    return False
        return True
    
    def missing(self) -> list[str]:
        return [dependency for dependency in self.dependencies if not dependency.check]

def getSevenZipPath() -> str | None:
    """Get 7z.exe path on Windows. Returns path if found, None otherwise."""
    if platform.system() != "Windows":
        return None
    
    # Check standard installation paths
    standard_paths = [
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files (x86)/7-Zip/7z.exe"),
    ]
    
    for path in standard_paths:
        if path.exists():
            return str(path)
    
    # Check if available in PATH
    try:
        runCommand(["7z", "--help"])
        return "7z"
    except Exception:
        pass
    
    return None
