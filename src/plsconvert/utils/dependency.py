import importlib.util
import os
import platform
from pathlib import Path
from plsconvert.utils.files import runCommand


def checkLibsDependencies(dependencies: list[str]) -> bool:
    for dependency in dependencies:
        if not importlib.util.find_spec(dependency):
            return False

    return True


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
    except:
        pass
    
    return None


def checkToolsDependencies(dependencies: list[str]) -> bool:
    # Special handling for 7z on Windows
    if "7z" in dependencies and platform.system() == "Windows":
        other_deps = [dep for dep in dependencies if dep != "7z"]
        
        # Check 7z with Windows-specific detection
        if getSevenZipPath() is None:
            return False
        
        # Check other dependencies normally
        if other_deps:
            try:
                for dependency in other_deps:
                    runCommand([dependency, "--help"])
            except:
                return False
        
        return True
    
    # Original implementation for non-Windows or non-7z cases
    try:
        for dependency in dependencies:
            runCommand([dependency, "--help"])
    except:
        return False

    return True
