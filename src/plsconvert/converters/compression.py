from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.files import runCommand
from plsconvert.utils.dependency import checkToolsDependencies, getSevenZipPath
import platform


class tar(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            ["generic", "tar", "tar.gz", "tar.bz2", "tar.xz"],
            ["generic", "tar", "tar.gz", "tar.bz2", "tar.xz"],
        )

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import tarfile

        extensionToMode = {
            "tar.gz": ("gzip", "w:gz"),
            "tar.bz2": ("bzip2", "w:bz2"),
            "tar.xz": ("xz", "w:xz"),
            "tar": ("", "w"),
        }
        if input_extension == "generic":
            # File/Folder => Compress
            mode = extensionToMode[output_extension][1]
            output.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(output), mode) as tar:
                tar.add(str(input), arcname=input.name)
        elif output_extension == "generic":
            # Compress => File/Folder
            output.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(input), "r") as tar:
                tar.extractall(path=output, filter="data")
        else:
            # Compress => Other compress
            input_command = extensionToMode[input_extension][0]
            output_command = extensionToMode[output_extension][0]
            command = [
                input_command,
                "-dc",
                str(input),
                "|",
                output_command,
                str(output),
            ]
            runCommand(command)

    def metDependencies(self) -> bool:
        return checkToolsDependencies(["gzip", "bzip2", "xz"])


class sevenZip(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            [
                "generic",
                "7z",
                "xz",
                "bz2",
                "gz",
                "tar",
                "zip",
                "wim",
                "apfs",
                "ar",
                "arj",
                "cab",
                "chm",
                "cpio",
                "cramfs",
                "dmg",
                "ext",
                "fat",
                "gpt",
                "hfs",
                "hex",
                "iso",
                "lzh",
                "lzma",
                "mbr",
                "msi",
                "nsi",
                "ntfs",
                "qcow2",
                "rar",
                "rpm",
                "squashfs",
                "udf",
                "uefi",
                "vdi",
                "vhd",
                "vhdx",
                "vmdk",
                "xar",
                "z",
            ],
            ["generic", "7z", "xz", "bz2", "gz", "tar", "zip", "wim"],
        )

    def _getSevenZipCommand(self) -> str:
        """Get the correct 7z command path based on platform and availability."""
        if platform.system() == "Windows":
            sevenzip_path = getSevenZipPath()
            if sevenzip_path:
                return sevenzip_path
        return "7z"  # Fallback for non-Windows or if available in PATH

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        sevenzip_cmd = self._getSevenZipCommand()
        
        if input_extension == "generic":
            # File/Folder => Compress
            command = [sevenzip_cmd, "a", str(output), str(input)]
        elif output_extension == "generic":
            # Compress => File/Folder (decompression)
            # Ensure the output directory exists and use it as extraction destination
            if output.is_dir() or str(output).endswith(('/', '\\')):
                # Output is already a directory or path ends with separator
                extraction_dir = output
            else:
                # If output doesn't exist, treat it as a directory path
                extraction_dir = output
            
            # Create the directory if it doesn't exist
            extraction_dir.mkdir(parents=True, exist_ok=True)
            
            # Use 'x' command to preserve directory structure and extract to specified directory
            command = [sevenzip_cmd, "x", str(input), f"-o{extraction_dir}", "-y"]
        else:
            # Compress => Other compress
            command = [sevenzip_cmd, "e", "-so", str(input), "|", sevenzip_cmd, "a", "-si", str(output)]

        runCommand(command)

    def metDependencies(self) -> bool:
        return checkToolsDependencies(["7z"])
