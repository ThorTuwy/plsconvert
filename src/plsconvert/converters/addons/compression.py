from pathlib import Path
import tempfile
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import addMethodData, registerConverter
from plsconvert.utils.graph import PairList
from plsconvert.utils.files import runCommand
from plsconvert.utils.dependency import Dependencies, ToolDependency as Tool
from plsconvert.utils.dependency import getSevenZipPath
import platform


@registerConverter
class tar(Converter):
    """
    Tar converter.
    """

    @property
    def name(self) -> str:
        return "Tar Converter"

    @property
    def dependencies(self) -> Dependencies:
        # TODO: Make gzip a Lib instead of Tool
        return Dependencies([Tool("gzip"), Tool("bzip2"), Tool("xz")])

    @addMethodData(PairList.all2all(["generic", "tar", "tar.gz", "tar.bz2", "tar.xz"], ["generic", "tar", "tar.gz", "tar.bz2", "tar.xz"]), False)
    def tar_to_tar(
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


@registerConverter
class sevenZip(Converter):
    """
    7z converter.
    """
    
    # Define supported pairs as class variable cause its easier this way in this instance :P
    _SUPPORTED_PAIRS = PairList.all2all(
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
    
    @property
    def name(self) -> str:
        return "7z Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Tool("7z")])

    def _getSevenZipCommand(self) -> str:
        """Get the correct 7z command path based on platform and availability."""
        if platform.system() == "Windows":
            sevenzip_path = getSevenZipPath()
            if sevenzip_path:
                return sevenzip_path
        return "7z"  # Fallback for non-Windows or if available in PATH

    @addMethodData(_SUPPORTED_PAIRS, False)
    def generic_to_generic(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        sevenzip_cmd = self._getSevenZipCommand()
        
        if input_extension == "generic":
            # File/Folder => Compress
            command = [sevenzip_cmd, "a", str(output), str(input)]
            runCommand(command)
        elif output_extension == "generic":
            # Compress => File/Folder (decompression)
            output.mkdir(parents=True, exist_ok=True)
            command = [sevenzip_cmd, "x", str(input), f"-o{output}", "-y"]
            runCommand(command)
        else:
            # Compress => Other compress (using temporary directory)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # First: extract input to temporary directory
                extract_command = [sevenzip_cmd, "x", str(input), f"-o{temp_path}", "-y"]
                runCommand(extract_command)
                
                # Second: compress temporary directory contents to output
                compress_command = [sevenzip_cmd, "a", str(output)]
                # Add all files from temp directory
                for item in temp_path.iterdir():
                    compress_command.append(str(item))
                
                runCommand(compress_command)

