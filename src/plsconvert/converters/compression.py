from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.files import runCommand
from plsconvert.utils.dependency import checkToolsDependencies


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

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        if input_extension == "generic":
            command = ["7z", "a", str(output), str(input)]
        elif output_extension == "generic":
            # Compress => File/Folder
            command = ["7z", "e", str(input), f"-o{output.parent}", "-y"]
        else:
            # Compress => Other compress
            command = ["7z", "e", "-so", str(input), "|", "7z", "a", "-si", str(output)]

        runCommand(command)

    def metDependencies(self) -> bool:
        return checkToolsDependencies(["7z"])
