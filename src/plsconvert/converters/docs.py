from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.dependency import checkToolsDependencies, checkLibsDependencies
from plsconvert.utils.files import runCommand


class pandoc(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            ["docx", "doc", "odt", "rtf", "txt", "html", "md", "epub"],
            [
                "docx",
                "doc",
                "odt",
                "rtf",
                "txt",
                "html",
                "md",
                "epub",
                "pdf",
                "tex",
                "pptx",
                "csv",
            ],
        )

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        command = ["pandoc", str(input), "-o", str(output)]
        runCommand(command)

    def metDependencies(self) -> bool:
        return checkToolsDependencies(["pandoc"])


class docxFromPdf(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {"pdf": ["docx"]}

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pdf2docx

        cv = pdf2docx.Converter(str(input))
        cv.convert(output, multi_processing=True)

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["pdf2docx"])
