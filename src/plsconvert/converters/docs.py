from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.abstract import conversionFromToAdj, runCommand

import pdf2docx


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


class docxFromPdf(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {"pdf": ["docx"]}

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        cv = pdf2docx.Converter(str(input))
        cv.convert(output, multi_processing=True)
