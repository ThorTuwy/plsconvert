from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.dependency import checkToolsDependencies, checkLibsDependencies
from plsconvert.utils.files import runCommand


class pandoc(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            [
                "bib",
                "biblatex",
                "bits",
                "commonmark",
                "creole",
                "csljson",
                "csv",
                "djot",
                "docbook",
                "docx",
                "dokuwiki",
                "emacs-muse",
                "endnotexml",
                "epub",
                "fb2",
                "gfm",
                "haddock",
                "html",
                "ipynb",
                "jats",
                "jira",
                "json",
                "latex",
                "lua",
                "man",
                "md",
                "mdoc",
                "mediawiki",
                "muse",
                "odt",
                "opml",
                "org",
                "pod",
                "ris",
                "rst",
                "rtf",
                "t2t",
                "textile",
                "tikiwiki",
                "tsv",
                "twiki",
                "typst",
                "vimwiki",
            ],
            [
                "adoc",
                "asciidoc",
                "beamer",
                "bib",
                "biblatex",
                "commonmark",
                "context",
                "csljson",
                "djot",
                "docbook",
                "docx",
                "dokuwiki",
                "dzslides",
                "emacs-muse",
                "epub",
                "fb2",
                "gfm",
                "haddock",
                "html",
                "html4",
                "html5",
                "icml",
                "ipynb",
                "jats",
                "jira",
                "json",
                "latex",
                "lua",
                "man",
                "markdown",
                "markua",
                "mediawiki",
                "ms",
                "muse",
                "odt",
                "opendocument",
                "opml",
                "org",
                "pdf",
                "plain",
                "pptx",
                "revealjs",
                "rst",
                "rtf",
                "s5",
                "slideous",
                "slidy",
                "tei",
                "tex",
                "texinfo",
                "textile",
                "typst",
                "xwiki",
                "zimwiki",
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

class csvFromExcel(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            ["xls", "xlsx", "xlsm", "xlsb"],
            ["csv"],
        )

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pandas as pd

        df = pd.read_excel(input)
        df.to_csv(output, index=False)

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["pandas", "openpyxl"])

