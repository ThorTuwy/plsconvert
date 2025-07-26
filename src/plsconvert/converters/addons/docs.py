from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import addMethodData, registerConverter
from plsconvert.utils.graph import PairList
from plsconvert.utils.files import runCommand
from plsconvert.utils.dependency import Dependencies, ToolDependency as Tool
from plsconvert.utils.dependency import LibDependency as Lib


@registerConverter
class pandoc(Converter):
    """
    Pandoc converter.
    """

    @property
    def name(self) -> str:
        return "Pandoc Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Tool("pandoc")])
    
    _SUPPORTED_PAIRS = PairList.all2all(
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

    @addMethodData(_SUPPORTED_PAIRS, False)
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        command = ["pandoc", str(input), "-o", str(output)]
        runCommand(command)


@registerConverter
class docxFromPdf(Converter):
    """
    Docx from pdf converter using pdf2docx.
    """

    @property
    def name(self) -> str:
        return "Pdf2Docx Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("pdf2docx")])

    @addMethodData(PairList(("pdf", "docx")), False)
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pdf2docx

        cv = pdf2docx.Converter(str(input))
        cv.convert(str(output), multi_processing=True)

@registerConverter
class csvFromExcel(Converter):
    """
    Csv from excel converter using pandas.
    """

    @property
    def name(self) -> str:
        return "Excel2Csv Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("pandas"), Lib("openpyxl")])

    @addMethodData(PairList.all2all(["xls", "xlsx", "xlsm", "xlsb"], ["csv"]), False)
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pandas as pd

        df = pd.read_excel(input)
        df.to_csv(output, index=False)
