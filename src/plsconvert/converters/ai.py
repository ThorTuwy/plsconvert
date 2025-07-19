from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import register_converter
from plsconvert.utils.graph import ConversionAdj
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.dependency import Dependencies, LibDependency as Lib

@register_converter
class ocr(Converter):
    """
    OCR converter using RapidOCR.
    """

    @property
    def name(self) -> str:
        return "OCR Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("rapidocr"), Lib("onnxruntime"), Lib("PIL")])

    def adjConverter(self) -> ConversionAdj:
        return conversionFromToAdj(["png"], ["md","txt"])

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import rapidocr

        # Run OCR
        ocr_engine = rapidocr.RapidOCR()
        results = ocr_engine(input) # type: ignore

        # Save to output file
        with open(output, "w", encoding="utf-8") as f:
            f.write(results.to_markdown()) # type: ignore
