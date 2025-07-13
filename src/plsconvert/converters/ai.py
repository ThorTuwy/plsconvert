from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.dependency import checkLibsDependencies

class ocr(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "png": ["md","txt"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import rapidocr
        from PIL import Image

        # Load the image
        img = Image.open(input)

        # Run OCR
        ocr_engine = rapidocr.RapidOCR()
        results = ocr_engine(img)

        # Save to output file
        with open(output, "w", encoding="utf-8") as f:
            f.write(results.to_markdown())

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["rapidocr", "onnxruntime","PIL"])