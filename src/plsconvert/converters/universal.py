from pathlib import Path
import tempfile
import sys
import copy
from plsconvert.utils.graph import bfs

from plsconvert.converters.compression import sevenZip, tar
from plsconvert.converters.docs import pandoc, docxFromPdf
from plsconvert.converters.media import ffmpeg, imagemagick
from plsconvert.converters.audio import spectrogramMaker, textToSpeech, audioFromMidi
from plsconvert.converters.configs import configParser
from plsconvert.converters.ai import ocr
from halo import Halo


class universalConverter:
    def __init__(self):
        self.converters = [
            spectrogramMaker(),
            docxFromPdf(),
            ffmpeg(),
            pandoc(),
            imagemagick(),
            sevenZip(),
            tar(),
            configParser(),
            textToSpeech(),
            ocr(),
            audioFromMidi()
        ]
        self.convertersMap = {}
        for converter in self.converters:
            self.convertersMap[converter.name] = converter

        self.adj = self.__practical_adj()

    def __converter_factory(self, converter: str):
        return self.convertersMap.get(converter)

    def __practical_adj(self) -> dict[str, list[list[str]]]:
        """Build adjacency dictionary with only available converters (dependencies met)"""
        adj = {}
        for converter in self.converters:
            if not converter.metDependencies():
                continue
            for key, value in converter.adj().items():
                if key not in adj:
                    adj[key] = copy.deepcopy(value)
                else:
                    adj[key].extend(value)
        return adj
    
    def __theoretical_adj(self) -> dict[str, list[list[str]]]:
        """Build adjacency dictionary with all converters (theoretical complete system)"""
        adj = {}
        for converter in self.converters:
            # No dependency check - include all converters
            for key, value in converter.adj().items():
                if key not in adj:
                    adj[key] = copy.deepcopy(value)
                else:
                    adj[key].extend(value)
        return adj
    
    def getAdjacency(self, theoretical: bool = False) -> dict[str, list[list[str]]]:
        """Get adjacency dictionary. If theoretical=True, returns complete graph without dependency checks."""
        if theoretical:
            return self.__theoretical_adj()
        else:
            return self.__practical_adj()

    def checkDependencies(self):
        for converter in self.converters:
            with Halo(
                text=f"Dependencies for {converter.name}",
                spinner="dots",
            ) as spinner:
                if converter.metDependencies():
                    spinner.succeed()
                else:
                    spinner.fail()

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        path = bfs(input_extension, output_extension, self.adj)

        if not path:
            input_extension = "generic"
            path = bfs("generic", output_extension, self.adj)

        if not path:
            print(f"No conversion path found from {input} to {output}.")
            sys.exit(1)

        print("Conversion path found:")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for conversion in path[:-1]:
                    with Halo(
                        text=f"Converting from {input_extension} to {conversion[0]} with {conversion[1]}",
                        spinner="dots",
                    ) as spinner:
                        converter = self.__converter_factory(conversion[1])
                        temp_output = (
                            Path(temp_dir) / f"{output.stem + '.' + conversion[0]}"
                        )
                        converter.convert(
                            input, temp_output, input_extension, conversion[0]
                        )
                        input = temp_output
                        input_extension = conversion[0]

                        spinner.succeed()

                with Halo(
                    text=f"Final conversion {input_extension} to {output_extension} with {path[-1][1]}",
                    spinner="dots",
                ) as spinner:
                    converter = self.__converter_factory(path[-1][1])
                    converter.convert(input, output, input_extension, output_extension)
                    spinner.succeed()

        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the input file exists.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
