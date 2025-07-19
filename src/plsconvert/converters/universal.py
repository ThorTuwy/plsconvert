from pathlib import Path
import tempfile
import sys
import copy
from plsconvert.utils.graph import bfs  
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import ConversionAdj
from plsconvert.converters.registry import ConverterRegistry
from halo import Halo

class universalConverter:
    """Universal converter that uses the centralized registry to access all available converters."""

    def __init__(self):
        """Initialize the universal converter with all registered converters."""
        self.converters = [converter_class() for converter_class in ConverterRegistry.get_all_converters()]
        self.convertersMap: dict[str, Converter] = {converter.name: converter for converter in self.converters}
        self.adj = self.getAdjacency(theoretical=False)

    def getAdjacency(self, theoretical: bool = False) -> ConversionAdj:
        """Get adjacency dictionary. If theoretical=True, returns complete graph without dependency checks."""
        adj: ConversionAdj = ConversionAdj()
        for converter in self.converters:
            if not (theoretical or converter.dependencies.check):
                continue
            for source, conversions in converter.adj().items():
                if source not in adj:
                    adj[source] = copy.deepcopy(conversions)
                else:
                    adj[source].extend(conversions)
        return adj

    def checkDependencies(self):
        """Check dependencies for all registered converters."""
        for converter in self.converters:
            if converter.dependencies.check:
                text=f"Dependencies for {converter}"
            else:
                text=f"Dependencies for {converter}. Check your dependencies: {converter.dependencies.missing()}"

            with Halo(
                text=text,
                spinner="dots",
            ) as spinner:
                if converter.dependencies.check:
                    spinner.succeed()
                else:
                    spinner.fail()

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        """Convert a file from one format to another using the best available conversion path."""
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
                        text=f"Converting from {input_extension} to {conversion.format} with {conversion.converter}",
                        spinner="dots",
                    ) as spinner:
                        temp_output = (
                            Path(temp_dir) / f"{output.stem + '.' + conversion.format}"
                        )
                        conversion.converter.convert(
                            input, temp_output, input_extension, conversion.format
                        )
                        input = temp_output
                        input_extension = conversion.format

                        spinner.succeed()

                with Halo(
                    text=f"Final conversion {input_extension} to {output_extension} with {path[-1].converter}",
                    spinner="dots",
                ) as spinner:
                    path[-1].converter.convert(input, output, input_extension, output_extension)
                    spinner.succeed()

        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the input file exists.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
