from pathlib import Path
import tempfile
import sys
import warnings
from plsconvert.converters.registry import ConverterRegistry
from halo import Halo
from plsconvert.utils.errors import OutputFileNotFoundError
import shutil

class universalConverter:
    """Universal converter that uses the centralized registry to access all available converters."""

    def checkAllDependencies(self):
        """Check dependencies for all registered converters."""
        for converter in ConverterRegistry.theoreticalGraph.getAllConverters():
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

    @staticmethod
    def save(tempOutput: Path, output: Path) -> Path:
        """
        Save the conversion result to the output file.
        """
        if not tempOutput.exists():
            raise OutputFileNotFoundError(f"Temp output file {tempOutput} not found. Conversion may have failed.")

        shutil.move(tempOutput, output)
        if not output.exists():
            raise OutputFileNotFoundError(f"Output file {output} not found. Transfer may have failed.")
        
        return output

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        """Convert a file from one format to another using the best available conversion path."""
        conversionToOutput = ConverterRegistry.practicalGraph.bfs(input_extension, output_extension)

        if not conversionToOutput:
            input_extension = "generic"
            conversionToOutput = ConverterRegistry.practicalGraph.bfs("generic", output_extension)

        if not conversionToOutput:
            print(f"No conversion path found from {input} to {output}.")
            sys.exit(1)

        print("Conversion path found:")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for conversion in conversionToOutput:
                    converter = conversion.converter
                    
                    # Check if this converter method has progress bar support for this specific conversion
                    if converter.hasPairProgressBar(conversion.pair):
                        print(f"Converting {input_extension} to {conversion.output} with {converter}")
                        
                        temp_output = (
                            Path(temp_dir) / f"{output.stem + '.' + conversion.output}"
                        )
                        converter.convert(
                            input, temp_output, input_extension, conversion.output
                        )
                        
                        # Close progress bar if it exists
                        if converter.progressBar:
                            converter.progressBar.close()
                            warnings.warn(f"Progress bar for {conversion.pair} in converter {converter} should be closed inside the converter.")
                    else:
                        # Use Halo for converters without progress bar
                        with Halo(
                            text=f"Converting from {input_extension} to {conversion.output} with {converter}",
                            spinner="dots",
                        ) as spinner:
                            temp_output = (
                                Path(temp_dir) / f"{output.stem + '.' + conversion.output}"
                            )
                            converter.convert(
                                input, temp_output, input_extension, conversion.output
                            )
                            spinner.succeed()
                    
                    input = temp_output
                    input_extension = conversion.output

                universalConverter.save(input, output)

        except OutputFileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the input file exists.", file=sys.stderr)
            sys.exit(1)

