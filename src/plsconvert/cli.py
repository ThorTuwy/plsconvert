from pathlib import Path
import argparse
import sys

import warnings
import logging

from plsconvert.converters.universal import universalConverter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def dependency_check():
    self = universalConverter()
    self.checkDependencies()


def cli():
    parser = argparse.ArgumentParser(description="Convert any to any.")
    parser.add_argument(
        "--dependencies", action="store_true", help="Show optional dependencies status"
    )

    parser.add_argument(
        "input_path_pos", nargs="?", help="Input file path (positional)."
    )
    parser.add_argument(
        "output_path_pos", nargs="?", help="Output file path (positional)."
    )

    parser.add_argument("--input", "-i", help="Input file path (named argument).")
    parser.add_argument("--output", "-o", help="Output file path (named argument).")
    args = parser.parse_args()

    if args.dependencies:
        dependency_check()
        sys.exit(0)

    input_file = args.input or args.input_path_pos
    output_file = args.output or args.output_path_pos

    # Enforce mandatory input and output
    if not input_file:
        print(
            "Error: Input file path is required. Use --input or provide it as the first positional argument.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)
    if not output_file:
        output_file = "./"

    input_file = Path(input_file)
    output_file = Path(output_file)

    if input_file.is_dir():
        extension_input = "generic"
    else:
        extension_input = input_file.suffix[1:].lower()

    if output_file.is_dir():
        extension_output = "generic"
    else:
        extension_output = "".join(output_file.suffixes)[1:].lower()

    converter = universalConverter()
    converter.convert(input_file, output_file, extension_input, extension_output)

    print("Conversion completed successfully.")
