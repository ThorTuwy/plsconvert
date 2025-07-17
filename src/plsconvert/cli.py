from pathlib import Path
import argparse
import sys

import warnings
import logging

from plsconvert.converters.universal import universalConverter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def dependencyCheck():
    self = universalConverter()
    self.checkDependencies()


def generateGraph(layout: str = 'community'):
    """Generate plsconvert graph using NetworkX. Always generates theoretical graph visualization."""
    from plsconvert.graph_representation import visualizeFormatGraph, printAllFormatsAndConnections, analyzeFormatGraph, FormatGraphVisualizer, getAllFormats
    
    print(f"Generating plsconvert graph with NetworkX (layout: {layout})...")
    # Print complete theoretical system information
    completeAdj, allFormats, allConnections = printAllFormatsAndConnections(theoretical=True)
    
    # Filter to selected formats and show analysis
    print("\nFiltering with selected formats")
    
    visualizer = FormatGraphVisualizer()
    filteredAdj = visualizer.filterSelectedFormats(completeAdj)
    filteredFormats, filteredConnections = getAllFormats(filteredAdj)
    
    print("\nFiltered overview:")
    print(f"  Filtered formats: {len(filteredFormats)}")
    print(f"  Filtered connections: {len(filteredConnections)}")
    
    # Generate analysis
    print()
    analyzeFormatGraph(filteredAdj)
    
    # Generate visualization
    print(f"\nGenerating visualization (layout: {layout})")
    visualizeFormatGraph(
        layout=layout,
        savePath='plsconvert_graph.png',
        showConverters=False
    )
        

def cli():
    parser = argparse.ArgumentParser(description="Convert any to any.")
    parser.add_argument(
        "--version", action="store_true", help="Show package version"
    )
    parser.add_argument(
        "--dependencies", "-d", action="store_true", help="Show optional dependencies status"
    )
    parser.add_argument(
        "--graph", nargs='?', const='community', 
        help="Generate plsconvert graph visualization. Optional layout: community (default, with edge bundling), spring, circular, kamada_kawai, hierarchical"
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

    if args.version:
        try:
            import importlib.metadata
            version = importlib.metadata.version("plsconvert")
        except Exception:
            version = "unknown"
        print(f"plsconvert version: {version}")
        sys.exit(0)

    if args.dependencies:
        dependencyCheck()
        sys.exit(0)

    if args.graph is not None:
        # Check dependencies

        # Validate layout
        validLayouts = ['spring', 'circular', 'kamada_kawai', 'hierarchical', 'community']
        if args.graph not in validLayouts:
            print(f"Error: Invalid layout '{args.graph}'. Valid options: {', '.join(validLayouts)}")
            sys.exit(1)
        
        generateGraph(args.graph)
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
