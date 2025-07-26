from pathlib import Path
import argparse
import sys

import warnings
import logging

from plsconvert.converters.universal import universalConverter
from plsconvert.converters.registry import ConverterRegistry

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def dependencyCheck():
    converter = universalConverter()
    converter.checkAllDependencies()


def generateGraph(layout: str = 'community'):
    """Generate plsconvert graph using NetworkX. Always generates theoretical graph visualization."""
    from plsconvert.graph_representation import FormatGraphVisualizer
    
    print(f"Generating plsconvert graph with NetworkX (layout: {layout})...")
    
    # Filter to selected formats and show analysis
    print("\nFiltering with selected formats")
    
    graphVisualizer = FormatGraphVisualizer()

    formatsToShow = graphVisualizer.getFormatsToShow()
    filteredGraph = ConverterRegistry.theoreticalGraph.hardFilter(formatsToShow)
    filteredConnections = filteredGraph.getAllConversions()
    
    print("\nFiltered overview:")
    print(f"  Filtered formats: {len(formatsToShow)}")
    print(f"  Filtered connections: {len(filteredConnections)}")
    
    # Generate visualization
    print(f"\nGenerating visualization (layout: {layout})")
    graphVisualizer.visualizeGraph(
        layout=layout,
        savePath='plsconvert_graph.png',
        showConverters=False
    )

def generateGraphInfo():
    """Generate graph analysis and save to local_data directory."""
    from plsconvert.graph_analysis import save_analysis_to_local
    save_analysis_to_local()
        

def cli():
    parser = argparse.ArgumentParser(description="Convert any to any.")
    parser.add_argument(
        "--version", action="store_true", help="Show package version"
    )
    parser.add_argument(
        "--dependencies", "-d", action="store_true", help="Show optional dependencies status"
    )
    parser.add_argument(
        "--graph", nargs='?', const='layout:community', 
        help="Graph operations. Options: 'info' (save JSON to local_data/) or 'layout:TYPE' where TYPE is spring, circular, kamada_kawai, hierarchical, community (default: layout:community)"
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

    # Handle --graph flag
    if args.graph is not None:
        if args.graph == 'info':
            generateGraphInfo()
            sys.exit(0)
        elif args.graph.startswith('layout:'):
            layout = args.graph.split(':', 1)[1] if ':' in args.graph else 'community'
            # Validate layout
            validLayouts = ['spring', 'circular', 'kamada_kawai', 'hierarchical', 'community']
            if layout not in validLayouts:
                print(f"Error: Invalid layout '{layout}'. Valid options: {', '.join(validLayouts)}")
                sys.exit(1)
            generateGraph(layout)
            sys.exit(0)
        else:
            print("Error: Invalid graph option. Use 'info' or 'layout:TYPE'.")
            print("Examples:")
            print("  plsconvert --graph info")
            print("  plsconvert --graph layout:community")
            print("  plsconvert --graph layout:spring")
            sys.exit(1)



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
        # Existing directory
        extension_output = "generic"
    elif str(output_file).endswith(('/', '\\')):
        # Path ends with directory separator - treat as directory
        extension_output = "generic"
    elif not output_file.suffix:
        # No file extension - likely a directory path
        extension_output = "generic"
    else:
        # Has file extension - treat as file
        extension_output = "".join(output_file.suffixes)[1:].lower()

    converter = universalConverter()
    converter.convert(input_file, output_file, extension_input, extension_output)

    print("Conversion completed successfully.")
