#!/usr/bin/env python3
"""
Análisis del Grafo de plsconvert
Script para generar el grafo teórico simplificado en formato JSON.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Import relative modules directly since we're in the plsconvert package

def generate_graph_analysis():
    """Generate simplified theoretical conversion graph analysis"""
    
    try:
        from plsconvert.converters.registry import ConverterRegistry
        
        print("Loading theoretical graph...")
        
        # Get complete theoretical adjacency (all possible conversions)
        theoreticalGraph = ConverterRegistry.theoreticalGraph
        
        # Create simplified structure: format -> [list of target formats]
        simple_graph = {}
        
        for source_format, targets in theoreticalGraph.items():
            # Extract just the target formats (ignore converter names)
            target_formats = [target.output for target in targets]
            # Remove duplicates and sort
            target_formats = sorted(list(set(target_formats)))
            simple_graph[source_format] = target_formats
        
        return simple_graph
        
    except ImportError as e:
        print(f"Error importing plsconvert modules: {e}")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating graph: {e}")
        sys.exit(1)

def save_analysis_to_local():
    """Generate and save the graph analysis to local directory"""
    
    # Generate the simplified graph data
    simple_graph = generate_graph_analysis()
    
    # Create local_data directory if it doesn't exist
    local_data_dir = Path("./local_data")
    local_data_dir.mkdir(exist_ok=True)
    
    # Define output file path
    output_file = local_data_dir / "plsconvert_graph.json"
    
    try:
        # Save to JSON file with nice formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_graph, f, indent=2, ensure_ascii=False)
        
        print(f"Graph analysis completed successfully!")
        print(f"File saved: {output_file}")
        
        # Generate statistics
        total_source_formats = len(simple_graph)
        total_connections = sum(len(targets) for targets in simple_graph.values())
        
        print(f"\nStatistics:")
        print(f"  Total source formats: {total_source_formats}")
        print(f"  Total conversions: {total_connections}")
        
        # Find most versatile formats
        format_connections = [(fmt, len(targets)) for fmt, targets in simple_graph.items()]
        format_connections.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 most versatile formats:")
        for i, (fmt, count) in enumerate(format_connections[:5], 1):
            print(f"  {i}. {fmt}: {count} conversions")
        
        # Show some examples
        print(f"\nConversion examples:")
        examples = ['png', 'pdf', 'mp4', 'docx', 'json']
        for fmt in examples:
            if fmt in simple_graph:
                targets = simple_graph[fmt]
                preview = targets[:5]
                more_text = f" and {len(targets) - 5} more" if len(targets) > 5 else ""
                print(f"  {fmt}: {preview}{more_text}")
        
        # Check for formats with no conversion options
        formats_with_no_conversion_options = []
        for source_format, targets in simple_graph.items():
            for target in targets:
                if target not in simple_graph and target not in formats_with_no_conversion_options:
                    formats_with_no_conversion_options.append(target)

        if formats_with_no_conversion_options:
            print(f"\nFormats with no conversion options: {len(formats_with_no_conversion_options)}")
            print(f"  {formats_with_no_conversion_options}")
        
        print(f"\nJSON file ready for external use")
        print(f"Location: {output_file.absolute()}")
        
    except Exception as e:
        print(f"Error saving analysis file: {e}")
        sys.exit(1)

def main():
    """Main function to run the graph analysis"""
    save_analysis_to_local()

if __name__ == "__main__":
    main() 