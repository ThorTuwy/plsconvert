#!/usr/bin/env python3
"""
Debug runner for plsconvert
This script allows proper debugging by running the command directly with dynamic arguments
"""

import sys
import os

def main():
    """Run plsconvert with debugging support"""
    
    # Add the src directory to Python path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Import the CLI function
    from plsconvert.cli import cli
    
    # Use the original sys.argv (arguments passed from launch.json)
    # This allows us to pass any arguments like --graph, -d, etc.
    # The first argument (script name) will be replaced with 'plsconvert'
    if len(sys.argv) > 1:
        # Replace the script name with 'plsconvert' and keep all other arguments
        sys.argv = ['plsconvert'] + sys.argv[1:]
    else:
        # Default to --graph if no arguments provided
        sys.argv = ['plsconvert', '--graph']
    
    # Run the CLI function
    cli()

if __name__ == "__main__":
    main() 