# type: ignore[reportPossiblyUnboundVariable]

import copy
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Deque
from plsconvert.utils.files import fileType

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import networkx as nx
    from netgraph import Graph as NetGraph
    VISUALIZATION_AVAILABLE = True
        
except ImportError:
    VISUALIZATION_AVAILABLE = False


def conversionFromToAdj(
    conversionFrom: List[str], conversionTo: List[str]
) -> Dict[str, List[str]]:
    """
    Create a dictionary mapping from conversionFrom to conversionTo.
    """
    adj = {}

    for ext in conversionFrom:
        adj[ext] = conversionTo

    return adj


def mergeAdj(adj1: Dict[str, List[str]], adj2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge two adjacency dictionaries.
    """
    for key, value in adj2.items():
        if key not in adj1:
            adj1[key] = copy.deepcopy(value)
        else:
            adj1[key].extend(value)

    return adj1


def bfs(start: str, end: str, adj: Dict[str, List[List[str]]]) -> List[List[str]]:
    visited = []
    queue: Deque[Tuple[str, List[List[str]]]] = deque([(start, [])])

    while queue:
        current, path = queue.popleft()

        if current == end:
            return path
        visited.append(current)

        # Never do things after audio=>video
        if (
            len(path) == 1
            and fileType(start) == "audio"
            and fileType(path[0][0]) == "video"
        ):
            continue

        for neighbor, converter in adj.get(current, []):
            if neighbor not in visited:
                path_copy = path.copy()
                path_copy.append([neighbor, converter])
                queue.append((neighbor, path_copy))

    return []


def getAllConvertersAdjacency(theoretical: bool = False) -> Dict[str, List[List[str]]]:
    """
    Get the adjacency dictionary from all converters.
    
    Args:
        theoretical: If True, returns complete theoretical graph (all converters).
                    If False, returns practical graph (only available converters).
    """
    from plsconvert.converters.universal import universalConverter
    
    # Get universal converter instance
    converter = universalConverter()
    
    # Get the adjacency dictionary based on type requested
    completeAdj = converter.getAdjacency(theoretical=theoretical)
    
    return completeAdj


def getAllFormats(adj: Dict[str, List[List[str]]]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract all unique formats (nodes) and connections (edges) from adjacency dictionary.
    
    Returns:
        Tuple containing:
        - List of all unique format nodes
        - List of connections as (source, target, converter) tuples
    """
    all_formats = set()
    all_connections = []
    
    for source, targets in adj.items():
        all_formats.add(source)
        for target, converter in targets:
            all_formats.add(target)
            all_connections.append((source, target, converter))
    
    return sorted(list(all_formats)), all_connections


def filterFormatsWithWhitelist(
    adj: Dict[str, List[List[str]]], 
    whitelist: Optional[List[str]] = None
) -> Dict[str, List[List[str]]]:
    """
    Filter adjacency dictionary to include only formats in the whitelist.
    If no whitelist is provided, uses the selected formats from FormatGraphVisualizer.
    """
    if whitelist is None:
        # Use selected formats as default whitelist
        visualizer = FormatGraphVisualizer()
        whitelist_set = set()
        for formats in visualizer.selected_formats.values():
            whitelist_set.update(formats)
    else:
        whitelist_set = set(whitelist)
    
    filtered_adj = {}
    for source, targets in adj.items():
        if source in whitelist_set:
            filtered_targets = []
            for target, converter in targets:
                if target in whitelist_set:
                    filtered_targets.append([target, converter])
            if filtered_targets:
                filtered_adj[source] = filtered_targets
    
    return filtered_adj


class FormatGraphVisualizer:
    """
    A class to visualize the directed graph of plsconvert.
    """
    
    def __init__(self):
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Graph visualization requires matplotlib and networkx. Install with: uv add matplotlib networkx")
        
        # Define selected formats by category
        self.selected_formats = {
            'image': ['jpg', 'png', 'gif', 'pdf', 'ico'],
            'video': ['mp4', 'mkv', 'mov'],
            'audio': ['mp3', 'wav', 'mid'],
            'document': ['docx', 'doc', 'odt', 'txt', 'html', 'tex', 'pptx', 'csv'],
            'config': ['json', 'toml', 'yaml', 'ini'],
            'compression': ['zip', '7z', 'tar', 'rar'],
            'other': []
        }
        
        # Define colors for each category
        self.category_colors = {
            'image': '#FF6B6B',      # Red
            'video': '#4ECDC4',      # Teal
            'audio': '#45B7D1',      # Blue
            'document': '#96CEB4',   # Green
            'config': '#FFEAA7',     # Yellow
            'compression': '#DDA0DD', # Plum
            'other': '#95A5A6'       # Gray
        }
        
        # Create reverse mapping from format to category
        self.format_to_category = {}
        for category, formats in self.selected_formats.items():
            for fmt in formats:
                self.format_to_category[fmt] = category
    
    def getFormatCategory(self, formatName: str) -> str:
        """Get the category of a given format."""
        return self.format_to_category.get(formatName, 'other')
    
    def getFormatColor(self, formatName: str) -> str:
        """Get the color for a given format based on its category."""
        category = self.getFormatCategory(formatName)
        return self.category_colors.get(category, self.category_colors['other'])
    
    def filterSelectedFormats(self, adj: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
        """
        Filter the adjacency dictionary to include only selected formats.
        """
        allSelected = set()
        for formats in self.selected_formats.values():
            allSelected.update(formats)
        
        filteredAdj = {}
        for source, targets in adj.items():
            if source in allSelected:
                filteredTargets = []
                for target, converter in targets:
                    if target in allSelected:
                        filteredTargets.append([target, converter])
                if filteredTargets:
                    filteredAdj[source] = filteredTargets
        
        return filteredAdj
    
    def createNetworkxGraph(self, adj: Dict[str, List[List[str]]], filterSelected: bool = True):
        """
        Create a NetworkX directed graph from the adjacency dictionary.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Graph visualization requires matplotlib and networkx. Install with: uv add matplotlib networkx")
            
        if filterSelected:
            adj = self.filterSelectedFormats(adj)
        
        G = nx.DiGraph()
        
        # Add nodes with their categories
        allNodes = set()
        for source, targets in adj.items():
            allNodes.add(source)
            for target, _ in targets:
                allNodes.add(target)
        
        for node in allNodes:
            category = self.getFormatCategory(node)
            # Map category to community number for edge bundling
            categoryToCommmunity = {
                'image': 0, 'video': 1, 'audio': 2, 'document': 3, 
                'config': 4, 'compression': 5, 'other': 6
            }
            community = categoryToCommmunity.get(category, 6)
            G.add_node(node, category=category, color=self.getFormatColor(node), community=community)
        
        # Add edges with converter information (excluding self-loops)
        for source, targets in adj.items():
            for target, converter in targets:
                if source != target:  # Exclude self-loops (conversions to same format)
                    G.add_edge(source, target, converter=converter)
        
        return G
    
    def visualizeGraph(self, 
                       layout: str = 'spring',
                       figsize: Tuple[int, int] = (20, 16),
                       savePath: Optional[str] = None,
                       showConverters: bool = False):
        """
        Visualize the plsconvert graph. Always uses theoretical complete graph for visualization.
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'hierarchical')
            figsize: Figure size as (width, height)
            savePath: Path to save the visualization (optional)
            showConverters: Whether to show converter names on edges
        """
        # Always use theoretical complete system for visualization
        completeAdj = getAllConvertersAdjacency(theoretical=True)
        totalFormats, totalConnections = getAllFormats(completeAdj)
        totalFormatCount = len(totalFormats)
        totalConnectionCount = len(totalConnections)
        
        # Filter to selected formats for display
        filteredAdj = self.filterSelectedFormats(completeAdj)
        G = self.createNetworkxGraph(filteredAdj, filterSelected=False)
        
        # Get filtered counts (what we're actually displaying)
        filteredFormats, filteredConnections = getAllFormats(filteredAdj)
        filteredFormatCount = len(filteredFormats)
        filteredConnectionCount = len(filteredConnections)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        if layout == 'community':
            # Use netgraph with edge bundling only for community layout
            print("Using community layout with edge bundling...")
            
            # Get node colors as dictionary
            nodeColors = {node: G.nodes[node]['color'] for node in G.nodes()}
            
            # Create node to community mapping for community layout
            nodeToCommunity = {node: G.nodes[node]['community'] for node in G.nodes()}
            
            # Create netgraph visualization with edge bundling
            NetGraph(
                G, 
                node_layout='community',
                node_layout_kwargs={'node_to_community': nodeToCommunity},
                edge_layout='bundled',
                node_color=nodeColors,
                node_size=3,
                node_labels=True,
                node_label_fontsize=6,
                edge_color='gray',
                edge_alpha=0.6,
                arrows=True,
                fig=plt.gcf()
            )
            
            # Create manual legend for categories since netgraph doesn't include it automatically
            legendElements = []
            for category, color in self.category_colors.items():
                if any(G.nodes[node]['category'] == category for node in G.nodes()):
                    legendElements.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=10, 
                                                    label=category.capitalize()))
            
            plt.legend(handles=legendElements, loc='upper left', bbox_to_anchor=(1, 1))
            
        else:
            # Use traditional NetworkX visualization
            print("Using traditional NetworkX visualization...")
            
            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(G, k=3, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                # Increase spacing to prevent node overlap
                pos = nx.kamada_kawai_layout(G, scale=2, pos=None)
            elif layout == 'hierarchical':
                pos = nx.multipartite_layout(G, subset_key='category')
            else:
                pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Get node colors
            nodeColors = [G.nodes[node]['color'] for node in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=nodeColors, node_size=1000, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                  arrowsize=20, arrowstyle='->', alpha=0.6, width=1)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            # Optionally draw converter names on edges
            if showConverters:
                edgeLabels = nx.get_edge_attributes(G, 'converter')
                nx.draw_networkx_edge_labels(G, pos, edgeLabels, font_size=6)
        
        # Create legend
        legendElements = []
        for category, color in self.category_colors.items():
            if any(G.nodes[node]['category'] == category for node in G.nodes()):
                legendElements.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=category.capitalize()))
        
        plt.legend(handles=legendElements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set title and remove axes
        title = "plsconvert Graph"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add statistics legend at the bottom
        statsText = f"Showing {filteredFormatCount}/{totalFormatCount} formats · {filteredConnectionCount}/{totalConnectionCount} conversions"
        plt.figtext(0.5, 0.02, statsText, ha='center', fontsize=10, style='italic', color='gray')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save or show
        if savePath:
            plt.savefig(savePath, dpi=300, bbox_inches='tight')
            print(f"Graph saved to: {savePath}")
        else:
            plt.show()
    
    def generate_format_summary(self, adj: Dict[str, List[List[str]]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate a summary of formats and their conversion capabilities.
        """
        summary = {}
        
        for category, formats in self.selected_formats.items():
            summary[category] = {
                'formats': formats,
                'color': self.category_colors[category],
                'conversions': {}
            }
            
            for fmt in formats:
                if fmt in adj:
                    targets = [target for target, _ in adj[fmt]]
                    summary[category]['conversions'][fmt] = {
                        'can_convert_to': targets,
                        'converter_count': len(targets)
                    }
        
        return summary
    
    def print_format_summary(self, adj: Dict[str, List[List[str]]]):
        """
        Print a formatted summary of formats.
        """
        # This method is now empty as the summary has been removed
        pass
    
    def analyzeGraphMetrics(self, adj: Dict[str, List[List[str]]]) -> Dict[str, Any]:
        """
        Analyze various metrics of the transformation graph.
        """
        G = self.createNetworkxGraph(adj, filterSelected=True)
        
        metrics = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'number_of_components': nx.number_weakly_connected_components(G),
            'average_clustering': nx.average_clustering(G.to_undirected()),
            'most_connected_formats': {},
            'format_categories': {}
        }
        
        # Find most connected formats
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        metrics['most_connected_formats'] = {
            'highest_in_degree': sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5],
            'highest_out_degree': sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        # Count formats by category
        for category in self.selected_formats:
            category_nodes = [node for node in G.nodes() if G.nodes[node]['category'] == category]
            metrics['format_categories'][category] = len(category_nodes)
        
        return metrics
    
    def printGraphMetrics(self, adj: Dict[str, List[List[str]]]):
        """
        Print detailed graph metrics.
        """
        metrics = self.analyzeGraphMetrics(adj)
        
        print("Graph metrics")
        
        print(f"Total formats (nodes): {metrics['total_nodes']}")
        print(f"Total conversions (edges): {metrics['total_edges']}")
        print(f"Graph density: {metrics['density']:.3f}")
        print(f"Is connected: {metrics['is_connected']}")
        print(f"Number of components: {metrics['number_of_components']}")
        print(f"Average clustering coefficient: {metrics['average_clustering']:.3f}")
        

        
        print("\nFormats by category:")
        for category, count in metrics['format_categories'].items():
            print(f"  {category}: {count} formats")


def printAllFormatsAndConnections(theoretical: bool = False):
    """
    Print complete information about all formats and connections available in the system.
    
    Args:
        theoretical: If True, shows complete theoretical capabilities.
                    If False, shows only practical capabilities with available dependencies.
    """
    if theoretical:
        print("Complete theoretical format information (all possible conversions)")
    else:
        print("Practical format information (available conversions)")
    
    # Get adjacency from all converters
    print("Loading all converters...")
    completeAdj = getAllConvertersAdjacency(theoretical=theoretical)
    
    # Get all formats and connections
    allFormats, allConnections = getAllFormats(completeAdj)
    
    graphType = "theoretical" if theoretical else "practical"
    print(f"\nSystem overview ({graphType}):")
    print(f"  Total unique formats: {len(allFormats)}")
    print(f"  Total connections: {len(allConnections)}")
    
    converterCounts = {}
    for source, target, converter in allConnections:
        if converter not in converterCounts:
            converterCounts[converter] = 0
        converterCounts[converter] += 1
    
    print(f"\nConverter statistics ({graphType}):")
    for converter, count in sorted(converterCounts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {converter}: {count} connections")
    
    return completeAdj, allFormats, allConnections


# Convenience function to create and use the visualizer
def visualizeFormatGraph(**kwargs):
    """
    Convenience function to quickly visualize a plsconvert graph. Always uses theoretical graph.
    """
    visualizer = FormatGraphVisualizer()
    visualizer.visualizeGraph(**kwargs)


def analyzeFormatGraph(adj: Dict[str, List[List[str]]]):
    """
    Convenience function to analyze and print plsconvert graph information.
    Always assumes filtered/selected formats are being analyzed.
    """
    visualizer = FormatGraphVisualizer()
    visualizer.printGraphMetrics(adj)


def main(theoretical: bool = False):
    """
    Main function to demonstrate complete graph functionality.
    
    Args:
        theoretical: If True, generates theoretical graph. If False, generates practical graph.
    """
    try:
        # Print system information for the requested graph type
        completeAdj, allFormats, allConnections = printAllFormatsAndConnections(theoretical=theoretical)
        
        # Filter with whitelist (selected formats)
        print("\nFiltering with selected formats")
        
        visualizer = FormatGraphVisualizer()
        filteredAdj = visualizer.filterSelectedFormats(completeAdj)
        filteredFormats, filteredConnections = getAllFormats(filteredAdj)
        
        print("\nFiltered overview:")
        print(f"  Filtered formats: {len(filteredFormats)}")
        print(f"  Filtered connections: {len(filteredConnections)}")
        
        # Generate analysis
        print("\n")
        analyzeFormatGraph(filteredAdj)
        
        # Generate visualization
        print("\nGenerating visualization")
        
        graphSuffix = "_theoretical" if theoretical else "_practical"
        savePath = f'plsconvert_graph{graphSuffix}.png'
        
        visualizeFormatGraph(
            layout='spring',
            savePath=savePath,
            showConverters=False
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install dependencies with: uv add matplotlib networkx")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(theoretical=False)  # Default to practical graph for backwards compatibility
    