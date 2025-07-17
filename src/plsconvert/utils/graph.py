# type: ignore[reportPossiblyUnboundVariable]

import copy
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Deque
from plsconvert.utils.files import fileType

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def conversionFromToAdj(
    conversion_from: List[str], conversion_to: List[str]
) -> Dict[str, List[str]]:
    """
    Create a dictionary mapping from conversion_from to conversion_to.
    """
    adj = {}

    for ext in conversion_from:
        adj[ext] = conversion_to

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


def getAllConvertersAdjacency() -> Dict[str, List[List[str]]]:
    """
    Get the complete adjacency dictionary from all available converters.
    This function dynamically imports and collects adjacencies from all converters.
    """
    from plsconvert.converters.universal import universalConverter
    
    # Get universal converter instance
    converter = universalConverter()
    
    # Get the complete adjacency dictionary
    complete_adj = converter.adj
    
    return complete_adj


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
    
    def get_format_category(self, format_name: str) -> str:
        """Get the category of a given format."""
        return self.format_to_category.get(format_name, 'other')
    
    def get_format_color(self, format_name: str) -> str:
        """Get the color for a given format based on its category."""
        category = self.get_format_category(format_name)
        return self.category_colors.get(category, self.category_colors['other'])
    
    def filter_selected_formats(self, adj: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
        """
        Filter the adjacency dictionary to include only selected formats.
        """
        all_selected = set()
        for formats in self.selected_formats.values():
            all_selected.update(formats)
        
        filtered_adj = {}
        for source, targets in adj.items():
            if source in all_selected:
                filtered_targets = []
                for target, converter in targets:
                    if target in all_selected:
                        filtered_targets.append([target, converter])
                if filtered_targets:
                    filtered_adj[source] = filtered_targets
        
        return filtered_adj
    
    def create_networkx_graph(self, adj: Dict[str, List[List[str]]], filter_selected: bool = True):
        """
        Create a NetworkX directed graph from the adjacency dictionary.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Graph visualization requires matplotlib and networkx. Install with: uv add matplotlib networkx")
            
        if filter_selected:
            adj = self.filter_selected_formats(adj)
        
        G = nx.DiGraph()
        
        # Add nodes with their categories
        all_nodes = set()
        for source, targets in adj.items():
            all_nodes.add(source)
            for target, _ in targets:
                all_nodes.add(target)
        
        for node in all_nodes:
            category = self.get_format_category(node)
            G.add_node(node, category=category, color=self.get_format_color(node))
        
        # Add edges with converter information (excluding self-loops)
        for source, targets in adj.items():
            for target, converter in targets:
                if source != target:  # Exclude self-loops (conversions to same format)
                    G.add_edge(source, target, converter=converter)
        
        return G
    
    def visualize_graph(self, adj: Dict[str, List[List[str]]], 
                       filter_selected: bool = True, 
                       layout: str = 'spring',
                       figsize: Tuple[int, int] = (20, 16),
                       save_path: Optional[str] = None,
                       show_converters: bool = False):
        """
        Visualize the plsconvert graph.
        
        Args:
            adj: Adjacency dictionary representing the graph
            filter_selected: Whether to show only selected formats
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'hierarchical')
            figsize: Figure size as (width, height)
            save_path: Path to save the visualization (optional)
            show_converters: Whether to show converter names on edges
        """
        # Get total counts for legend (complete system)
        from plsconvert.converters.universal import universalConverter
        complete_adj = universalConverter().adj
        total_formats, total_connections = getAllFormats(complete_adj)
        total_format_count = len(total_formats)
        total_connection_count = len(total_connections)
        
        G = self.create_networkx_graph(adj, filter_selected)
        
        # Get filtered counts (what we're actually displaying)
        filtered_formats, filtered_connections = getAllFormats(self.filter_selected_formats(adj))
        filtered_format_count = len(filtered_formats)
        filtered_connection_count = len(filtered_connections)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'hierarchical':
            pos = nx.multipartite_layout(G, subset_key='category')
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Get node colors
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', alpha=0.6, width=1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Optionally draw converter names on edges
        if show_converters:
            edge_labels = nx.get_edge_attributes(G, 'converter')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        # Create legend
        legend_elements = []
        for category, color in self.category_colors.items():
            if any(G.nodes[node]['category'] == category for node in G.nodes()):
                legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=category.capitalize()))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set title and remove axes
        title = "plsconvert Graph"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add statistics legend at the bottom
        stats_text = f"Showing {filtered_format_count}/{total_format_count} formats · {filtered_connection_count}/{total_connection_count} conversions"
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic', color='gray')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to: {save_path}")
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
    
    def analyze_graph_metrics(self, adj: Dict[str, List[List[str]]]) -> Dict[str, Any]:
        """
        Analyze various metrics of the transformation graph.
        """
        G = self.create_networkx_graph(adj, filter_selected=True)
        
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
    
    def print_graph_metrics(self, adj: Dict[str, List[List[str]]]):
        """
        Print detailed graph metrics.
        """
        metrics = self.analyze_graph_metrics(adj)
        
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


def print_all_formats_and_connections():
    """
    Print complete information about all formats and connections available in the system.
    """
    print("Complete format information")
    
    # Get complete adjacency from all converters
    print("Loading all converters...")
    complete_adj = getAllConvertersAdjacency()
    
    # Get all formats and connections
    all_formats, all_connections = getAllFormats(complete_adj)
    
    print("\nSystem overview:")
    print(f"  Total unique formats: {len(all_formats)}")
    print(f"  Total connections: {len(all_connections)}")
    
    converter_counts = {}
    for source, target, converter in all_connections:
        if converter not in converter_counts:
            converter_counts[converter] = 0
        converter_counts[converter] += 1
    
    print("\nConverter statistics:")
    for converter, count in sorted(converter_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {converter}: {count} connections")
    
    return complete_adj, all_formats, all_connections


# Convenience function to create and use the visualizer
def visualize_format_graph(adj: Dict[str, List[List[str]]], **kwargs):
    """
    Convenience function to quickly visualize a plsconvert graph.
    """
    visualizer = FormatGraphVisualizer()
    visualizer.visualize_graph(adj, **kwargs)


def analyze_format_graph(adj: Dict[str, List[List[str]]]):
    """
    Convenience function to analyze and print plsconvert graph information.
    Always assumes filtered/selected formats are being analyzed.
    """
    visualizer = FormatGraphVisualizer()
    visualizer.print_graph_metrics(adj)


def main():
    """
    Main function to demonstrate complete graph functionality.
    """
    try:
        # Print complete system information
        complete_adj, all_formats, all_connections = print_all_formats_and_connections()
        
        # Filter with whitelist (selected formats)
        print("\nFiltering with selected formats")
        
        visualizer = FormatGraphVisualizer()
        filtered_adj = visualizer.filter_selected_formats(complete_adj)
        filtered_formats, filtered_connections = getAllFormats(filtered_adj)
        
        print("\nFiltered overview:")
        print(f"  Filtered formats: {len(filtered_formats)}")
        print(f"  Filtered connections: {len(filtered_connections)}")
        
        # Generate analysis
        print("\n")
        analyze_format_graph(filtered_adj)
        
        # Generate visualization
        print("\nGenerating visualization")
        
        visualize_format_graph(
            filtered_adj,
            filter_selected=False,  # Already filtered
            layout='spring',
            save_path='plsconvert_graph.png',
            show_converters=False
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install dependencies with: uv add matplotlib networkx")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
    