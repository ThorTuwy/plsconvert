from typing import Tuple, Optional, Any
from plsconvert.utils.graph import Graph, Format
from plsconvert.converters.registry import ConverterRegistry

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import networkx as nx
    from netgraph import Graph as NetGraph        
except ImportError:
    raise ImportError("Missing dependencies for graph representation, install with: uv install plsconvert[graph] or if you cloned the repository, run: uv sync --extra graph")


class FormatGraphVisualizer:
    """
    A class to visualize the directed graph of plsconvert.
    """
    
    def __init__(self):
        self.selected_formats = {
            'image': ['jpg', 'png', 'gif', 'pdf', 'ico', 'svg'],
            'video': ['mp4', 'mkv', 'mov'],
            '3d': ['glb', 'gltf', 'obj'],
            'audio': ['mp3', 'wav', 'mid'],
            'document': ['docx', 'doc', 'odt', 'txt', 'html', 'tex', 'pptx', 'csv', 'braille', 'md'],
            'config': ['json', 'toml', 'yaml', 'ini'],
            'compression': ['zip', '7z', 'tar', 'rar'],
            'other': []
        }
        
        self.category_colors = {
            'image': '#FF6B6B',      # Red
            'video': '#4ECDC4',      # Teal
            '3d': '#556B2F',         # Olive
            'audio': '#45B7D1',      # Blue
            'document': '#96CEB4',   # Green
            'config': '#FFEAA7',     # Yellow
            'compression': '#DDA0DD',# Plum
            'other': '#95A5A6'       # Gray
        }
        
        self.format_to_category = {}
        for category, formats in self.selected_formats.items():
            for fmt in formats:
                self.format_to_category[fmt] = category

    def getFormatsToShow(self) -> list[Format]:
        """Get the formats to show."""
        if hasattr(self, '_formatsToShow'):
            return self._formatsToShow
        else:
            self._formatsToShow = [format for formats in self.selected_formats.values() for format in formats]
            return self._formatsToShow
    
    def getFormatCategory(self, formatName: str) -> str:
        """Get the category of a given format."""
        return self.format_to_category.get(formatName, 'other')
    
    def getFormatColor(self, formatName: str) -> str:
        """Get the color for a given format based on its category."""
        category = self.getFormatCategory(formatName)
        return self.category_colors.get(category, self.category_colors['other'])
    
    def createNetworkxGraph(self, graph: Graph, filterSelected: bool = True):
        """
        Create a NetworkX directed graph from the adjacency dictionary.
        """
        formatsToShow = self.getFormatsToShow()

        if filterSelected:
            graph = graph.hardFilter(formatsToShow)
        
        G = nx.DiGraph()
        
        allNodes = set()
        for source, targets in graph.items():
            allNodes.add(source)
            for target, _ in targets:
                allNodes.add(target[1])
        
        for node in allNodes:
            category = self.getFormatCategory(node)
            # Map category to community number for edge bundling
            categoryToCommmunity = {
                'image': 0, 'video': 1, '3d': 2, 'audio': 3, 'document': 4, 
                'config': 5, 'compression': 6, 'other': 7
            }
            community = categoryToCommmunity.get(category, 6)
            G.add_node(node, category=category, color=self.getFormatColor(node), community=community)
        
        # Add edges with converter information
        for source, targets in graph.items():
            for target, converter in targets:
                G.add_edge(source, target[1], converter=converter)

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
        completeGraph = ConverterRegistry.theoreticalGraph

        formatsToShow = self.getFormatsToShow()
        filteredGraph = completeGraph.hardFilter(formatsToShow)
        
        # Filter to selected formats for display
        G = self.createNetworkxGraph(filteredGraph, filterSelected=False)
        
        # Create the plot
        plt.figure(figsize=figsize)

        # Get node colors as dictionary
       
        
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
                node_layout_kwargs={
                    'node_to_community': nodeToCommunity,
                    'pad_by': 0.018
                },
                edge_layout='bundled',
                edge_layout_kwargs={
                    'k': 2000.0,  # Higher k = stronger bundling
                    'compatibility_threshold': 0.02,  # Lower = more bundling
                    'total_cycles': 8,  # More cycles = stronger bundling
                    'total_iterations': 80,  # More iterations = stronger bundling
                    'step_size': 0.06,  # Higher step size = faster convergence
                    'straighten_by': 0.1  # Slight straightening
                },
                node_color=nodeColors,
                node_size=3,
                node_labels=True,
                node_label_fontsize=6,
                edge_color='gray',
                edge_alpha=0.6,
                arrows=True,
                fig=plt.gcf()
            )
            
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
            nx.draw_networkx_nodes(G, pos, node_color=nodeColors, node_size=1000, alpha=0.8) # type: ignore
            
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
        filteredFormats, filteredConversions = len(self.getFormatsToShow()), len(filteredGraph.getAllConversions())
        totalFormats, totalConversions = len(completeGraph.getAllSourceFormats()), len(completeGraph.getAllConversions())
        statsText = f"Showing {filteredFormats}/{totalFormats} formats · {filteredConversions}/{totalConversions} conversions · {len(completeGraph.getAllConverters())} converters"
        plt.figtext(0.5, 0.02, statsText, ha='center', fontsize=10, style='italic', color='gray')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save or show
        if savePath:
            plt.savefig(savePath, dpi=300, bbox_inches='tight')
            print(f"Graph saved to: {savePath}")
        else:
            plt.show()
    
    def analyzeGraphMetrics(self, graph: Graph) -> dict[str, Any]:
        """
        Analyze various metrics of the transformation graph.
        """
        
        G = self.createNetworkxGraph(graph, filterSelected=True)
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
            #debug same as bellow but multiline
            for node in G.nodes():
                print(node)
                print(G.nodes[node])
                if G.nodes[node]['category'] == category:
                    pass
            category_nodes = [node for node in G.nodes() if G.nodes[node]['category'] == category]
            metrics['format_categories'][category] = len(category_nodes)
        
        return metrics
    
    def printGraphMetrics(self, graph: Graph):
        """
        Print detailed graph metrics.
        """
        metrics = self.analyzeGraphMetrics(graph)
        
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
    completeGraph = ConverterRegistry.practicalGraph
    
    # Get all formats and connections
    allFormats, allConnections = completeGraph.getAllUniqueFormats(), completeGraph.getAllConversions()
    
    graphType = "theoretical" if theoretical else "practical"
    print(f"\nSystem overview ({graphType}):")
    print(f"  Total unique formats: {len(allFormats)}")
    print(f"  Total connections: {len(allConnections)}")
    
    converterCounts = completeGraph.numConnectionsPerConverter()
    print(f"  Total converters: {len(converterCounts)}")
    
    totalConnections = sum(converterCounts.values())
    print(f"\nConverter statistics ({graphType}):")
    for converter, count in sorted(converterCounts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {converter}: {count} connections")
        totalConnections += count
    print(f"  Total connections: {totalConnections}")
    
    return completeGraph, allFormats, allConnections
