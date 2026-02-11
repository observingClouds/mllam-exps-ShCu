import matplotlib.pylab as plt
import numpy as np
import networkx as nx

def create_graph_with_positions(cells, grid):
    """Create a networkx graph with nodes positioned by lat/lon coordinates.
    
    Parameters
    ----------
    cells : array-like
        Cell indices to add as nodes
    grid : xr.Dataset
        ICON grid dataset with clat and clon coordinates
        
    Returns
    -------
    G : nx.Graph
        Graph with nodes
    pos : dict
        Dictionary mapping cell IDs to (lon, lat) positions
    """
    G = nx.Graph()
    pos = {}
    
    for cell in cells:
        G.add_node(cell)
        clat = np.rad2deg(grid.sel(cell=cell).clat.values)
        clon = np.rad2deg(grid.sel(cell=cell).clon.values)
        pos[cell] = (clon, clat)
    
    return G, pos


def add_triangle_edges(G, triangles, connection_pattern='star'):
    """Add edges to graph based on triangle structure.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to add edges to
    triangles : array-like, shape (n, 4)
        Array of cell quadruplets forming triangles
    connection_pattern : str, optional
        'star' - connect cell3 to cell1, cell2, cell4 (default)
        'clique' - connect cell1 to cell2, cell3, cell4
    """
    for cell1, cell2, cell3, cell4 in triangles:
        if connection_pattern == 'star':
            G.add_edge(cell3, cell1)
            G.add_edge(cell3, cell2)
            G.add_edge(cell3, cell4)
        elif connection_pattern == 'clique':
            G.add_edge(cell1, cell2)
            G.add_edge(cell1, cell3)
            G.add_edge(cell1, cell4)
        else:
            raise ValueError(f"Unknown connection pattern: {connection_pattern}")
    
    return G


def coarsen_cells(cells, step=4, offset=2):
    """Extract coarser level cells from a cell array.
    
    Parameters
    ----------
    cells : array-like
        Input cell indices
    step : int, optional
        Stepping interval (default: 4)
    offset : int, optional
        Starting offset (default: 2)
        
    Returns
    -------
    coarse_cells : array
        Coarsened cell indices
    triangles : array, shape (n, 4)
        Cells reshaped into quadruplets
    """
    coarse_cells = cells[offset::step]
    triangles = coarse_cells.reshape(-1, 4)
    return coarse_cells, triangles


def add_neighbor_edges(G, cells, grid):
    """Add edges between neighboring cells in the ICON grid.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to add edges to
    cells : array-like
        Cell indices to process
    grid : xr.Dataset
        ICON grid dataset with neighbor_cell_index
    """
    for cell in cells:
        for neighbor in grid.sel(cell=cell).neighbor_cell_index.values - 1:
            if neighbor in cells and neighbor != cell:
                G.add_edge(cell, neighbor)
    
    return G


def create_multilevel_hierarchy(initial_cells, grid, num_levels=4, 
                                coarsening_configs=None):
    """Create a multi-level graph hierarchy with automatic coarsening.
    
    Parameters
    ----------
    initial_cells : array-like
        Initial finest-level cell indices
    grid : xr.Dataset
        ICON grid dataset with clat, clon, and neighbor_cell_index
    num_levels : int, optional
        Number of coarsening levels to create (default: 4)
    coarsening_configs : list of dict, optional
        List of configuration dicts for each level with keys:
        - 'step': stepping interval (default: 4)
        - 'offset': starting offset (default: 2 for first, 0 for others)
        - 'connection_pattern': 'star' or 'clique' (default: 'star' for first, 'clique' for others)
        - 'name': level name (default: 'Level 0', 'Level 1', etc.)
        
    Returns
    -------
    levels : list of dict
        List of dictionaries, one per level, each containing:
        - 'graph': nx.Graph object
        - 'pos': position dictionary {cell_id: (lon, lat)}
        - 'cells': array of cell indices at this level
        - 'triangles': array of cell quadruplets
        - 'name': level name
        - 'config': configuration used
    """
    if coarsening_configs is None:
        # Default configuration
        coarsening_configs = [
            {'step': 4, 'offset': 2, 'connection_pattern': 'star', 'name': 'Level 0 (1st coarsening)'}
        ]
        for i in range(1, num_levels):
            coarsening_configs.append({
                'step': 4, 'offset': 0 if i > 1 else 2, 
                'connection_pattern': 'clique',
                'name': f'Level {i} ({i+1}{"st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"} coarsening)'
            })
    
    levels = []
    current_cells = initial_cells
    
    for i, config in enumerate(coarsening_configs[:num_levels]):
        step = config.get('step', 4)
        offset = config.get('offset', 2 if i == 0 else 0)
        pattern = config.get('connection_pattern', 'star' if i == 0 else 'clique')
        name = config.get('name', f'Level {i}')
        
        # Coarsen cells
        coarse_cells, triangles = coarsen_cells(current_cells, step=step, offset=offset)
        
        # Create graph with positions
        G, pos = create_graph_with_positions(coarse_cells, grid)
        
        # Add edges
        add_triangle_edges(G, triangles, connection_pattern=pattern)
        
        # Store level information
        level_info = {
            'graph': G,
            'pos': pos,
            'cells': coarse_cells,
            'triangles': triangles,
            'name': name,
            'config': config,
            'level': i
        }
        levels.append(level_info)
        
        # Update current cells for next iteration
        current_cells = coarse_cells
    
    return levels


def create_finest_level(cells, grid, subset_cells, triangles_for_neighbors=None):
    """Create the finest (original) level graph with neighbor connections.
    
    Parameters
    ----------
    cells : array-like
        All cell indices at finest level
    grid : xr.Dataset
        ICON grid dataset
    subset_cells : xr.DataArray
        Subset cells dataset for neighbor lookups
    triangles_for_neighbors : array-like, optional
        Triangles to use for adding neighbor edges. If None, no neighbor edges added.
        
    Returns
    -------
    dict
        Dictionary with 'graph', 'pos', 'cells', 'name' keys
    """
    G, pos = create_graph_with_positions(cells, grid)
    
    if triangles_for_neighbors is not None:
        for cell1, cell2, cell3, cell4 in triangles_for_neighbors:
            for cell in [cell1, cell2, cell3, cell4]:
                for n in subset_cells.sel(cell=cell).neighbor_cell_index.values - 1:
                    if n in cells and n != cell:
                        G.add_edge(cell, n)
    
    return {
        'graph': G,
        'pos': pos,
        'cells': cells,
        'name': 'Original (finest)',
        'level': -1
    }

def visualize_multilevel_graph(graphs_data, figsize=(12, 10), title='Graph of Overlapping Coarse Cells'):
    """Visualize multiple graph levels with different colors and sizes.
    
    Parameters
    ----------
    graphs_data : list of dict
        List of dictionaries with keys: 'graph', 'pos', 'node_size', 'node_color', 
        'edge_width', 'alpha', 'label'
    figsize : tuple, optional
        Figure size (default: (12, 10))
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for data in graphs_data:
        G = data['graph']
        pos = data['pos']
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=data.get('node_size', 100), 
            node_color=data.get('node_color', 'blue'),
            alpha=data.get('node_alpha', 1.0),
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos, 
            alpha=data.get('edge_alpha', 0.5), 
            width=data.get('edge_width', 2),
            ax=ax
        )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax


from scipy.spatial import KDTree

def connect_graph_components(G, pos, k_neighbors=3, max_distance=None):
    """
    Connect disconnected components in a graph using k-nearest neighbors
    to complete a fully connected layer.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to modify (will be modified in-place)
    pos : dict
        Position dictionary mapping node IDs to (lon, lat) tuples
    k_neighbors : int
        Number of nearest neighbors to consider for each node
    max_distance : float, optional
        Maximum distance for adding edges (in degrees). If None, no limit.
    
    Returns
    -------
    G : nx.Graph
        Modified graph with new edges connecting components
    n_edges_added : int
        Number of edges added
    """
    # Get all node positions as array
    nodes = list(G.nodes())
    positions = np.array([pos[node] for node in nodes])
    
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(positions)
    
    edges_added = 0
    
    # For each node, find k nearest neighbors and add edges to nodes in different components
    for i, node in enumerate(nodes):
        # Query k+1 neighbors (including self)
        distances, indices = tree.query(positions[i], k=k_neighbors + 1)
        
        # Skip self (first result)
        for dist, idx in zip(distances[1:], indices[1:]):
            neighbor = nodes[idx]
            
            # Check if edge should be added
            if max_distance is not None and dist > max_distance:
                continue
                
            # Add edge if not already present and nodes are in different components
            if not G.has_edge(node, neighbor):
                # Check if nodes are in different components
                # try:
                #     if not nx.has_path(G, node, neighbor):
                G.add_edge(node, neighbor)
                edges_added += 1
                # except nx.NetworkXError:
                #     # Nodes are in different components
                #     G.add_edge(node, neighbor)
                #     edges_added += 1
    
    return G, edges_added

if __name__ == "__main__":
    import json
    import xarray as xr
    import pickle
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create multi-level graph hierarchy from ICON grid data"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="domain03",
        help="Domain name to process (default: domain03)"
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=4,
        help="Number of coarsening levels to create (default: 4)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../data/multilevel_graph_hierarchy.pkl",
        help="Output path for the hierarchy file (default: ../data/multilevel_graph_hierarchy.pkl)"
    )
    parser.add_argument(
        "--fully-connected",
        action="store_true",
        help="Apply within-level connectivity to fully connect graph components (default: False)"
    )
    
    args = parser.parse_args()
    
    icon_grid = xr.open_dataset("../analysis/grid.nc")
    icon_grid['cell'] = icon_grid.cell.values
    with open("../src/domains_indices.json", "r") as f:
        domains = json.load(f)
    domain_cells = domains[args.domain]
    grid_sel = icon_grid.sel(cell=domain_cells)

    # Create multi-level graph hierarchy
    all_cells = grid_sel.cell.values

    # Create coarsened levels (automatically generates multiple levels)
    coarsened_levels = create_multilevel_hierarchy(
        all_cells, 
        icon_grid, 
        num_levels=args.num_levels
    )

    # Create finest level with neighbor edges
    finest_level = create_finest_level(
        all_cells, 
        icon_grid, 
        grid_sel,
        triangles_for_neighbors=coarsened_levels[0]['triangles']
    )

    # Combine all levels (finest first, then coarsened levels)
    all_levels = [finest_level] + coarsened_levels

    if args.fully_connected:
        all_levels_connectedwithinlevel = all_levels.copy()
        distance_limits = {0: 0.0375, 1: 0.075, 2: 0.0125, 3: 0.025, 4: 0.05, 5: 0.1, 6: 0.2, 7: 0.4, 8: 0.8}
        for l, level in enumerate(all_levels):
            G = level['graph'].copy()
            pos = level['pos']
            G, _ = connect_graph_components(G, pos, k_neighbors=3, max_distance=distance_limits[l])
            all_levels_connectedwithinlevel[l]['graph'] = G.copy()
        all_levels = all_levels_connectedwithinlevel

            

    # Print summary
    print("Multi-level hierarchy created:\n")
    for level in all_levels:
        print(f"{level['name']:25s} | "
            f"Nodes: {level['graph'].number_of_nodes():5d} | "
            f"Edges: {level['graph'].number_of_edges():5d}")

    # Save the multi-level graph hierarchy to disk
    with open(args.output_path, 'wb') as f:
        pickle.dump(all_levels, f)

    print(f"Saved {len(all_levels)} graph levels to {args.output_path}")
