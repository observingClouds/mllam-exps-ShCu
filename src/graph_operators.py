import operator

def find_local_extrema(G, attribute='value', comparison_func=operator.lt):
    """
    Find local extrema (minima or maxima) in a graph based on a node attribute.
    
    A node is a local extremum if it satisfies the comparison function
    against all of its neighbors' values.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph with nodes containing the specified attribute
    attribute : str, optional
        Name of the node attribute to use for comparison (default: 'value')
    comparison_func : callable, optional
        Comparison function to use. Default is operator.lt (less than) for minima.
        Use operator.gt (greater than) for maxima.
        Function should take (node_value, neighbor_value) and return bool.
    
    Returns
    -------
    extrema : list
        List of nodes that are local extrema
        
    Examples
    --------
    >>> # Find local minima
    >>> minima = find_local_extrema(G, 't_2m', operator.lt)
    >>> # Find local maxima
    >>> maxima = find_local_extrema(G, 't_2m', operator.gt)
    """
    extrema = []
    
    for node in G.nodes():
        node_value = G.nodes[node].get(attribute)
        
        # Skip nodes without the attribute
        if node_value is None:
            continue
            
        neighbors = list(G.neighbors(node))
        
        # Skip isolated nodes
        if not neighbors:
            continue
        
        # Check if current node satisfies the comparison against all neighbors
        is_extremum = all(
            comparison_func(node_value, G.nodes[neighbor].get(attribute, float('inf'))) 
            for neighbor in neighbors
        )
        
        if is_extremum:
            extrema.append(node)
    
    return extrema


# Convenience functions for common use cases
def find_local_minima(G, attribute='value'):
    """Find local minima. Shorthand for find_local_extrema with operator.lt"""
    return find_local_extrema(G, attribute, operator.lt)


def find_local_maxima(G, attribute='value'):
    """Find local maxima. Shorthand for find_local_extrema with operator.gt"""
    return find_local_extrema(G, attribute, operator.gt)