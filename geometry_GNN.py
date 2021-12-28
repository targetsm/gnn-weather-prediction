import numpy as np
import itertools



def build_cube_edges(width, height, depth=1, mode="4_neighbors", connect_time_steps=None):
    """
    Returns the edge indices of a cube graph with dimensions 'depth' x 'width' x 'height'

    Args:
        width: The number of columns of the cube graph
        height: The number of rows of the cube graph
        depth: The number of slices of the cube graph
        mode: The following modes decide how the graph vertices will be connected:
            '4_neighbors': connect each vertex to the one above, below, left and right to itself
            '8_neighbors': First connect the same edges like '4_neighbors', then also connect all
                corner edges (above and left, above and right etc...)

    Returns:
        An edge list of the form 'torch.tensor([[v1,v2], [v3, v4], ...])' where 'v1,v2,v3,v4' are simple
        integer values denoting the different vertices. The vertices are numbered row-wise then columnwise.
        Assuming we would only have a 3x3 grid, this would result in the following numbering of the vertices:
                0 1 2
                3 4 5
                6 7 8
    """
    supported_modes = ['4_neighbors', '8_neighbors']
    if mode not in supported_modes:
        raise Exception("Unsupported mode. The only modes currently supported are:" + str(supported_modes))

    def is_valid_edge(edge):
        """
        Args:
            edge: A tuple of the form [(a,b), (c,d)] denoting two vertices v1 = (a,b) and v2 = (c,d)

        Returns:
            True if the edge contains valid vertex coordinates and isn't a self-loop
        """
        # For both vertices, check if they contain invalid vertex coordinates
        for vertex in edge:
            if min(vertex) < 0 or vertex[0] >= height or vertex[1] >= width:
                return False

        # Ensure that the edge isn't a self loop
        return edge[0] != edge[1]

    def map_vertex_to_index(vertex):
        """
        Args:
            vertex: A tuple of the form (a,b) denoting grid coordinates
        Returns:
            A single integer denoting the number of said vertex in the grid
        """
        return vertex[0] * width + vertex[1]

    final_edge_list = []
    number_of_vertices = 0

    for layer in range(depth):
        edge_list = []

        vertex_coordinates = list(itertools.product(range(height), range(width)))
        if mode == '4_neighbors':

            # Connect all vertices to their neighbor above
            above_edges = [((a, b), (a-1, b)) for a, b in vertex_coordinates]

            # Connect all vertices to their neighbor below
            below_edges = [((a, b), (a+1, b)) for a, b in vertex_coordinates]

            # Connect all vertices to their neighbor to the left
            left_edges = [((a, b), (a, b-1)) for a, b in vertex_coordinates]

            # Connect all vertices to their neighbor to the right
            right_edges = [((a, b), (a, b+1)) for a, b in vertex_coordinates]

            edge_list = edge_list + above_edges + below_edges + left_edges + right_edges

        elif mode == '8_neighbors':
            # Creates pairs pointing to all 8 neighbors of a node
            offset_directions = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
            for y, x in offset_directions:
                # Build the new edges
                new_edges = [((a, b), (a + y, b + x)) for a, b in vertex_coordinates]

                # Extend the new edges
                edge_list.extend(new_edges)

        # Filter out invalid edges
        edge_list = [edge for edge in edge_list if is_valid_edge(edge)]

        # Map the grid coordinates to vertex indices
        edge_list = [(map_vertex_to_index(edge[0]), map_vertex_to_index(edge[1])) for edge in edge_list]
        edge_list = np.array(edge_list) + number_of_vertices

        final_edge_list.append(edge_list)

        # Connect each vertex to all other identical vertices in the other time steps
        if connect_time_steps == "all":
            start_vertices = list(np.array(range(width*height)) + number_of_vertices)
            for other_layer in range(depth):
                if other_layer == layer:
                    continue
                end_vertices = list(np.array(range(width*height)) + other_layer * width * height)
                final_edge_list.append(np.array(list(zip(start_vertices, end_vertices))))
        elif connect_time_steps == "nearest":
            start_vertices = list(np.array(range(width*height)) + number_of_vertices)

            if layer > 0:
                end_vertices = list(np.array(range(width*height)) + (layer - 1) * width * height)
                final_edge_list.append(np.array(list(zip(start_vertices, end_vertices))))
            if layer < depth - 1:
                end_vertices = list(np.array(range(width*height)) + (layer + 1) * width * height)
                final_edge_list.append(np.array(list(zip(start_vertices, end_vertices))))

        number_of_vertices += width * height

    return np.concatenate(final_edge_list)

# def build_cube_edges(width, height, depth=1, mode="4_neighbors", connect_time_steps=False):
#     """
#     Returns the edge indices of a cube graph with dimensions 'depth' x 'width' x 'height'

#     Args:
#         width: The number of columns of the cube graph
#         height: The number of rows of the cube graph
#         depth: The number of slices of the cube graph
#         mode: The following modes decide how the graph vertices will be connected:
#             '4_neighbors': connect each vertex to the one above, below, left and right to itself
#             '8_neighbors': First connect the same edges like '4_neighbors', then also connect all
#                 corner edges (above and left, above and right etc...)

#     Returns:
#         An edge list of the form 'torch.tensor([[v1,v2], [v3, v4], ...])' where 'v1,v2,v3,v4' are simple
#         integer values denoting the different vertices. The vertices are numbered row-wise then columnwise.
#         Assuming we would only have a 3x3 grid, this would result in the following numbering of the vertices:
#                 0 1 2
#                 3 4 5
#                 6 7 8
#     """
#     supported_modes = ['4_neighbors', '8_neighbors']
#     if mode not in supported_modes:
#         raise Exception("Unsupported mode. The only modes currently supported are:" + str(supported_modes))

#     def is_valid_edge(edge):
#         """
#         Args:
#             edge: A tuple of the form [(a,b), (c,d)] denoting two vertices v1 = (a,b) and v2 = (c,d)

#         Returns:
#             True if the edge contains valid vertex coordinates and isn't a self-loop
#         """
#         # For both vertices, check if they contain invalid vertex coordinates
#         for vertex in edge:
#             if min(vertex) < 0 or vertex[0] >= height or vertex[1] >= width:
#                 return False

#         # Ensure that the edge isn't a self loop
#         return edge[0] != edge[1]

#     def map_vertex_to_index(vertex):
#         """
#         Args:
#             vertex: A tuple of the form (a,b) denoting grid coordinates
#         Returns:
#             A single integer denoting the number of said vertex in the grid
#         """
#         return vertex[0] * width + vertex[1]

#     final_edge_list = []
#     number_of_vertices = 0

#     for layer in range(depth):
#         edge_list = []

#         vertex_coordinates = list(itertools.product(range(height), range(width)))
#         if mode == '4_neighbors':

#             # Connect all vertices to their neighbor above
#             above_edges = [((a, b), (a-1, b)) for a, b in vertex_coordinates]

#             # Connect all vertices to their neighbor below
#             below_edges = [((a, b), (a+1, b)) for a, b in vertex_coordinates]

#             # Connect all vertices to their neighbor to the left
#             left_edges = [((a, b), (a, b-1)) for a, b in vertex_coordinates]

#             # Connect all vertices to their neighbor to the right
#             right_edges = [((a, b), (a, b+1)) for a, b in vertex_coordinates]

#             edge_list = edge_list + above_edges + below_edges + left_edges + right_edges

#         elif mode == '8_neighbors':
#             # Creates pairs pointing to all 8 neighbors of a node
#             offset_directions = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
#             for y, x in offset_directions:
#                 # Build the new edges
#                 new_edges = [((a, b), (a + y, b + x)) for a, b in vertex_coordinates]

#                 # Extend the new edges
#                 edge_list.extend(new_edges)

#         # Filter out invalid edges
#         edge_list = [edge for edge in edge_list if is_valid_edge(edge)]

#         # Map the grid coordinates to vertex indices
#         edge_list = [(map_vertex_to_index(edge[0]), map_vertex_to_index(edge[1])) for edge in edge_list]
#         edge_list = np.array(edge_list) + number_of_vertices

#         final_edge_list.append(edge_list)

#         # Connect each vertex to all other identical vertices in the other time steps
#         if connect_time_steps:
#             start_vertices = list(np.array(range(width*height)) + number_of_vertices)
#             for other_layer in range(depth):
#                 if other_layer == layer:
#                     continue
#                 end_vertices = list(np.array(range(width*height)) + other_layer * width * height)
#                 final_edge_list.append(np.array(list(zip(start_vertices, end_vertices))))

#         number_of_vertices += width * height

#     return np.concatenate(final_edge_list)