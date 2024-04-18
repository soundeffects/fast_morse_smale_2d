from matplotlib.tri import Triangulation
from numpy import ndindex, arctan2
from scipy.cluster.hierarchy import DisjointSet

def rescale_range(value, old_range, new_range):
    normalized = (value - old_range[0]) / (old_range[1] - old_range[0])
    return normalized * (new_range[1] - new_range[0]) + new_range[0]

def triangulate_partition(values, partitions, range_x, range_y):
    shape = values.shape
    partition_count = len(set(partitions.flatten()))
    corner_groups = DisjointSet()

    x_list = []
    y_list = []
    value_list = []
    face_indices = [ [] for _ in range(partition_count)]
    triangles = []

    adjacent_partitions = set()
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Find corners, group adjacent corners
    for x, y in ndindex(shape):
        adjacent_partitions.clear()

        for offset_x, offset_y in offsets:
            new_x, new_y = x + offset_x, y + offset_y
            if new_x < 0 or new_x > shape[0] -1:
                adjacent_partitions.add(-1)
            elif new_y < 0 or new_y > shape[1] - 1:
                adjacent_partitions.add(-2)
            else:
                adjacent_partitions.add(partitions[new_x, new_y])

        if len(adjacent_partitions) > 2:
            position = (x, y)
            corner_groups.add(position)
            for offset_x, offset_y in offsets:
                adjacent = (x + offset_x, y + offset_y)
                if adjacent in corner_groups:
                    #if len(corner_groups.subset(adjacent)) < 12:
                    contribution_constraint = len(list(filter(lambda c: partitions[*c] == partitions[x, y], corner_groups.subset(adjacent)))) < 1
                    if contribution_constraint:
                        corner_groups.merge(position, adjacent)

    
    # Merge corner groups into single vertices and create faces for vertices
    for index, corner_group in enumerate(corner_groups.subsets()):
        x, y, value, faces = 0, 0, 0, set()
        for corner_x, corner_y in corner_group:
            x += corner_x
            y += corner_y
            value += values[corner_x, corner_y]
            faces.add(partitions[corner_x, corner_y])

        x = rescale_range(x / len(corner_group), (0, shape[0] - 1), range_x)
        y = rescale_range(y / len(corner_group), (0, shape[1] - 1), range_y)
        value /= len(corner_group)
        x_list.append(x)
        y_list.append(y)
        value_list.append(value)

        for face in faces:
            face_indices[face].append(index)

    face_warning = False
    # Convert faces into triangles
    for face in face_indices:
        if len(face) < 3 and not face_warning:
            face_warning = True
            print('WARNING: at least one face without at least three vertices')
            continue

        center_x = sum(map(lambda index: x_list[index], face)) / len(face)
        center_y = sum(map(lambda index: y_list[index], face)) / len(face)
        
        # Indices must be ordered in counter-clockwise order
        sorted_indices = sorted(face, key=lambda index: arctan2(y_list[index] - center_y, x_list[index] - center_x))
        maximum_in_sorted = sorted_indices.index(max(face, key=lambda index: value_list[index]))

        for offset in range(len(face) - 2):
            triangles.append([
                sorted_indices[maximum_in_sorted],
                sorted_indices[(maximum_in_sorted + 1 + offset) % len(sorted_indices)],
                sorted_indices[(maximum_in_sorted + 2 + offset) % len(sorted_indices)]
            ])

    return Triangulation(x_list, y_list, triangles=triangles), value_list
