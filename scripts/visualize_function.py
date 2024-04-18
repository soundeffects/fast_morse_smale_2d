from argparse import ArgumentParser
from matplotlib.pyplot import subplots, show
from matplotlib.tri import Triangulation
from morse_smale import nn_merged_partition as morse_smale_complex
from numpy import meshgrid, linspace, stack, concatenate
from test_functions import function_map
from triangulate_partitions import triangulate_partition

parser = ArgumentParser(description="visualize the Morse-Smale complex of a predefined two-dimensional function.")
parser.add_argument("function_name", choices=function_map.keys(), help="The function to visualize.")
parser.add_argument("--resolution", default=100, required=False, type=int, help='The resolution with which to evaluate the function on the domain [-1, 1].')

args = parser.parse_args()

x = y = linspace(-1, 1, args.resolution)
sample_grid = [ array.flatten() for array in meshgrid(x, y) ]
sample_points = stack(sample_grid)
sample_x, sample_y = sample_grid
sample_values = function_map[args.function_name](sample_x, sample_y)

msc = morse_smale_complex(sample_points=sample_points, sample_values=sample_values, k_neighbors=8)
print("{} partitions found".format(len(set(msc.partitions))))

_, spatial_axes = subplots(1, 2, subplot_kw={"projection": "3d"})
spatial_axes[0].set_title('Manifold and Extrema Plot')
spatial_axes[0].scatter(sample_x, sample_y, sample_values, c=sample_values, cmap='gray', s=0.95)
spatial_axes[0].scatter(sample_x[msc.max_indices], sample_y[msc.max_indices], sample_values[msc.max_indices], c='red')
spatial_axes[0].scatter(sample_x[msc.min_indices], sample_y[msc.min_indices], sample_values[msc.min_indices], c='c')

spatial_axes[1].set_title('Partition Plot')
spatial_axes[1].scatter(sample_x, sample_y, sample_values, c=msc.partitions)

shape = (args.resolution, args.resolution)
values = sample_values.reshape(shape).T
partitions = msc.partitions.reshape(shape).T
triangulation, vertex_values = triangulate_partition(values, partitions, (-1, 1), (-1, 1))

critical_x = concatenate([sample_x[msc.max_indices], sample_x[msc.min_indices]])
critical_y = concatenate([sample_y[msc.max_indices], sample_y[msc.min_indices]])
critical_value = concatenate([sample_values[msc.max_indices], sample_values[msc.min_indices]])
delaunay = Triangulation(critical_x, critical_y)

_, planar_axes = subplots(1, 2)
planar_axes[0].set_title('My triangulation')
planar_axes[0].tripcolor(triangulation, vertex_values, shading='gouraud', cmap='gray')

planar_axes[1].set_title('Delaunay triangulation')
planar_axes[1].tripcolor(delaunay, critical_value, shading='gouraud', cmap='gray')

show()
