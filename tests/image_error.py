from numpy import meshgrid, sqrt, linspace, stack
from numpy.random import rand
from morse_smale import nn_merged_partition as morse_smale_complex
from matplotlib.pyplot import subplots, show

# Function for a monkey saddle
def monkey_saddle(x, y):
    return (x * -1.5) ** 3 - 3.0 * (x * -1.5) * (y * -1.5) ** 2

def plot_manifold_partitions(manifold_generator, sample_count):
    x = y = linspace(-1, 1, int(sqrt(sample_count)))
    sample_grid = [ array.flatten() for array in meshgrid(x, y) ]
    sample_points = stack(sample_grid)
    sample_x, sample_y = sample_grid
    sample_values = manifold_generator(sample_x, sample_y)
    
    msc = morse_smale_complex(sample_points=sample_points, sample_values=sample_values, k_neighbors=8)

    print("{} partitions found".format(len(set(msc.partitions))))
    _, axes = subplots(1, 2, subplot_kw={"projection": "3d"})
    axes[0].set_title('Manifold and Extrema Plot')
    axes[0].scatter(sample_x, sample_y, sample_values, c=sample_values, cmap='gray', s=0.95)
    axes[0].scatter(sample_x[msc.max_indices], sample_y[msc.max_indices], sample_values[msc.max_indices], c='red')
    axes[0].scatter(sample_x[msc.min_indices], sample_y[msc.min_indices], sample_values[msc.min_indices], c='c')
    
    axes[1].set_title('Partition Plot')
    axes[1].scatter(sample_x, sample_y, sample_values, c=msc.partitions)
    
    show()


plot_manifold_partitions(monkey_saddle, 10000)

"""
sample_positions = mgrid[0:100, 0:100].T.reshape((10000, 2)).astype(float) / 10

msc = MorseSmaleComplex(
    graph=EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0),
    gradient='steepest',
    normalization='feature'
)
msc.build(sample_positions, values.flatten())
complexes = msc.get_partitions(persistence=0.5)

# Display image as height plane
_, ax = subplots(subplot_kw={"projection": "3d"})
surface = ax.plot_surface(x, y, values, cmap="gray", linewidth=0)
show()
"""
