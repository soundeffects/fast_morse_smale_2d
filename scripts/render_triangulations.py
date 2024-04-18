from argparse import ArgumentParser
from csv import reader, writer
from matplotlib.image import imread
from matplotlib.pyplot import savefig, axis, figure, clf, subplots_adjust, tripcolor
from matplotlib.tri import Triangulation
from morse_smale import nn_merged_partition as morse_smale_complex
from numpy import ndindex, stack, meshgrid, double, concatenate, linspace
from os.path import join
from triangulate_partitions import triangulate_partition

parser = ArgumentParser(description='Render triangulations to a bitmap image, for image error comparison.')
parser.add_argument('source', help='The directory to get triangulations from. Must have a metadata.csv file.')
parser.add_argument('destination', help='The directory to save triangulation renders to.')
parser.add_argument('--delaunay', action='store_true', help='If enabled, it will render delaunay triangulations as well.')

args = parser.parse_args()

image_paths = []
with open(join(args.source, 'metadata.csv'), 'r') as file:
    rows = reader(file)
    next(rows)
    for path, _, _, _ in rows:
        image_paths.append(path)

fig = figure()
axis('off')
subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-1.05)

with open(join(args.destination, 'metadata.csv'), 'w') as file:
    rows = writer(file)
    if args.delaunay:
        rows.writerow(["Source", "Successful", "Path", "Triangulation Size", "Delaunay Path", "Delaunay Triangulation Size"])
    else:
        rows.writerow(["Source", "Successful", "Path", "Compression"])

    for index, path in enumerate(image_paths):
        image = imread(join(args.source, path))
        print("{}/{}".format(index, len(image_paths)), image.shape)
        
        x = linspace(0, image.shape[0] - 1, image.shape[0])
        y = linspace(0, image.shape[1] - 1, image.shape[1])
        sample_grid = [ array.flatten() for array in meshgrid(x, y) ]
        sample_points = stack(sample_grid)
        sample_values = image.flatten().astype(double)
        
        msc = morse_smale_complex(sample_points=sample_points, sample_values=sample_values, k_neighbors=8)
        partition_count = len(set(msc.partitions))
        if partition_count > (image.shape[0] * image.shape[1] / 4):
            print("too many partitions, skipping")
            if args.delaunay:
                rows.writerow([path, False, None, None, None, None])
            else:
                rows.writerow([path, False, None, None])
            continue
        
        partitions = msc.partitions.reshape(image.shape).T
        triangulation, vertex_values = triangulate_partition(image, partitions, (0, image.shape[0]), (0, image.shape[1]))

        triangulation_size = (len(triangulation.triangles) * 3 * 4) + (len(vertex_values) * 4)


        clf()
        tripcolor(triangulation, vertex_values, shading='gouraud', cmap='gray')
        fig.set_size_inches((image.shape[0] / 100., image.shape[1] / 100.))            
        
        new_path = path + '.png'
        savefig(join(args.destination, new_path), dpi=100)
        
        if args.delaunay:
            sample_x, sample_y = sample_grid
            critical_x = concatenate([sample_x[msc.max_indices], sample_x[msc.min_indices]])
            critical_y = concatenate([sample_y[msc.max_indices], sample_y[msc.min_indices]])
            triangulation = Triangulation(critical_x, critical_y)
            vertex_values = concatenate([sample_values[msc.max_indices], sample_values[msc.min_indices]])

            delaunay_triangulation_size = (len(triangulation.triangles) * 3 * 4) + (len(vertex_values) * 4)

            clf()
            tripcolor(triangulation, vertex_values, shading='gouraud', cmap='gray')

            delaunay_path = path + '_delaunay.png'
            savefig(join(args.destination, delaunay_path), dpi=100)

            rows.writerow([path, True, new_path, triangulation_size, delaunay_path, delaunay_triangulation_size])
        else:
            rows.writerow([path, True, new_path, triangulation_size])

