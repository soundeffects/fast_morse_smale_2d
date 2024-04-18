from argparse import ArgumentParser
from csv import reader, writer
from numpy import frombuffer, uint8
from os.path import join
from PIL import Image
from random import randrange, choice, seed

parser = ArgumentParser(description='Take volume data from the repository and take random slices as images')
parser.add_argument('count', type=int, help='The number of image slices to create.')
parser.add_argument('source', help='The directory to read data from. Must have a metadata.csv file.')
parser.add_argument('destination', help='The directory to write the images to.')
parser.add_argument('--seed', required=False, help='The seed for random slice selection. Set to some constant for reproducibility.')
parser.add_argument('--extension', default='png', required=False, help='The image extension to save the slices as.')
parser.add_argument('--size', required=False, type=int, help='Limit slice generation to slices that will not exceed a certain size.')

args = parser.parse_args()

seed(args.seed)

metadata = []
with open(join(args.source, 'metadata.csv'), 'r') as file:
    rows = reader(file)
    next(rows)
    for path, name, x, y, z in rows:
        x, y, z = int(x), int(y), int(z)
        if not args.size or (x <= args.size and y <= args.size and z <= args.size):
            metadata.append((path, name, x, y, z))

data = []
for path, _, x, y, z in metadata:
    with open(join(args.source, path), 'rb') as file:
        bytes = file.read()
        volume = frombuffer(bytes, dtype=uint8).reshape((z, y, x))
        data.append(volume)

with open(join(args.destination, 'metadata.csv'), 'w') as file:
    rows = writer(file)
    rows.writerow(['Path', 'Source', 'Slice Dimension', 'Slice Coordinate'])
    
    for index in range(args.count):
        data_selection = randrange(len(data))
        volume = data[data_selection]
        path, _, x, y, z = metadata[data_selection]
        dimension = choice(['x', 'y', 'z'])
        export_x, export_y, export_z = x, y, z
        slice_coordinate = 0
        image = None

        if dimension == 'x':
            slice_coordinate = randrange(x)
            image = Image.fromarray(volume[:, :, slice_coordinate])
        elif dimension == 'y':
            slice_coordinate = randrange(y)
            image = Image.fromarray(volume[:, slice_coordinate, :])
        else:
            slice_coordinate = randrange(z)
            image = Image.fromarray(volume[slice_coordinate, :, :])

        export_name = '{}.{}'.format(index, args.extension)
        rows.writerow([export_name, path, dimension, slice_coordinate])
        image.save(join(args.destination, export_name))
