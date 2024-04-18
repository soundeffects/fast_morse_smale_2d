from argparse import ArgumentParser
from csv import reader, writer
from numpy import frombuffer, uint8
from os.path import join
from PIL import Image
from random import randrange, choice, seed

parser = ArgumentParser(description='Take set of images and report compression ratios for Morse-Smale representation of the scalar fields that the images represent.')
parser.add_argument('source', help='The directory to read data from. Must have a metadata.csv file.')
parser.add_argument('k_neighbors', type=int, help='How many neigbors to check for each pixel location as we compute the Morse-Smale complex.')

args = parser.parse_args()

worst_compression_ratios = []
worst_compression_images = []
best_compression_ratios = []
best_compression_images = []
with open(join(args.source, 'metadata.csv'), 'r') as file:
    rows = reader(file)
    next(rows)
    for path, _, _, _ in rows:
        with Image.open(join(args.source, path)) as image:
            
        metadata.append((path, name, int(x), int(y), int(z)))

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

        export_name = '{}.png'.format(index)
        rows.writerow([export_name, path, dimension, slice_coordinate])
        image.save(join(args.destination, export_name))
