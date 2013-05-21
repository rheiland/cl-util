__author__ = 'marcdeklerk'

import csv
from evaluate import global_dims

res_file = open('results/mp_from_res_and_tile.csv', 'w')
resWriter = csv.writer(res_file)

tile_dims = [(16, 16), (32, 32)]

columns = [None]
columns += ["{0}x{1}".format(dim[0], dim[1]) for dim in tile_dims]

resWriter.writerow(columns)

for global_dim in global_dims:
    row = [
        "{0}x{1} ({2:.2f} MPix)".format(global_dim[0], global_dim[1], float(global_dim[0]*global_dim[1])/(1024*1024))
    ]

    for tile_dim in tile_dims:
        n_tiles = (global_dim[0]/tile_dim[0])*(global_dim[1]/tile_dim[1])

        row.append(n_tiles)

    resWriter.writerow(row)

True