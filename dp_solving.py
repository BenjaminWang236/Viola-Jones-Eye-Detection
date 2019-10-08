import numpy as np
import pprint as pp
import sys

arguments = len(sys.argv) - 1
m = int(sys.argv[1])
n = int(sys.argv[2])

grid = np.zeros((m, n))

for i in range(m):
    grid[i][0] = 1
for j in range(n):
    grid[0][j] = 1
for i in range(1, m):
    for j in range(1, n):
        grid[i][j] = int(grid[i-1][j] + grid[i][j-1])

# for i in range(m):
#     for j in range(n):
#         print(int(grid[i][j]), end='\t')
#     print('\n')

print(f'Solution ({m}, {n}): {int(grid[i][j])}')
# print(grid)
# pp.pprint(grid)
