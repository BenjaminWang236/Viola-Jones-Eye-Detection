import numpy as np
import logging
import sys

arguments = len(sys.argv) - 1
if arguments != 2:
    logging.error(f"Number of arguments should be 2 but is {arguments}")
    sys.exit(1)
m = int(sys.argv[1])
n = int(sys.argv[2])
if m < 1 or n < 1:
    logging.error(f"m/n must be => 1 but is ({m}, {n}) currently. Try again")
    sys.exit(1)
grid = np.zeros((m, n))

for i in range(m):
    grid[i][0] = 1
    grid[i][1] = i + 1
for j in range(n):
    grid[0][j] = 1
    grid[1][j] = j + 1
for i in range(2, m):
    for j in range(i, n):
        grid[i][j] = int(grid[i - 1][j] + grid[i][j - 1])
        if j < m and i < n:
            grid[j][i] = grid[i][j]

# for i in range(m):
#     for j in range(n):
#         print(int(grid[i][j]), end='\t')
#     print('\n')

# Since python variables are global, i and j are at n and m where they exited
print(f"Solution ({m}, {n}): {int(grid[i][j])}")
