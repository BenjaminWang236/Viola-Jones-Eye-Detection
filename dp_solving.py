import numpy as np
import logging
import sys

arguments = len(sys.argv) - 1
if arguments != 2:
    logging.warning(
        f"Number of arguments should be 2 but is {arguments}. \
        Try again"
    )
    sys.exit(1)
m = int(sys.argv[1])
n = int(sys.argv[2])
if m < 1 or n < 1:
    logging.warning(
        f"(m, n) both must be => 1 but is ({m}, {n}) currently. \
        Try again"
    )
    sys.exit(1)
grid = np.zeros((m, n))

for i in range(m):
    grid[i][0] = 1
for j in range(n):
    grid[0][j] = 1
for i in range(1, m):
    for j in range(1, n):
        try:
            grid[i][j] = int(grid[i - 1][j] + grid[i][j - 1])
        except OverflowError as err:
            logging.error(f"{err}. Exiting")
            sys.exit(1)

# for i in range(m):
#     for j in range(n):
#         print(int(grid[i][j]), end='\t')
#     print('\n')

print(f"Solution ({m}, {n}): {int(grid[m-1][n-1])}")
