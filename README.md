This repository contains functions/kernels, oriented to parallel image processing.

Images are treated as tensors whose order are: rows, columns, channels, and batches (RCCB). The memory storage is col-major, so all functions obey this data storage. It is not possible to execute the kernels in row-major order or with a different tensor order.

## Operations implemented

- Dot
- Convolve 2d
- Convolve 3d
- Softmax
