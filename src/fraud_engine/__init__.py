import os
import warnings

# Prevent OpenMP runtime collision on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress Apple Accelerate BLAS spurious floating-point status register warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

__version__ = "1.0.0"
