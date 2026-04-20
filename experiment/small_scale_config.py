import os


SMALL_SCALE_SIDE_LENGTH_KM = 3.0
SMALL_SCALE_DOWNLOAD_DIST_M = 1500
SMALL_SCALE_WORKER_COUNTS = [15, 30, 45, 60, 75]
SMALL_SCALE_FIXED_WORKER_COUNT = 45
SMALL_SCALE_BATCH_COUNTS = [2, 4, 6, 8, 10]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT_DIR = os.path.join(PROJECT_ROOT, "result", "small_scale_3x3")
WORKER_RESULT_DIR = os.path.join(RESULT_ROOT_DIR, "worker_count_sweep")
BATCH_RESULT_DIR = os.path.join(RESULT_ROOT_DIR, "batch_count_sweep")

