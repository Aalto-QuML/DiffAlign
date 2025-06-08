import os
import sys
import multiprocessing
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def main_subprocess(queue, parallel_num, data, calculation_range):
    import torch
    # device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(gpu_id)
    # log.info(f"Doing calculations on device {device}")

    local_data = torch.tensor(data[calculation_range[0]:calculation_range[1]])

    t0 = time.time()
    for n in range(len(local_data)):
        torch.linalg.inv(local_data[n])
    t1 = time.time()
    log.info(f"Time taken on process {parallel_num}: {t1 - t0}")

def main():
    num_parallel = int(sys.argv[1])

    N = 800
    N_matrix = int(1e3)
    data = np.random.randn(N, N_matrix, N_matrix)

    assert N % num_parallel == 0
    calculations_per_gpu = N // num_parallel
    calculation_ranges = [(calculations_per_gpu*i, calculations_per_gpu*(i+1)) for i in range(num_parallel)]

    q = multiprocessing.Queue() # To aggregate the results in the end
    processes = []
    for i in range(num_parallel):
        p = multiprocessing.Process(target=main_subprocess, args=(q, i, data, calculation_ranges[i]))
        p.start()
        processes.append(p)
    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()