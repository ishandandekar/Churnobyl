import time
from multiprocess import Pool


def f(args):
    def task(x):
        return x * x * x * x * x * x * x * x * x + 1_000_000

    start_wo_pool = time.time()
    r_wo_pool = list(map(task, range(1_000)))
    wo_time = time.time() - start_wo_pool

    start_w_pool = time.time()
    with Pool() as p:
        result = p.map(task, range(1_000))
    w_time = time.time() - start_w_pool
    return wo_time, w_time


print(f("a"))
