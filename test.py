import multiprocessing

num_cores = multiprocessing.cpu_count()
optimal_processes = max(1, num_cores - 1)
print(optimal_processes)