import multiprocessing as mp
import numpy as np

# def square(x):
#     return np.square(x)

def square(i, x, queue):
    print("In process {}".format(i, ))
    queue.put(np.square(x))

if __name__ == '__main__':
    # x = np.arange(1024)
    # print(x)
    # print(mp.cpu_count())
    # pool = mp.Pool(32)
    # squared = pool.map(square, [x[32*i:32*i+32] for i in range(32)])
    # print(squared)
    # print("+")
    processes = []  # A
    queue = mp.Queue()  # B
    x = np.arange(64)  # C
    for i in range(8):  # D
        start_index = 8 * i
        proc = mp.Process(target=square, args=(i, x[start_index:start_index + 8], queue))
        proc.start()
        processes.append(proc)

    for proc in processes:  # E
        proc.join()

    for proc in processes:  # F
        proc.terminate()
    results = []
    while not queue.empty():  # G
        results.append(queue.get())

    print(results)