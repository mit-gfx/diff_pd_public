from multiprocessing import Process, Queue

# Input arguments: tasks = [(func, args)].
def parallel_work(tasks, num_workers):
    task_queue = Queue()
    done_queue = Queue()
    num_workers = min(num_workers, len(tasks))
    for func, args in tasks:
        task_queue.put((func, args))
    def worker(input, output):
        for func, args in iter(input.get, 'stop'):
            result = func(*args)
            output.put((args, result))
    for _ in range(num_workers):
        Process(target=worker, args=(task_queue, done_queue)).start()
    for _ in tasks:
        _, _ = done_queue.get()
    for _ in range(num_workers):
        task_queue.put('stop')