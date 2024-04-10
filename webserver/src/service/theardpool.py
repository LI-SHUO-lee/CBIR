import threading
from concurrent.futures import ThreadPoolExecutor
from service.train import do_train


def thread_runner(thread_num, func, *args):
    executor = ThreadPoolExecutor(thread_num)
    f = executor.submit(do_train, *args)


def direct_call_do_train(table_name, database_path):
    do_train(table_name, database_path)
