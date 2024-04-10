import concurrent.futures
import os
import threading, logging
import multiprocessing
import numpy as np
from common.config import DATA_PATH as database_path
from encoder.utils import get_imlist
from preprocessor.vggnet import VGGNet
from diskcache import Cache
from common.const import default_cache_dir
from functools import partial
from multiprocessing import Process, Queue, Pool


# single process and thread
def feature_extract(database_path, model):
    cache = Cache(default_cache_dir)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    for i, img_path in enumerate(img_list):
        norm_feat = model.vgg_extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name.encode())
        current = i + 1
        total = len(img_list)
        cache['current'] = current
        cache['total'] = total
        print("extracting feature from image No. %d , %d images in total" % (current, total))
    #    feats = np.array(feats)
    return feats, names


# multiple threads
def feature_extract_with_thread_pool(database_path, model):
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        cache = Cache(default_cache_dir)
        feats = []
        names = []
        img_list = get_imlist(database_path)
        futures = {}
        for i, img_path in enumerate(img_list):
            # 将类的方法包装成可调用的对象
            model = model
            callable_obj = partial(model.vgg_extract_feat, img_path)
            future = executor.submit(callable_obj)
            futures[img_path] = future

        for i, (img_path, future) in enumerate(futures.items()):
            norm_feat = future.result()
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name.encode())
            current = i + 1
            total = len(img_list)
            cache['current'] = current
            cache['total'] = total
            print("extracting feature from image No. %d , %d images in total" % (current, total))
        #    feats = np.array(feats)
        return feats, names


# multiple process use pool
def feature_extract_with_process_pool(database_path, model):
    try:
        with concurrent.futures.ProcessPoolExecutor(2) as executor:
            cache = Cache(default_cache_dir)
            feats = []
            names = []
            img_list = get_imlist(database_path)

            results = executor.map(model.vgg_extract_feat, img_list)

            for i, (img_path, norm_feat) in enumerate(zip(img_list, results)):
                img_name = os.path.split(img_path)[1]
                feats.append(norm_feat)
                names.append(img_name.encode())
                current = i + 1
                total = len(img_list)
                cache['current'] = current
                cache['total'] = total
                print("extracting feature from image:%s No. %d , %d images in total" % (img_path, current, total))
            return feats, names
    except Exception as e:
        logging.error(e)
        return "multiple process: Error with {}".format(e)
