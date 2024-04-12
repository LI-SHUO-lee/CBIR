import concurrent.futures
import os
import threading, logging
import multiprocessing
import time

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


'''
tensorflow 自带多线程处理
'''
import tensorflow as tf


def parse_function(image_path, label):
    # 读取和预处理图像
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # 将图像调整为 VGG 模型所需的大小
    image = tf.keras.applications.vgg16.preprocess_input(image)  # 预处理图像
    return image, label


def tf_dataset_function(database_path):
    img_list = get_imlist(database_path)
    names = [os.path.split(img_path)[1] for img_path in img_list]
    dataset = tf.data.Dataset.from_tensor_slices((img_list, names))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # 加载 VGG 模型
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

    # 向量化图像
    vectors = vgg_model.predict(dataset)
    print(f'--------反回的vector数量为：{len(vectors)}')


if __name__ == '__main__':
    start = time.time()
    tf_dataset_function('F:\\deep learning\\temp')
    print(f'used time : {time.time() - start} s')
