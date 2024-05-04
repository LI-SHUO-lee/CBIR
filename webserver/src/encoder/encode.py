import os, time
import numpy as np
from common.config import DATA_PATH as database_path
from common.config import COMPUTE_VALUE
from encoder.utils import get_imlist
from preprocessor.vggnet import VGGNet
from diskcache import Cache
from common.const import default_cache_dir


def feature_extract(database_path, model):
    cache = Cache(default_cache_dir)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    start = time.time()
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
        if current in COMPUTE_VALUE or i == len(img_list) - 1:
            time_cost = (time.time() - start)
            print(f'It has processed {current} images and the time cost {time_cost:.2f} seconds...')
    #    feats = np.array(feats)
    return feats, names

# def get_features_from_kafka(database_path, model):
#     kafkautill = KafkaInterface()
#     topic = 'first'
#
#     img_list = get_imlist(database_path)
#     executor = ThreadPoolExecutor(1)
#     f = executor.submit(
#         # 发消息
#         kafkautill.send_message_asyn_producer_callback(topic, ','.join(img_list)))
#
#     # 获取消息
#     f = executor.submit(get_msg_response)
#
#
# def get_msg_response():
#     kafkautill = KafkaInterface()
#     topic_response = 'first_response'
#     feats = []
#     names = []
#     # 读消息
#     from kafka import KafkaConsumer
#     consumer = KafkaConsumer('first_response', bootstrap_servers='192.168.1.93:9092', auto_offset_reset='latest')
#     for k, v in enumerate(consumer, start=1):
#         try:
#             mesage_res = v.value.decode('utf-8')
#
#             # 解析 JSON 数据
#             message_data = json.loads(mesage_res)
#             # 获取字符串和数组数据
#             img_name = message_data['img_name']
#             norm_feat = message_data['norm_feat']
#
#             feats.append(norm_feat)
#             names.append(img_name)
#
#             print(f'=========> feats size is {len(feats)}')
#             print(f'=========> names size is {len(names)}')
#             print(f'{names}:特征化流程结束！！！')
#         except Exception as e:
#             print(format(f'出异常了：{e}'))
