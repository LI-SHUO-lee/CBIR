import os, time
import os.path as path
import logging as log
from common.config import DATA_PATH, DEFAULT_TABLE, COMPUTE_VALUE
from common.const import UPLOAD_PATH
from common.const import input_shape
from common.const import default_cache_dir
from service.train import do_train
from service.search import do_search
from service.count import do_count
from service.delete import do_delete
from service.theardpool import thread_runner, thread_runner_fun
from preprocessor.vggnet import vgg_extract_feat
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from service.search import query_name_from_ids
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify
from flask_restful import reqparse
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16
# from encoder.encode import get_response_msg
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.preprocessing import image
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from diskcache import Cache
import shutil

config = tf.ConfigProto(
    # device_count={"CPU": 4},
    # inter_op_parallelism_threads=1,
    # intra_op_parallelism_threads=4,
    # log_device_placement=True
)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
global sess
sess = tf.Session(config=config)
set_session(sess)

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)

model = None


def load_model():
    global graph
    graph = tf.get_default_graph()

    global model
    model = VGG16(weights='imagenet',
                  input_shape=input_shape,
                  pooling='max',
                  include_top=False)


@app.route('/api/v1/train', methods=['POST'])
def do_train_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        add_argument('File', type=str). \
        parse_args()
    table_name = args['Table']
    file_path = args['File']
    try:
        thread_runner(1, do_train, table_name, file_path)
        filenames = os.listdir(file_path)
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        for filename in filenames:
            shutil.copy(file_path + '/' + filename, DATA_PATH)
        return "Start"
    except Exception as e:
        return "Error with {}".format(e)


@app.route('/api/v1/delete', methods=['POST'])
def do_delete_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    log.info("delete table.")
    status = do_delete(table_name)
    try:
        shutil.rmtree(DATA_PATH)
    except:
        log.error("cannot remove {DATA_PATH}")
    return "{}".format(status)


@app.route('/api/v1/count', methods=['POST'])
def do_count_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    rows = do_count(table_name)
    return "{}".format(rows)


@app.route('/api/v1/process')
def thread_status_api():
    cache = Cache(default_cache_dir)

    if not 'current' in cache :
        return "current: {}, total: {}".format(1, 0)

    return "current: {}, total: {}".format(cache['current'], cache['total'])


@app.route('/data/<image_name>')
def image_path(image_name):
    file_name = DATA_PATH + '/' + image_name
    if path.exists(file_name):
        return send_file(file_name)
    return "file not exist"


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str). \
        add_argument("Num", type=int, default=1). \
        parse_args()

    table_name = args['Table']
    if not table_name:
        table_name = DEFAULT_TABLE
    top_k = args['Num']
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        start = time.time()
        res_id, res_distance = do_search(table_name, file_path, top_k, model, graph, sess)
        time_cost = (time.time() - start)
        log.info(f'The search process cost {time_cost:.2f} seconds...')
        print(f'The search process cost {time_cost:.2f} seconds...')
        if isinstance(res_id, str):
            return res_id
        res_img = [request.url_root + "data/" + x for x in res_id]
        res = dict(zip(res_img, res_distance))
        res = sorted(res.items(), key=lambda item: item[1])
        return jsonify(res), 200
    return "not found", 400


@app.route('/api/v1/accuracy', methods=['POST'])
def do_accuracy_api():
    args = reqparse.RequestParser(). \
        add_argument("search_path", type=str). \
        parse_args()

    search_path = args['search_path']
    dirs = [f for f in os.listdir(search_path) if (f.endswith('.jpg') or f.endswith('.png'))]
    table_name = DEFAULT_TABLE
    count = 0
    correct = 0
    error = 0
    try:
        for index, file_name in enumerate(dirs):
            count += 1
            file_path = os.path.join(search_path, file_name)
            res_id, res_distance = do_search(table_name, file_path, 1, model, graph, sess)
            log.info(f'{count} is processing,{res_id} ---> {res_distance} --> {file_path}')
            if res_id[0] == file_name:
                correct += 1
            else:
                error += 1
                log.error(f'the query and result is different:{file_name}')
            if count in COMPUTE_VALUE or index == len(dirs) - 1:
                accuracy = (correct / count) * 100
                log.info(f'the correct num is {correct} and error num is {error}')
                log.info(f'It has processed {count} images and the accuracy is {accuracy}%...')
    except Exception as e:
        log.error(f'{count} error，the massage is {format(e)}')

    log.info('accuracy evaluate is end')
    return 'ok'


if __name__ == "__main__":
    # thread_runner_fun(1, consumer_get_imgs_paths)
    # thread_runner_fun(1, get_response_msg)
    load_model()
    app.run(host="0.0.0.0", debug=True)
