import logging
import time
from encoder.encode import feature_extract
from preprocessor.vggnet import VGGNet



def do_train_without_milvus(table_name, database_path):

    try:
        vectors, names = feature_extract(database_path, VGGNet())
        return vectors, names
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)
