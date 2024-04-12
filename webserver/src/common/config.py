import os
import platform


def get_system_type():
    system_type = platform.system()
    return system_type


system = get_system_type()
print(f"The system type is: {system}")

if system == 'Windows':
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
    VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
    DATA_PATH = os.getenv("DATA_PATH", "F:\\deep learning\\temp")
    DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
    UPLOAD_PATH = "/tmp/search-images"
    # 线程池中的线程数
    THREAD_NUM = 1
else:
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
    VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
    # DATA_PATH = os.getenv("DATA_PATH", "/data/jpegimages")
    DATA_PATH = os.getenv("DATA_PATH", "/home/lishuo/upm/data/temp")
    DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
    UPLOAD_PATH = "/tmp/search-images"
    # 线程池中的线程数
    THREAD_NUM = 1
