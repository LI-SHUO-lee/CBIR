import os
import platform


def get_system_type():
    system_type = platform.system()
    return system_type


system = get_system_type()
print(f"The config type is: {system}")

if system == 'Windows':
    MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.92")
    MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
    VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
    DATA_PATH = os.getenv("DATA_PATH", "F:\\deep learning\\temp")
    DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
    UPLOAD_PATH = "F:\\tmp\\search-images"
    COMPUTE_VALUE = [1, 10, 100, 1000, 2011, 5952, 10000, 11963, 17953, 20000, 1000000]
else:
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
    VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)
    DATA_PATH = os.getenv("DATA_PATH", "/data/jpegimages")
    DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
    UPLOAD_PATH = "/tmp/search-images"
    COMPUTE_VALUE = [1, 10, 100, 1000, 2011, 5952, 10000, 11963, 17953, 20000, 1000000]
