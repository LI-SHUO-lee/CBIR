import platform


def get_system_type():
    system_type = platform.system()
    return system_type


system = get_system_type()
print(f"The const type is: {system}")

if system == 'Windows':
    UPLOAD_PATH="F:\\tmp\\search-images"
    default_indexer = "milvus"
    default_cache_dir = "F:\\deep learning\\temp_251"
    input_shape = (224, 224, 3)

else:
    UPLOAD_PATH = "/tmp/search-images"
    default_indexer = "milvus"
    default_cache_dir = "./tmp"
    input_shape = (224, 224, 3)
