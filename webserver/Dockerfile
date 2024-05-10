From python:3.6.5

WORKDIR /app/src
COPY . /app

ENV TF_XLA_FLAGS --tf_xla_cpu_global_jit
#RUN mkdir -p /root/.keras/models && mv /app/data/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/

RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt

RUN mkdir -p /tmp/search-images

CMD python3 app.py