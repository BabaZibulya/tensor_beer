FROM tensorflow/tensorflow
RUN mkdir -p /workdir && mkdir /workdir/tf_res
WORKDIR /workdir

COPY beer_data.zip /workdir
RUN unzip beer_data.zip && chmod -R 777 /workdir

COPY *.py /workdir/
CMD sh
# python3 tensor_beer.py

