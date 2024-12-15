FROM ubuntu:20.04

RUN apt-get update -qq &&  \
    apt-get install -yqq --no-install-recommends \
        python3-pip                 \
    &&  rm -rf /var/lib/apt/lists/* && \
    pip install                     \
        numpy==1.22.4               \
        tqdm==4.64.0                \
	matplotlib==3.7.5

RUN \
    apt-get update   -qq                  && \
    apt-get install -yqq --no-install-recommends \
        libxext6 libsm6 ffmpeg            && \
    rm -rf /var/lib/apt/lists/*           && \    
    pip install                              \
       	jupyterlab==3.2.9                    \
        scikit-learn==1.1.1                  \
        scikit-learn-intelex==2021.5.3       \
        torch==1.13.0                        \
        torchvision==0.14.0                  \
        torchaudio==0.13.0                   \
        --extra-index-url                    \
        https://download.pytorch.org/whl/cu114 


WORKDIR /src

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/src", "--allow-root"]
