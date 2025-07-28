# docker build -t easydistill:0.0.1 .
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set install.trusted-host mirrors.aliyun.com

COPY . .

RUN pip install -U pip && pip install -r requirements.txt

RUN pip install -v -e .

ENV LANG=C.UTF-8
