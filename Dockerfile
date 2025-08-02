# docker build -t easydistill:0.0.1 .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 设置环境变量和pip源
ENV LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set install.trusted-host mirrors.aliyun.com

# 先安装依赖文件（利用Docker缓存层）
COPY requirements.txt .
RUN pip install -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 再拷贝项目代码
COPY . .

# 安装当前项目（使用-no-cache-dir减少镜像大小）
RUN pip install --no-cache-dir -v -e .

# 清理临时文件（如有）
RUN find / -type d -name '__pycache__' -exec rm -rf {} + \
    && rm -rf /root/.cache \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
