#FROM docker.1ms.run/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM dockerproxy.net/pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel


RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 gnupg libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update &&  \
    apt-get install -y rsync && \
    apt-get install -y vim && \
    apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:123456' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo "export VISIBLE=now" >> /etc/profile

# RUN pip install --no-cache-dir semilearn --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /data2
