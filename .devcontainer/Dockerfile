FROM mambaorg/micromamba:focal

USER root

ENV USERNAME=mambauser \
    DEBIAN_FRONTEND=noninteractive


ENV PATH=/opt/conda/envs/env_label/bin:$PATH

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl sudo git nodejs wget curl git-flow vim gpg

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN chown -R mambauser:1000 /opt/conda/

RUN \
    echo "**** install Stars ****" && \
    stars_version=2.10.17 && \
    curl -L https://github.com/Terradue/Stars/releases/download/${stars_version}/Stars.${stars_version}.linux-x64.deb > Stars.deb && \
    echo "deb http://security.ubuntu.com/ubuntu focal-security main" | tee /etc/apt/sources.list.d/focal-security.list && \
    apt-get update && \
    apt-get install libssl1.1 && \
    dpkg -i Stars.deb && \
    rm Stars.deb
    
USER mambauser

WORKDIR /home/mambauser

ADD .devcontainer/environment.yml /tmp/environment.yml

RUN micromamba create -f /tmp/environment.yml


