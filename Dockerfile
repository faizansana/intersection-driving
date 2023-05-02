FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS base
SHELL [ "/bin/bash", "-c" ]

ARG CARLA_VERSION=0.9.13

ENV DEBIAN_FRONTEND noninteractive
ENV CARLA_VERSION $CARLA_VERSION
RUN apt-get update && apt-get install -y sudo curl wget git python3-venv

# Add a docker user so we that created files in the docker container are owned by a non-root user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Remap the docker user and group to be the same uid and group as the host user.
# Any created files by the docker container will be owned by the host user.
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

# Install vnc server
USER root:root
RUN apt-get update && apt-get install -y lxde x11vnc xvfb mesa-utils && apt-get purge -y light-locker
RUN apt-get install -y supervisor
EXPOSE 5900

COPY --chown=docker ./supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc
# Putting conda in path to use 'conda activate'
ENV PATH=$CONDA_DIR/bin:$PATH


COPY ./environment.yml /tmp
RUN conda update conda \
    && conda env create -n driving -f /tmp/environment.yml

USER docker:docker
RUN conda init && \
    echo "conda activate driving" >> ~/.bashrc
ENV SHELL=/bin/bash
ENV DISPLAY=:1.0
WORKDIR /home/docker
ENTRYPOINT ["fixuid"]
CMD ["/usr/bin/supervisord"]