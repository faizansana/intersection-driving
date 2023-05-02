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

# Install Python environment
ENV VIRTUAL_ENV=/home/docker/env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt /tmp
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt && chown -R docker:docker /home/docker/env

USER docker:docker
ENV SHELL=/bin/bash
ENV DISPLAY=:1.0
WORKDIR /home/docker
ENTRYPOINT ["fixuid"]
CMD ["/usr/bin/supervisord"]