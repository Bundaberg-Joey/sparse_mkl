FROM continuumio/miniconda3

RUN apt-get -y update \
    && apt-get -y install \
    && apt-get install -y git

COPY environment.yml .

RUN conda env update -n base --file environment.yml \
    && conda clean -afy

RUN pip install gpflow pandas

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 
