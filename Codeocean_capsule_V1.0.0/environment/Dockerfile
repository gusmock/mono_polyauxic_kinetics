# hash:sha256:178f99a03d8cca81ac468bbfc0964e83b9920014383e8dcf291d04513a15871a
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    matplotlib==3.10.8 \
    numpy==2.4.0 \
    openpyxl==3.1.5 \
    pandas==2.3.3 \
    scipy==1.16.3

RUN pip3 install -U --no-cache-dir \
    streamlit==1.52.2
