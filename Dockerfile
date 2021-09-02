FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY dataset.py load_data.py main.py model.pth model.py parameters.py run_saved_model.py utils.py .

RUN python main.py
