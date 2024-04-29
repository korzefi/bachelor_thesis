FROM base:ubuntu

USER root

ENV PYTHONPATH="${PYTHONPATH}:/bachelor_thesis"

WORKDIR /bachelor_thesis
COPY . .

RUN pip install -r requirements.txt

# TODO: should these files be generated inside docker or immidiately copied outside?
# TODO: Maybe some result folder should be just mapped?

# TODO: Maybe just add some copy command to script to copy files when docker process is finished?
CMD ["python3", "scripts/preprocessing/grouping_raw_data.py"]
#CMD ["python3", "scripts/preprocessing/clustering.py"]
