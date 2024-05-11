FROM base:ubuntu

USER root

ENV PYTHONPATH="${PYTHONPATH}:/bachelor_thesis"

WORKDIR /bachelor_thesis
COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
