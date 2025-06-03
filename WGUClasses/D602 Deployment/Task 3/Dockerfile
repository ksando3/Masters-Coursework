
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install fastapi uvicorn gunicorn


EXPOSE 8000


COPY ./app /code/app/