#FROM python:3.8-rc-slim

FROM python:3.7
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

#COPY app.py /app/app.py
#COPY routes /app/routes
#COPY static /app/static

COPY requirements.txt /usr/src/app/requirements.txt

#RUN pip install -r /app/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /usr/src/app
#WORKDIR /app

EXPOSE 5002
ENTRYPOINT ["python3"]

CMD ["app.py"]