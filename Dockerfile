# pull base image
FROM python:3.7-slim

# set work directory
WORKDIR /app

# install dependencies
COPY requirements.txt /app/
RUN pip --no-cache-dir install -r requirements.txt

# copy project
COPY . /app/

# set environment variable for application file
ENV FLASK_APP=app.py 

# set port number
EXPOSE 5000

# run application
ENTRYPOINT ["python"]
CMD ["app.py"]
