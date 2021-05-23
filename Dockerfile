FROM python:3.8.8-slim 

# Make directories suited to your application
RUN mkdir -p /home/yolo-app
RUN mkdir /home/yolo-app/model
RUN mkdir /home/yolo-app/static
RUN mkdir /home/yolo-app/static/detection

# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8000

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

WORKDIR /home/yolo-app

# Copy and install requirements
COPY requirements.txt /home/yolo-app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy contents from your local to your docker container
COPY . /home/yolo-app

CMD ["python", "main.py"]
