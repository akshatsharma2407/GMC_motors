#baseimage
FROM python:3.11-slim

#workdir
WORKDIR /app

#copy
COPY flask_app/ /app/
COPY models/pipe.pkl /app/models/pipe.pkl

#requirements.txt
RUN pip install -r requirements.txt
RUN pip install category-encoders==2.7.0 

#port
EXPOSE 5000

# CMD
CMD ["gunicorn","-b","0.0.0.0:5000","app:app"]
