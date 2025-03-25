#!/bin/bash

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 851725541946.dkr.ecr.ap-south-1.amazonaws.com

docker pull 851725541946.dkr.ecr.ap-south-1.amazonaws.com/gmcdocker:v15

if [ "$(docker ps -q -f name=my-app)" ]; then
    docker stop my-app
fi 

if [ "$(docker ps -aq -f name=my-app)" ]; then
    docker rm my-app || true
fi 

sudo docker run -d -p 80:5000 -e DAGSHUB_TOKEN=ea3e52cfe4a018dc25e349377aaf17a193a5bc60 --name my-app 851725541946.dkr.ecr.ap-south-1.amazonaws.com/gmcdocker:v15