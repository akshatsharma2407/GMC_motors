#!/bin/bash

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 851725541946.dkr.ecr.ap-south-1.amazonaws.com

docker pull 851725541946.dkr.ecr.ap-south-1.amazonaws.com/gmcdocker:v15

docker stop my-app || true
docker rm my-app || true

sudo docker run -d -p 80:5000 -e DAGSHUB_TOKEN=ea3e52cfe4a018dc25e349377aaf17a193a5bc60 --name my-app 851725541946.dkr.ecr.ap-south-1.amazonaws.com/gmcdocker:v15