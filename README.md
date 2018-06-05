# Install Docker 
sudo apt-get remove docker docker-engine docker.io docker-ce
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce

# Install gcloud
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
gcloud config set project datamining-204710

# Pull base image
sudo docker pull frolvlad/alpine-python-machinelearning
sudo docker pull tensorflow/tensorflow

# Create Dockerfile (image) over base image
cd <workdir>
gedit Dockerfile # 7 lines
FROM tensorflow/tensorflow
RUN mkdir -p /workdir
WORKDIR /workdir
COPY beer_data.zip /workdir
RUN unzip beer_data.zip
COPY *.py /workdir/
CMD sh
#EOF

sudo docker build . -t datamining:latest

# Push image to Dockerhub
sudo docker login 
sudo docker tag datamining mbondarenko2228/datamining
sudo docker push mbondarenko2228/datamining

# Run image (on cloud)
# https://console.cloud.google.com/compute/instances?project=datamining-204710
docker pull mbondarenko2228/datamining
docker run -p 2223:2223 -it mbondarenko2228/datamining
# run workers before parameter server
 # instance 1
 python tensor_beer.py --job_name ps --steps 10
 # instance 2
 python tensor_beer.py --job_name worker --task_index 0 --steps 10
 # instance 3
 python tensor_beer.py --job_name worker --task_index 1 --steps 10
 # instance 4
 python tensor_beer.py --job_name worker --task_index 2 --steps 10

# Compare START and FINISH time
# with http://www.onlineconversion.com/days_between_advanced.htm


# Clean docker containers
docker ps -alq # get <containerId>
docker cp <containerId>:/workdir/mlp.pkl /home/m_bondarenko_2228/workdir # copy result from Docker to Cloud
#docker cp 97c6864f8024:/workdir/mlp.pkl /home/m_bondarenko_2228/workdir

sudo apt-get install docker docker-engine docker.io

# Copy result (locally)
cd ~/Beer
gcloud compute scp instance-2:/home/m_bondarenko_2228/workdir/mlp.pkl .

# Remove unnecessary images (locally)
sudo docker image ls # get <image_hash>
sudo docker rmi -f <image_hash>


