#!/bin/bash

REGION=us-west1
ZONE=us-west1-b
GPU_TYPE=nvidia-tesla-k80
GPU_NUM=1
NAME=hw4

# Config zone.
gcloud config set compute/zone ${ZONE}

# Create GPU instance.
gcloud compute instances create ${NAME} \
    --machine-type n1-standard-2 \
    --accelerator type=${GPU_TYPE},count=${GPU_NUM} \
    --image-family ubuntu-1804-lts --image-project ubuntu-os-cloud \
    --boot-disk-size 20GB \
    --maintenance-policy TERMINATE --restart-on-failure \
    --metadata startup-script='#!/bin/bash 
        # Check for CUDA and try to install.
        # Determine whether or not we are first-time logging in by existence of CUDA
        if ! dpkg-query -W cuda; then
          # Disable ssh
          service ssh stop

          apt-get update

          # Install openmpi
          apt-get install openmpi-bin openmpi-common libopenmpi-dev -y

          # Install make and gcc
          apt-get install make gcc g++ -y

          # Install ruby and rmate
          apt install ruby -y && gem install rmate

          # Install cmake
          apt-get install cmake -y

          # kernel headers and development packages for the currently running kernel
          apt-get install linux-headers-$(uname -r)

          curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
          dpkg -i ./cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
          apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
          apt-get update
          apt-get install cuda -y
          rm -f ./cuda-repo*.deb
          echo >> /etc/profile
          echo "# CUDA" >> /etc/profile
          echo "export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}" >> /etc/profile
          echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
" >> /etc/profile

          apt-get install cuda-toolkit-10-2 -y

          # Setup auto shutdown
          wget -O /bin/auto_shutdown.sh https://raw.githubusercontent.com/stanford-cme213/stanford-cme213.github.io/master/gcp/auto_shutdown.sh
          chmod +x /bin/auto_shutdown.sh

          wget -O /etc/init.d/auto_shutdown https://raw.githubusercontent.com/stanford-cme213/stanford-cme213.github.io/master/gcp/auto_shutdown
          chmod +x /etc/init.d/auto_shutdown

          update-rc.d auto_shutdown defaults
          update-rc.d auto_shutdown enable
          service auto_shutdown start

          # Enable ssh
          service ssh start
        fi
        '

echo "Installing necessary libraries. You will be able to log into the VM after several minutes with:
gcloud compute ssh ${NAME}"
