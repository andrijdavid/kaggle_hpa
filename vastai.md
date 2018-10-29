# Vast.ai setup
It consists of:

    -Creating a new instance on the Vast.ai console (using the fast.ai OS Image and “Run interactive shell server, SSH”)
    -Installing install pip, awscli, and git.
    -Creating my file structure, and downloading my code from github
    -Activating the fast.ai python environment
    -Starting Jupyter notebook
    -Pulling the data I need for a particular project

## Here’s the code I use for steps 2 thru 5:
```
#install pip, awscli, git
pip install --upgrade pip
echo “export PATH=~/.local/bin:$PATH” >> ~/.bashrc
source ~/.bashrc
pip install awscli --upgrade --user
apt install git
git config --global user.email "name@email.com"
git config --global user.name “First Last”
#create / update folder structure
cd /notebooks/fastai/
git pull
cd /notebooks
git clone https://github.com/YOURGIT.git
cd /data/
mkdir sav
mkdir localtoremote
#get ready to use jupyter
source activate fastai
aws configure
cd /notebooks
pip install jupyter; jupyter notebook --ip=127.0.0.1 --port=8080 --allow-root
```

# Then in my Jupyter notebook for a particular project I’ll do something like:
```
!wget https://s3-us-west-2.amazonaws.com/youramazons3bucket/data/cifar10/cifar10.tgz 1 -P /data/
!tar xvzf /data/cifar10.tgz -C /data/
!aws s3 cp --recursive s3://youramazons3bucket/data/sav/cifar10 /data/sav/cifar10
```
Hope this is helpful to someone.