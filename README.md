# Vast.ai setup
```
conda update conda -y
conda install anaconda -y

pip install --upgrade pip
pip isntall --user kaggle
apt-get install nano
apt-get install unzip


echo export PATH=~/.local/bin:$PATH >> ~/.bashrc
source ~/.bashrc

#Install fastai
conda install -c pytorch pytorch-nightly cuda92
conda install -c fastai torchvision-nightly
conda install -c fastai fastai

conda install -c conda-forge opencv 

git config --global user.email "name@email.com"
git config --global user.name “First Last”

git clone https://github.com/tcapelle/kaggle_hpa

#kaggle recover data
cd kaggle_hpa
kaggle competitions download human-protein-atlas-image-classification
mkdir train
mkdir test
unzip -q train.zip -d train/
unzip -q test.zip -d test/
rm *.zip
```
