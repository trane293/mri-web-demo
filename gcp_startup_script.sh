sudo apt install python-dev python-pip build-essential gcc git
pip install virtualenvwrapper --user
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
echo 'export PROJECT_HOME=$HOME/Devel' >> ~/.bashrc
echo 'source $HOME/.local/bin/virtualenvwrapper.sh' >> ~/.bashrc
echo 'export PATH="$PATH:/home/anmol_sharma293/.local/bin/"' >> ~/.bashrc
source ~/.bashrc
mkvirtualenv gui-proj
workon gui-proj
pip install -r requirements.txt
sudo pip install git+https://www.github.com/keras-team/keras-contrib.git
