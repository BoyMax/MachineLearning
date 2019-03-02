# Create instance
1. GPU 

2. Ubuntu

3. Allow http and https

4. VPC network > static external IP

5. Firewall rules > all regions, specified tcp port number

# SSh to gcp
```
conda install -c conda-forge google-cloud-sdk
```

```
gcloud auth login
gcloud config set project PROJECT_ID

```
View gcloud command, and use it to ssh to gcp

#Install CUDA

### get driver for CUDA 8.0
```
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
```

```
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
```

### update to 9.0
```
sudo apt-get update
sudo apt-get install cuda-9-0
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875 # p
```

### add to environment variable
```
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```
nvidia-smi 确认安装成功


#Install cuDNN

1. 官网下载包（https://developer.nvidia.com/rdp/cudnn-download）


2. scp （在本机操作）

```
gcloud compute scp ~/Downloads/cudnn-9.0-linux-x64-v7.4.2.24.tgz {instance_name}:~/
```

3. install (在server上)

```
tar xzvf cudnn-9.0-linux-x64-v7.4.2.24.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
rm -rf ~/cuda
rm cudnn-9.0-linux-x64-v7.tgz
view raw
```

#Install conda
1.安装conda

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc
```
2.新建虚拟环境(dip是环境名)

```
conda create -n dip python=3.6
```

3.激活环境
```
conda activate dip
```

4.安装numpy scipy matplotlib scikit-image jupyter

```
conda install numpy scipy matplotlib scikit-image jupyter
```

安装pytorch

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

```


#Configure jupyter
1. 生成配置文件
```
$jupyter notebook --generate-config
```

2. 生成密码 进入python终端

```
from IPython.lib import passwd

In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: u'sha1:0e422dfccef2:84cfbcbb3ef95872fb8e23be3999c123f862d856' 
```

3. 打开配置

```
vim ~/.jupyter/jupyter_notebook_config.py
```

4. 配置

```
c = get_config()
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 6666
c.NotebookApp.password = u'sha1:0e422dfccef2:84cfbcbb3ef95872fb8e23be3999c123f862d856'

```
port number (c.NotebookApp.port) 应该与第一步firewall rule里的端口一致。

#Install git
```
conda install git
```
generate SSH key to the git account

```
ssh-keygen -t rsa -C "445072942@qq.com"
vim ~/.ssh/id_rsa.pub
```
copy the ssh key to the github setting > ssh and GPG keys > add new ssh key(paste)

get clone 下载项目
```
git clone git@github.com:BoyMax/deepImagePrior.git
```


#Run jupyter notebook
jupyter notebook --no-browser --port=6666
