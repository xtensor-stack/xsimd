git clone https://github.com/marehr/intel-sde-downloader
cd intel-sde-downloader
pip install -r requirements.txt
python ./intel-sde-downloader.py sde-external-8.16.0-2018-01-30-lin.tar.bz2
tar xvf sde-external-8.16.0-2018-01-30-lin.tar.bz2
sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"