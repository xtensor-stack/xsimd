#git clone https://github.com/marehr/intel-sde-downloader
#cd intel-sde-downloader
#pip install -r requirements.txt
#python ./intel-sde-downloader.py sde-external-8.35.0-2019-03-11-lin.tar.bz2
#wget http://software.intel.com/content/dam/develop/external/us/en/protected/sde-external-8.50.0-2020-03-26-lin.tar.bz2

curl 'https://software.intel.com/content/dam/develop/external/us/en/documents/sde-external-8.56.0-2020-07-05-lin.tar.bz2' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Connection: keep-alive' -H 'Referer: https://software.intel.com/content/www/us/en/develop/articles/pre-release-license-agreement-for-intel-software-development-emulator-accept-end-user-license-agreement-and-download.html' -H 'Cookie: AWSALB=xdEyuBs+g/QOsC4iy4IiI9PDOuiExTuglYNXQzDO4xoupxFgOsrOCkq4CfnhHc7XBY2SbhWsjn83d6MtgpFxtcCQvBqy9VPBWg+W885Kz5aCj6uHJlTH7gS+t6NS; AWSALBCORS=xdEyuBs+g/QOsC4iy4IiI9PDOuiExTuglYNXQzDO4xoupxFgOsrOCkq4CfnhHc7XBY2SbhWsjn83d6MtgpFxtcCQvBqy9VPBWg+W885Kz5aCj6uHJlTH7gS+t6NS; ref=; OldBrowsersCookie=Cookie for old browser popup message' -H 'Upgrade-Insecure-Requests: 1' --output sde-external-8.56.0-2020-07-05-lin.tar.bz2

tar xvf sde-external-8.56.0-2020-07-05-lin.tar.bz2
sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
