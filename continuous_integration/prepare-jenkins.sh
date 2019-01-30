catkin config --cmake-args -DCATKIN_ENABLE_TESTING=ON -DCMAKE_CXX_STANDARD=14

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  echo "**** Installing additional necessary packages. ****"
  sudo apt-get install -y libbullet-dev
elif [[ "$unamestr" == 'Darwin' ]]; then
  brew install bullet
else
  echo "Platform $unamestr is not supported!"
  exit -1
fi
