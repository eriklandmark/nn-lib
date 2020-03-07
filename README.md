# nn-lib

##### A minimal and pretty lightweight neural network library. 

For no only support dense (fully connected) layers but will add convolution layer. Are also working/planing on to add
GPU support for training and inference, but this will take a while. Because it is written in typescript it only 
runs on a single thread. Which is nice when run on a more simpler setup (like raspberry pi). 

### To install:
##### Before install:
If on linux, run:
````
sudo apt install mesa-common-dev libxi-dev
````
Else, make shore to have windows-builder-tool, Visual Studio or Xcode installed depending on your operating system.
##### Then:
````
npm install eriklandmark/nn-lib
````

### TODO:
* Add propper GPU support.
* Add convolutional layers.
* Try to make matrix multiplication multithreaded via c++ binding.