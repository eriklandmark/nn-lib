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
Else, make sure to have windows-builder-tool, Visual Studio or Xcode installed depending on your operating system.
##### Then:
````
npm install eriklandmark/nn-lib
````

### Example:
##### MNIST training example:
Simple example on how to create a dataset from MNIST, create a model, train and then save the model for use later.
````typescript
let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
dataset.loadMnistTrain(path-to-dir-with-mnist-files)

let layers = [
    new DenseLayer(32,"sigmoid"),
    new DenseLayer(32,"sigmoid"),
    new DenseLayer(32,"sigmoid"),
    new OutputLayer(10,"softmax")
]

let model = new Model(layers)

model.build(28*28, Losses.squared_error_derivative)

async function run() {
    await model.train(dataset, 30, 0.0005)
    model.save("./nn.json")
    console.log("Done")
}

run()
````

### TODO:
* Add proper GPU support.
* Add convolutional layers.
* Try to make matrix multiplication multithreaded via c++ binding.
* Add callbacks to training.
* Make promised-based version of the main functions.