# nn-lib

##### A minimal and very lightweight neural network library. 

For no only support dense (fully connected) layers but will add convolution layer. Are also working/planing on to add
GPU support for training and inference, but this will take a while. Because it is written in typescript it only 
runs on a single thread. Which is nice when you want to run on a less powerful setup (like raspberry pi). 

### To install:
##### Before install:
If on linux, run:
````
sudo apt install pkg-config mesa-common-dev libxi-dev
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
dataset.loadMnistTrain(path-to-dir-with-mnist-files, 1000, true)

let layers = [
    new DenseLayer(32, new Sigmoid()),
    new DenseLayer(32, new Sigmoid()),
    new DropoutLayer(0.25),
    new DenseLayer(32, new Sigmoid()),
    new DropoutLayer(0.20),
    new OutputLayer(10, new Softmax())
]

let model = new Model(layers)

model.build([28*28], MeanSquaredError, StochasticGradientDescent)

async function run() {
    await model.train(dataset, 30, 0.0005, true)
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
* Maybe create documentation if needed.

#### Benchmark Scores:
 - **208910** = AMD Ryzen Threadripper 1920X
 - **134751** = Intel Core i5-6300U
 - **108919** = AMD FX-8350
 - **42005** = Raspberry Pi 4B (4G) - Non overclocked
 - **13273** = Raspberry Pi 3B - Non overclocked