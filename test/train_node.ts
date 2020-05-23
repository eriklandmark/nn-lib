import Dataset from "../src/dataset";
import ModelNode from "../src/model_node";

const dataset = new Dataset()
dataset.loadMnistTrain("./dataset/mnist", 1000, false)

dataset.BATCH_SIZE = 50
dataset.TOTAL_EXAMPLES = dataset.size()
dataset.IS_GENERATOR = false
dataset.DATA_SHAPE = [28, 28, 1]
dataset.VERBOSE = false

const eval_dataset = new Dataset()
eval_dataset.loadMnistTest("./dataset/mnist", 200, false)

eval_dataset.BATCH_SIZE = 200
eval_dataset.TOTAL_EXAMPLES = dataset.size()
eval_dataset.IS_GENERATOR = false
eval_dataset.DATA_SHAPE = [28, 28, 1]
eval_dataset.VERBOSE = false

new ModelNode(dataset, eval_dataset)