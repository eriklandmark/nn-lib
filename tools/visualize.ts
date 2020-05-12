import Visualizer from "../src/visualizer/visualizer";
const vis = new Visualizer(["./model_best_mnist/backlog.json", "./model/backlog.json", "../nn-lib-test/model/backlog.json"])
vis.PORT = 3000
vis.run()