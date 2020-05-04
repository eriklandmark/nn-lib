import Visualizer from "../src/visualizer/visualizer";
const vis = new Visualizer("./model/backlog.json")
vis.PORT = 3000
vis.run()