const Visualizer = require("nn-lib-visualizer/dist/visualizer").default;
const vis = new Visualizer([
    "./models/model_test_0/backlog.json",
    "./models/model_test_1/backlog.json",
    "./models/model_test_2/backlog.json",
    "./models/model_test_3/backlog.json"
])
vis.PORT = 3500
vis.run()