import fs from "fs"
import Path from "path";
let ncp = require('ncp').ncp;

console.log("Moving interface files...")
const destPath = "./dist/visualizer/interface"

if(fs.existsSync(destPath)) {
    fs.rmdirSync(destPath)
}

fs.mkdirSync(destPath)

ncp("./src/visualizer/interface", destPath, function (err) {
    if (err) {
        return console.error(err);
    }
    console.log('Done!');
});