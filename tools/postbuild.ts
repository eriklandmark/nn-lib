import fs from "fs"
import Path from "path";
const ncp = require('ncp').ncp;
const rimraf = require("rimraf");
const destPath = "./dist/visualizer/interface"

if(fs.existsSync(destPath)) {
    console.log("Files already exists.. deleting..")
    rimraf.sync(destPath);
    console.log("Done!")
}

console.log("Moving interface files...")

ncp("./src/visualizer/interface", destPath, function (err) {
    if (err) {
        return console.error(err);
    }
    console.log('Done!');
});