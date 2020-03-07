"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
var vector_1 = __importDefault(require("./vector"));
var fs = __importStar(require("fs"));
var path = __importStar(require("path"));
var jimp_1 = __importDefault(require("jimp"));
var Dataset = /** @class */ (function () {
    function Dataset() {
        this.data = [];
        this.BATCH_SIZE = 1;
        this.IS_GENERATOR = false;
        this.TOTAL_EXAMPLES = 0;
        this.GENERATOR = function () { };
    }
    Dataset.prototype.size = function () {
        return this.data.length;
    };
    Dataset.prototype.setGenerator = function (gen) {
        this.GENERATOR = gen;
    };
    Dataset.read_image = function (path) {
        return __awaiter(this, void 0, void 0, function () {
            var image, v, i, avg;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, jimp_1.default.read(path)];
                    case 1:
                        image = _a.sent();
                        v = new vector_1.default(image.bitmap.data.length / 4);
                        for (i = 0; i < image.bitmap.data.length; i += 4) {
                            avg = (image.bitmap.data[i] + image.bitmap.data[i + 1] + image.bitmap.data[i + 2]) / 3;
                            v.set(i / 4, avg);
                        }
                        return [2 /*return*/, v];
                }
            });
        });
    };
    Dataset.prototype.loadMnistTrain = function (folderPath, maxExamples) {
        if (maxExamples === void 0) { maxExamples = 60000; }
        this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples);
    };
    Dataset.prototype.loadMnistTest = function (folderPath, maxExamples) {
        if (maxExamples === void 0) { maxExamples = 60000; }
        this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples);
    };
    Dataset.prototype.loadMnist = function (folderPath, imageFileName, labelFileName, maxExamples) {
        var trainFileBuffer = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
        var labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));
        for (var imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            var pixels = [];
            for (var x = 0; x < 28; x++) {
                for (var y = 0; y < 28; y++) {
                    pixels.push(trainFileBuffer[(imageIndex * 28 * 28) + (x + (y * 28)) + 15]);
                }
            }
            var exampleData = new vector_1.default(pixels);
            exampleData = exampleData.div(255);
            var example = {
                data: exampleData,
                label: vector_1.default.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };
            this.data.push(example);
        }
    };
    Dataset.prototype.loadTestData = function (path, maxExamples) {
        if (maxExamples === void 0) { maxExamples = 2100; }
        var data = JSON.parse(fs.readFileSync(path, { encoding: "UTF-8" }));
        for (var imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            var example = {
                data: new vector_1.default(data["features"][imageIndex]),
                label: vector_1.default.toCategorical(data["labels"][imageIndex], 3)
            };
            this.data.push(example);
        }
    };
    Dataset.prototype.getBatch = function (batch) {
        return this.data.slice(batch * this.BATCH_SIZE, batch * this.BATCH_SIZE + this.BATCH_SIZE);
    };
    return Dataset;
}());
exports.default = Dataset;