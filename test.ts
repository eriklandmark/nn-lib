import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import * as fs from "fs";
import {Example} from "./lib/dataset";
import MatrixHelper from "./lib/matrix_helper";

const x = new Vector([1,2,3, 6])
const y = new Vector([1,3,2, 8])


MatrixHelper.linear_least_squares(x, y)
