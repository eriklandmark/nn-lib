import CrossEntropy from "./cross_entropy";
import MeanSquaredError from "./mean_squared_error";
export default class Losses {
    static fromName(name) {
        switch (name) {
            case "cross_entropy": return CrossEntropy;
            case "mean_squared_error": return MeanSquaredError;
        }
    }
}
