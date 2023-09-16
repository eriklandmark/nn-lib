export default class Gradients {
    static get_gradient(actvFunc, lossFunc) {
        let gradientFunc;
        if (actvFunc.name == "softmax" && lossFunc.name == "cross_entropy") {
            gradientFunc = function (input, labels) {
                return input.sub(labels);
            };
        }
        else if (actvFunc.name == "sigmoid" && lossFunc.name == "mean_squared_error") {
            gradientFunc = function (input, labels) {
                return input.sub(labels);
            };
        }
        return gradientFunc;
    }
}
