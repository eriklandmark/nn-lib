import Model from "./model";
import Dataset from "./dataset";
export default class ModelNode {
    model: Model;
    train_settings: any;
    constructor(dataset: Dataset, eval_dataset?: Dataset);
}
