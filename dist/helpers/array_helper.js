"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class ArrayHelper {
    static shuffle(array) {
        let currentIndex = array.length, temporaryValue, randomIndex;
        while (0 !== currentIndex) {
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;
            temporaryValue = array[currentIndex];
            array[currentIndex] = array[randomIndex];
            array[randomIndex] = temporaryValue;
        }
        return array;
    }
    static flatten(array) {
        const new_array = [];
        const flatten_rec = (a) => {
            for (let item of a) {
                if (Array.isArray(item)) {
                    flatten_rec(item);
                }
                else {
                    new_array.push(item);
                }
            }
        };
        flatten_rec(array);
        return new_array;
    }
    static delete_doublets(array) {
        return array.reduce((acc, el) => {
            if (!acc.includes(el))
                acc.push(el);
            return acc;
        }, []);
    }
}
exports.default = ArrayHelper;
