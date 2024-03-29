var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
export default class Helper {
    static timeit(func, floorIt = true) {
        return new Promise((resolve) => __awaiter(this, void 0, void 0, function* () {
            const startTime = Date.now();
            yield func();
            const duration = (Date.now() - startTime) / 1000.0;
            if (floorIt) {
                resolve(Math.floor(duration));
            }
            else {
                resolve(duration);
            }
        }));
    }
    static timeitSync(func, floorIt = true) {
        const startTime = Date.now();
        func();
        const duration = (Date.now() - startTime) / 1000.0;
        if (floorIt) {
            return Math.floor(duration);
        }
        else {
            return duration;
        }
    }
}
