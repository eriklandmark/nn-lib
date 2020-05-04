export default class ArrayHelper {

    public static shuffle(array: any[]) {
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

    public static flatten(array: any[]): any[] {
        const new_array: any[] = []
        const flatten_rec = (a) => {
            for(let item of a) {
                if(Array.isArray(item)) {
                    flatten_rec(item)
                } else {
                    new_array.push(item)
                }
            }
        }

        flatten_rec(array)

        return new_array
    }

    public static delete_doublets(array: any[]): any[] {
        return array.reduce((acc, el) => {
            if(!acc.includes(el))
                acc.push(el)
            return acc
        },[])
    }
}