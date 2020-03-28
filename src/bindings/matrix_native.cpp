#include <napi.h>
#include <vector>

using namespace std;
using namespace Napi;

Array dot(const CallbackInfo &info) {
    Env env = info.Env();
    if (info.Length() < 6) {
        TypeError::New(env, "Too few arguments").ThrowAsJavaScriptException();
    }

    Array a = info[0].As<Array>();
    Array b = info[1].As<Array>();
    int a_r = info[2].As<Number>().Int32Value();
    int a_c = info[3].As<Number>().Int32Value();
    int b_r = info[4].As<Number>().Int32Value();
    int b_c = info[4].As<Number>().Int32Value();
    Array c = Array::New(env, a_r);

    //TypedArrayOf<float> f = first.Get(uint32_t(0)).As<TypedArrayOf<float>>();

    if (a_c != b_r) {
        TypeError::New(env, "Wrong dimension on matrix A or B").ThrowAsJavaScriptException();
    }

    /*

    for (int i = 0; i < a_r; i++) {
        TypedArrayOf t = TypedArrayOf::New(env, b_c);
        for (int j = 0; j < b_c; j++) {
            //t.Data()[j] = 0;
            TypedArrayOf<float> f = a.Get(uint32_t(i)).As<TypedArrayOf<float>>();
            for (int k = 0; k < b_r; k++) {
                TypedArrayOf<float> b_v = b.Get(uint32_t(k)).As<TypedArrayOf<float>>();
                t[j] += f[k] * b_v[j];
            }
        }
        c.Set(i, t);
    }*/

    return c;
}

Object Init(Env env, Object exports) {
    exports.Set("mm", Function::New(env, dot));

    return exports;
}

NODE_API_MODULE(matrix_native, Init)