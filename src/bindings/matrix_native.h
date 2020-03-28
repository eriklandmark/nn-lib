#include <napi.h>
#include <vector>

using namespace std;

namespace matrix_native {
  std::string hello();
  void mm(vector<vector<double>> a, vector<vector<double>> b, vector<vector<double>> c);
  Napi::String HelloWrapped(const Napi::CallbackInfo& info);
  Napi::Number MMWrapped(const Napi::CallbackInfo& info);
  Napi::Object Init(Napi::Env env, Napi::Object exports);
}