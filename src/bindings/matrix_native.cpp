#include <napi.h>

std::string matrix::hello(){
  return "Hello World";
}
Napi::String matrix::HelloWrapped(const Napi::CallbackInfo& info)
{
  Napi::Env env = info.Env();
  Napi::String returnValue = Napi::String::New(env, matrix::hello());

  return returnValue;
}
Napi::Object matrix::Init(Napi::Env env, Napi::Object exports)
{
  exports.Set("hello", Napi::Function::New(env, matrix::HelloWrapped));

  return exports;
}