{
  "targets": [
    {
      "target_name": "matrix_native",
      "sources": [
        "./src/bindings/matrix_native.cpp"
      ],
     'include_dirs': [
         "<!@(node -p \"require('node-addon-api').include\")"
     ],
     'libraries': [],
     'dependencies': [
         "<!(node -p \"require('node-addon-api').gyp\")"
     ],
     'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ]
    }
  ]
}