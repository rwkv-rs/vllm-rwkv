#include "core/registration.h"

PyMODINIT_FUNC CONCAT(PyInit_, TORCH_EXTENSION_NAME)() {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, STRINGIFY(TORCH_EXTENSION_NAME), nullptr, 0, nullptr};
  PyObject* module = PyModule_Create(&module_def);
  if (module == nullptr) {
    return nullptr;
  }
#ifdef VLLM_RWKV_ONLY_BUILD
  constexpr int rwkv_only_build = 1;
#else
  constexpr int rwkv_only_build = 0;
#endif
  if (PyModule_AddIntConstant(module, "_rwkv_only_build", rwkv_only_build) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
