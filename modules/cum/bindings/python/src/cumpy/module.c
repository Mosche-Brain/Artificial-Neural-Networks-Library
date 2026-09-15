#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cum/cum.h>

static PyObject* cumpy_error(cum_status_t status)
{
    const char* message = cum_last_error();
    if(message == NULL || message[0] == '\0')
        message = "cum operation failed";

    if(status == CUM_STATUS_INVALID_ARGUMENT)
        PyErr_SetString(PyExc_ValueError, message);
    else if(status == CUM_STATUS_OUT_OF_MEMORY)
        PyErr_SetString(PyExc_MemoryError, message);
    else
        PyErr_SetString(PyExc_RuntimeError, message);
    return NULL;
}

static PyObject* cumpy_init(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int device = CUM_DEVICE_AUTO;
    static char* keywords[] = {"device", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:init", keywords, &device))
        return NULL;
    if(cum_init((cum_device_t)device) != CUM_STATUS_SUCCESS)
        return cumpy_error(CUM_STATUS_RUNTIME_ERROR);
    Py_RETURN_NONE;
}

static PyObject* cumpy_shutdown(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, ":shutdown"))
        return NULL;
    if(cum_shutdown() != CUM_STATUS_SUCCESS)
        return cumpy_error(CUM_STATUS_RUNTIME_ERROR);
    Py_RETURN_NONE;
}

static PyObject* cumpy_last_error(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, ":last_error"))
        return NULL;
    return PyUnicode_FromString(cum_last_error());
}

static PyMethodDef cumpy_methods[] = {
    {"init", (PyCFunction)cumpy_init, METH_VARARGS | METH_KEYWORDS, "Initialize cum runtime."},
    {"shutdown", cumpy_shutdown, METH_VARARGS, "Shut down cum runtime."},
    {"last_error", cumpy_last_error, METH_VARARGS, "Return the last cum error."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cumpy_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Low-level Python bindings for cum.",
    -1,
    cumpy_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&cumpy_module);
}
