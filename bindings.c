#include <Python.h>
#include <dlfcn.h>
#include <stdint.h>

/*
 * This file is auto-generated.  It provides Python bindings to the
 * `@export`-annotated functions located in:
 *   - kernels/make_4d_causal_mask.mojo      -> make_4d_causal_mask_invoke()
 *   - kernels/apply_rotary_emb.mojo         -> apply_rotary_emb()
 *
 * The generated Python module is called `bindings` and, when imported, will
 * `dlopen` the shared libraries that result from compiling the respective
 * `.mojo` files (i.e. `make_4d_causal_mask.so` and `apply_rotary_emb.so`).
 *
 *  All errors that occur during shared-library loading or symbol resolution
 *  are surfaced to Python by raising `ImportError` (via `PyErr_Format`).
 */

/* typedefs that mirror the C signatures produced from the Mojo sources */
typedef void (*make_4d_causal_mask_invoke_t)(
    void *in_ptr,
    void *out_ptr,
    unsigned stride,
    unsigned numel);

typedef void (*apply_rotary_emb_t)(
    unsigned s0,
    unsigned s1,
    void *cos_ptr,
    void *position_ids_ptr,
    void *sin_ptr,
    unsigned s4,
    unsigned s5,
    void *q_ptr,
    unsigned s6,
    void *k_ptr,
    void *out0_ptr,
    void *out1_ptr);

/* function-pointer globals initialised at import time */
static make_4d_causal_mask_invoke_t p_make_4d_causal_mask_invoke = NULL;
static apply_rotary_emb_t p_apply_rotary_emb = NULL;

/* Keep the dlopen handles alive for the lifetime of the process. */
static void *handle_mask = NULL;
static void *handle_rotary = NULL;

/* Utility to load a shared library and resolve one symbol.  If any step fails
 * an ImportError is raised and -1 returned. */
static int
load_symbol(const char *lib_name, void **handle_out, const char *sym_name, void **sym_out)
{
    void *handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL);
    if (!handle)
    {
        PyErr_Format(PyExc_ImportError, "Failed to load shared library '%s': %s", lib_name, dlerror());
        return -1;
    }

    dlerror(); /* clear any existing error */
    void *fn = dlsym(handle, sym_name);
    const char *err = dlerror();
    if (err || !fn)
    {
        PyErr_Format(PyExc_ImportError, "Failed to locate symbol '%s' in '%s': %s", sym_name, lib_name, err ? err : "unknown error");
        dlclose(handle);
        return -1;
    }

    *handle_out = handle; /* leak intentionally so functions remain valid */
    *sym_out = fn;
    return 0;
}

/* Resolve all required symbols from the generated shared libraries. */
static int
resolve_all_symbols(void)
{
    if (load_symbol("make_4d_causal_mask.so",
                    &handle_mask,
                    "make_4d_causal_mask_invoke",
                    (void **)&p_make_4d_causal_mask_invoke) < 0)
    {
        return -1; /* Exception already set */
    }

    if (load_symbol("apply_rotary_emb.so",
                    &handle_rotary,
                    "apply_rotary_emb",
                    (void **)&p_apply_rotary_emb) < 0)
    {
        return -1; /* Exception already set */
    }

    return 0;
}

/* ===================== Python wrapper helpers ============================ */

static PyObject *
py_make_4d_causal_mask_invoke(PyObject *self, PyObject *args)
{
    unsigned long long in_ptr_val;
    unsigned long long out_ptr_val;
    unsigned stride;
    unsigned numel;

    if (!PyArg_ParseTuple(args, "KKII",
                          &in_ptr_val,
                          &out_ptr_val,
                          &stride,
                          &numel))
    {
        return NULL; /* TypeError already raised by PyArg_ParseTuple */
    }

    if (!p_make_4d_causal_mask_invoke)
    {
        PyErr_SetString(PyExc_RuntimeError, "make_4d_causal_mask_invoke symbol not resolved");
        return NULL;
    }

    p_make_4d_causal_mask_invoke((void *)(uintptr_t)in_ptr_val,
                                 (void *)(uintptr_t)out_ptr_val,
                                 stride,
                                 numel);

    Py_RETURN_NONE;
}

static PyObject *
py_apply_rotary_emb(PyObject *self, PyObject *args)
{
    unsigned s0, s1, s4, s5, s6;
    unsigned long long cos_ptr_val;
    unsigned long long position_ids_ptr_val;
    unsigned long long sin_ptr_val;
    unsigned long long q_ptr_val;
    unsigned long long k_ptr_val;
    unsigned long long out0_ptr_val;
    unsigned long long out1_ptr_val;

    if (!PyArg_ParseTuple(args,
                          "IIKKKIIKIKKK",
                          &s0,
                          &s1,
                          &cos_ptr_val,
                          &position_ids_ptr_val,
                          &sin_ptr_val,
                          &s4,
                          &s5,
                          &q_ptr_val,
                          &s6,
                          &k_ptr_val,
                          &out0_ptr_val,
                          &out1_ptr_val))
    {
        return NULL;
    }

    if (!p_apply_rotary_emb)
    {
        PyErr_SetString(PyExc_RuntimeError, "apply_rotary_emb symbol not resolved");
        return NULL;
    }

    p_apply_rotary_emb(s0,
                       s1,
                       (void *)(uintptr_t)cos_ptr_val,
                       (void *)(uintptr_t)position_ids_ptr_val,
                       (void *)(uintptr_t)sin_ptr_val,
                       s4,
                       s5,
                       (void *)(uintptr_t)q_ptr_val,
                       s6,
                       (void *)(uintptr_t)k_ptr_val,
                       (void *)(uintptr_t)out0_ptr_val,
                       (void *)(uintptr_t)out1_ptr_val);

    Py_RETURN_NONE;
}

/* ===================== Module definition ================================ */

static PyMethodDef module_methods[] = {
    {"make_4d_causal_mask_invoke", py_make_4d_causal_mask_invoke, METH_VARARGS, "Invoke make_4d_causal_mask_invoke from Mojo kernel."},
    {"apply_rotary_emb", py_apply_rotary_emb, METH_VARARGS, "Invoke apply_rotary_emb from Mojo kernel."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef bindings_module = {
    PyModuleDef_HEAD_INIT,
    "bindings",    /* m_name */
    NULL,          /* m_doc  */
    -1,            /* m_size */
    module_methods /* m_methods */
};

PyMODINIT_FUNC
PyInit_bindings(void)
{
    if (resolve_all_symbols() < 0)
    {
        return NULL; /* ImportError already set */
    }

    return PyModule_Create(&bindings_module);
}
