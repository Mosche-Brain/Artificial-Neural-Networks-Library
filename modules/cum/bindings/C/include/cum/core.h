#ifndef CUM_CORE_H
#define CUM_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float cumeric_t;
typedef int64_t dim_t;

typedef enum cum_status_t
{
    CUM_STATUS_SUCCESS = 0,
    CUM_STATUS_INVALID_ARGUMENT = 1,
    CUM_STATUS_OUT_OF_MEMORY = 2,
    CUM_STATUS_RUNTIME_ERROR = 3,
    CUM_STATUS_NOT_IMPLEMENTED = 4
} cum_status_t;

typedef enum cum_device_t
{
    CUM_DEVICE_AUTO = 0,
    CUM_DEVICE_CPU = 1,
    CUM_DEVICE_GPU = 2
} cum_device_t;

typedef enum cum_datatype_t
{
    CUM_DATATYPE_FP64 = 0,
    CUM_DATATYPE_FP32 = 1,
    CUM_DATATYPE_FP16 = 2,
    CUM_DATATYPE_BF16 = 3,
    CUM_DATATYPE_FP8 = 4,
    CUM_DATATYPE_S64 = 5,
    CUM_DATATYPE_S32 = 6,
    CUM_DATATYPE_S16 = 7,
    CUM_DATATYPE_S8 = 8,
    CUM_DATATYPE_U64 = 9,
    CUM_DATATYPE_U32 = 10,
    CUM_DATATYPE_U16 = 11,
    CUM_DATATYPE_U8 = 12,
    CUM_DATATYPE_UNDEF = 13
} cum_datatype_t;

typedef enum cum_layout_t
{
    CUM_LAYOUT_ANY = 0,
    CUM_LAYOUT_X = 1,
    CUM_LAYOUT_NC = 2,
    CUM_LAYOUT_OI = 3,
    CUM_LAYOUT_IO = 4,
    CUM_LAYOUT_NCHW = 5,
    CUM_LAYOUT_NHWC = 6,
    CUM_LAYOUT_OIHW = 7,
    CUM_LAYOUT_HWIO = 8,
    CUM_LAYOUT_NCDHW = 9,
    CUM_LAYOUT_NDHWC = 10,
    CUM_LAYOUT_TNC = 11,
    CUM_LAYOUT_NTC = 12,
    CUM_LAYOUT_STRIDED = 13,
    CUM_LAYOUT_UNDEF = 14
} cum_layout_t;

typedef struct cum_event_t cum_event_t;
typedef struct cum_tensor_t cum_tensor_t;

typedef struct cum_shape_t
{
    const dim_t* dims;
    size_t rank;
} cum_shape_t;

cum_status_t cum_init(cum_device_t device);
cum_status_t cum_shutdown(void);
const char* cum_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
