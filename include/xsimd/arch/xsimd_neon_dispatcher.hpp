/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
* Copyright (c) Serge Guelton                                              *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Few macros to select neon intrinsic function based on the scalar type
#define NEON_DISPATCHER_BINARY(U8, S8, U16, S16, U32, S32, U64, S64, F32, type, arg1, arg2, result)\
    if (std::is_same<type, uint8_t>::value) {\
        result = U8(arg1, arg2);\
    } else if(std::is_same<type, int8_t>::value) {\
        result = S8(arg1, arg2);\
    } else if(std::is_same<type, uint16_t>::value) {\
        result = U16(arg1, arg2);\
    } else if(std::is_same<type, int16_t>::value) {\
        result = S16(arg1, arg2);\
    } else if(std::is_same<type, uint32_t>::value) {\
        result = U32(arg1, arg2);\
    } else if(std::is_same<type, int32_t>::value) {\
        result = S32(arg1, arg2);\
    } else if(std::is_same<type, uint64_t>::value) {\
        result = U64(arg1, arg2);\
    } else if(std::is_same<type, int64_t>::value) {\
        result = S64(arg1, arg2);\
    } else if(std::is_same<type, float32_t>::value) {\
        result = F32(arg1, arg2);\
    } else {\
        assert(false && "unsupported type");\
    }

#define NEON_DISPATCHER_BINARY_EXCLUDE_64(U8, S8, U16, S16, U32, S32, F32, type, arg1, arg2, result)\
    if (std::is_same<type, uint8_t>::value) {\
        result = U8(arg1, arg2);\
    } else if(std::is_same<type, int8_t>::value) {\
        result = S8(arg1, arg2);\
    } else if(std::is_same<type, uint16_t>::value) {\
        result = U16(arg1, arg2);\
    } else if(std::is_same<type, int16_t>::value) {\
        result = S16(arg1, arg2);\
    } else if(std::is_same<type, uint32_t>::value) {\
        result = U32(arg1, arg2);\
    } else if(std::is_same<type, int32_t>::value) {\
        result = S32(arg1, arg2);\
    } else if(std::is_same<type, float32_t>::value) {\
        result = F32(arg1, arg2);\
    } else {\
        assert(false && "unsupported type");\
    }

#define NEON_DISPATCHER_UNARY(U8, S8, U16, S16, U32, S32, U64, S64, F32, type, arg, result)\
    if (std::is_same<type, uint8_t>::value) {\
        result = U8(arg);\
    } else if(std::is_same<type, int8_t>::value) {\
        result = S8(arg);\
    } else if(std::is_same<type, uint16_t>::value) {\
        result = U16(arg);\
    } else if(std::is_same<type, int16_t>::value) {\
        result = S16(arg);\
    } else if(std::is_same<type, uint32_t>::value) {\
        result = U32(arg);\
    } else if(std::is_same<type, int32_t>::value) {\
        result = S32(arg);\
    } else if(std::is_same<type, uint64_t>::value) {\
        result = U64(arg);\
    } else if(std::is_same<type, int64_t>::value) {\
        result = S64(arg);\
    } else if(std::is_same<type, float32_t>::value) {\
        result = F32(arg);\
    } else {\
        assert(false && "unsupported type");\
    }

#define NEON_DISPATCHER_UNARY_EXCLUDE_64(U8, S8, U16, S16, U32, S32, F32, type, arg, result)\
    if (std::is_same<type, uint8_t>::value) {\
        result = U8(arg);\
    } else if(std::is_same<type, int8_t>::value) {\
        result = S8(arg);\
    } else if(std::is_same<type, uint16_t>::value) {\
        result = U16(arg);\
    } else if(std::is_same<type, int16_t>::value) {\
        result = S16(arg);\
    } else if(std::is_same<type, uint32_t>::value) {\
        result = U32(arg);\
    } else if(std::is_same<type, int32_t>::value) {\
        result = S32(arg);\
    } else if(std::is_same<type, float32_t>::value) {\
        result = F32(arg);\
    } else {\
        assert(false && "unsupported type");\
    }

#define NEON_DISPATCHER_SELECT(U8, S8, U16, S16, U32, S32, U64, S64, F32, type, cond, arg1, arg2, result)\
    if (std::is_same<type, uint8_t>::value) {\
        result = U8(cond, arg1, arg2);\
    } else if(std::is_same<type, int8_t>::value) {\
        result = S8(cond, arg1, arg2);\
    } else if(std::is_same<type, uint16_t>::value) {\
        result = U16(cond, arg1, arg2);\
    } else if(std::is_same<type, int16_t>::value) {\
        result = S16(cond, arg1, arg2);\
    } else if(std::is_same<type, uint32_t>::value) {\
        result = U32(cond, arg1, arg2);\
    } else if(std::is_same<type, int32_t>::value) {\
        result = S32(cond, arg1, arg2);\
    } else if(std::is_same<type, uint64_t>::value) {\
        result = U64(cond, arg1, arg2);\
    } else if(std::is_same<type, int64_t>::value) {\
        result = S64(cond, arg1, arg2);\
    } else if(std::is_same<type, float32_t>::value) {\
        result = F32(cond, arg1, arg2);\
    } else {\
        assert(false && "unsupported type");\
    }

