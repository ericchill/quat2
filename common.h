#pragma once

#pragma warning( disable : 4324 )

#include <ctype.h>
#include <stdio.h>          
#include <string.h>
#include <time.h>  /* time, ctime */
#include <exception>
#include <string>
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
               
#define PROGNAME "Quat"

/* WARNING: The following string MUST have 4 characters!!! */
/* (So use "1.20" instead of "1.2" !! ) */
#define PROGVERSION "2.00"

#define PROGSUBVERSION ""
/*#define PROGSTATE " development\n(Build: " __DATE__ ", " __TIME__ ")"*/
#define PROGSTATE ""
/*#define COMMENT "Development version - Please do not distribute."*/
#define COMMENT ""


template<typename T>
T clamp(T v, T min, T max) {
    if (v < min) {
        return min;
    } else if (v > max) {
        return max;
    } else {
        return v;
    }
}


inline long threeBytesToLong(unsigned char* bytes) {
    return 
        bytes[0] << 16
        | bytes[1] << 8
        | bytes[2];
}


template<typename T>
inline void fillArray(T* array, const T& value, size_t nElems) {
    for (size_t i = 0; i < nElems; i++) {
        array[i] = value;
    }
}


template<typename T>
inline void copyArray(T* dst, const T* src, size_t nElems) {
    memcpy(dst, src, nElems * sizeof(T));
}


class QuatException : public std::exception {
    std::string _msg;
public:
    QuatException() {}
    QuatException(const std::string& msg) : _msg(msg) {}
    const char* what() const { return _msg.c_str(); }
};



#ifdef __NVCC__
#define CUDA_CALLABLE __host__ __device__
#define GPU_ONLY __device__
#define CPU_ONLY __host__
#else
#define CUDA_CALLABLE
#define GPU_ONLY
#define CPU_ONLY
#endif
