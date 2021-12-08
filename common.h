#pragma once


#include <ctype.h>
#include <stdio.h>          
#include <string.h>
#include <stdlib.h>
#include <time.h>  /* time, ctime */

               
#define PROGNAME "Quat"
/* Define VERSION through configure.in */
#ifdef VERSION
#define PROGVERSION VERSION 
#else
/* WARNING: The following string MUST have 4 characters!!! */
/* (So use "1.20" instead of "1.2" !! ) */
#define PROGVERSION "2.00"
#endif
#define PROGSUBVERSION ""
/*#define PROGSTATE " development\n(Build: " __DATE__ ", " __TIME__ ")"*/
#define PROGSTATE ""
/*#define COMMENT "Development version - Please do not distribute."*/
#define COMMENT ""

/* define codes for "mode" in WriteINI (quat.c) */
#define PS_OBJ 1
#define PS_VIEW 2
#define PS_COL 4
#define PS_OTHER 8
#define PS_USEFILE 128


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
