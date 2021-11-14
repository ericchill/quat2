#ifndef QUAT_KERNEL_VER_H
#define QUAT_KERNEL_VER_H 1

#include "parameters.h"

typedef int (*VER_Initialize) (int x, int y, char* Error);
typedef int (*VER_Done)(void);
typedef int (*VER_getline) (unsigned char* line, int y, long xres, ZFlag whichbuf);
typedef int (*VER_check_event) (void);
typedef int (*VER_Change_Name) (const char* s);
typedef void (*VER_eol) (int line);


extern VER_check_event check_event;
extern VER_Initialize Initialize;
extern VER_Done Done;
extern VER_getline QU_getline;
extern VER_check_event check_event;
extern VER_Change_Name Change_Name;
extern VER_eol eol;


#endif /* QUAT_KERNEL_VER_H 1 */
