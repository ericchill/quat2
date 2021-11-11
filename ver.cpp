/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* Further work 2004 by Eric C. Hill (eric@stochastic.com) */
/* */
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */
/* */
/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */
/* */
/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#include "common.h"
#include "quat.h"
#include "ver.h"

/* These variables (functions) have to be declared in the special
   version files !
   use an extern statement like "extern VER_Done Done;" */
VER_ReturnVideoInfo ReturnVideoInfo = DUMMY_ReturnVideoInfo;
VER_SetColors SetColors = DUMMY_SetColors;
VER_Initialize Initialize = DUMMY_Initialize;
VER_Done Done = DUMMY_Done;
VER_Change_Name Change_Name = DUMMY_Change_Name;
VER_Debug Debug = DUMMY_Debug;
VER_check_event check_event = DUMMY_check_event;
VER_update_bitmap update_bitmap = DUMMY_update_bitmap;
VER_getline QU_getline = DUMMY_getline;
VER_eol eol = DUMMY_eol;


/* DUMMY functions which do nothing / Available to all versions */
int DUMMY_ReturnVideoInfo(struct vidinfo_struct* v) {
    return 0;
}
int DUMMY_SetColors(struct disppal_struct* d) {
    return 0;
}
int DUMMY_Initialize(int x, int y, char* c) {
    return 0;
}
int DUMMY_Done(void) {
    return 0;
}
int DUMMY_Change_Name(const char* a) {
    return 0; 
}
void DUMMY_Debug(const char* a) {
    return; 
}
void DUMMY_eol(int a) {
    return;
}

int DUMMY_check_event(void) {
    return 0; 
}

int DUMMY_update_bitmap(long a, long b, long c, 
			int d, unsigned char *e, int f) {
    return 0; 
}
int DUMMY_getline(unsigned char *a, int b, long c, ZFlag d) {
    return 0; 
}

