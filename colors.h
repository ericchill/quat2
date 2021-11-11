#pragma once

#include "parameters.h"

int CreateDispPal(disppal_struct *disppal, RealPalette& realpal, int maxcol, double phongmax, int rdepth, int gdepth, int bdepth);
int PixelvaluePalMode(int x1, int x2, int colmax, int brightmax, unsigned char *line, float *CBuf, float *BBuf);
int PixelvalueTrueMode(int x1, int x2, int rmax, int gmax, int bmax, 
   RealPalette& realpal, unsigned char *line, float *CBuf, float *BBuf);
int CalcWeightsum(RealPalette& realpal);
int FindNearestColor(disppal_struct *disppal, unsigned char r, unsigned char g, unsigned char b);
