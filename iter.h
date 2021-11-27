#pragma once

#include "qmath.h"

struct iter_struct {
	Quat xstart;
	Quat c;
	double bailout;
	int maxiter;
	int exactiter;
	Quat p[4];
	Quat *orbit;
};


#ifndef __NVCC__

#include "parameters.h"

int iterate_0(iter_struct*);
   /* formula: x[n+1] = x[n]^2 - c */
   /* xstart: 4 start iteration values */
   /* c: same values as in x^2-c */
   /* bailout: square of the value at which iteration is seen to be divergent */
   /* maxiter: number of iterations after that iteration is seen to be convergent */
   /* exactiter: !=0: bailout is ignored, iteration done up to maxiter. If orbit gets */
   /*            too large for double-type, functions returns -1 */
   /* returns number of iterations done before bailout */
   /* fills in orbit: array of points (MUST hold up to maxiter points!) that make */
   /*                 the orbit */
int iterate_0_no_orbit(iter_struct*);

int iternorm_0(iter_struct* is, Vec3& norm);
   /* formula: x[n+1] = x[n]^2 - c */
   /* iterates a point and calculates the normal vector "norm" at this point */
   /* parameters same as "iterate_0" */
   /* returns number of iterations done before bailout */

int iterate_1(iter_struct*);
   /* formula: x[n+1] = c*x[n]*(1-x[n]^2) */
   /* xstart: 4 start iteration values */
   /* bailout: square of the value at which iteration is seen to be divergent */
   /* maxiter: number of iterations after that iteration is seen to be convergent */
   /* exactiter: !=0: bailout is ignored, iteration done up to maxiter. If orbit gets */
   /*            too large for double-type, functions returns -1 */
   /* returns number of iterations done before bailout */
   /* fills in orbit: array of points (MUST hold up to maxiter points!) that make */
   /*                 the orbit */
int iterate_1_no_orbit(iter_struct*);
int iternorm_1(iter_struct*, Vec3& norm);
int iterate_2(iter_struct*);
int iterate_3(iter_struct*);
int iterate_3_no_orbit(iter_struct*);
int iterate_4(iter_struct*);
int iterate_4_no_orbit(iter_struct*);
int iterate_bulb(iter_struct*);


 
float brightpoint(
		  long x,
		  int y,
		  float *LBuf,
		  calc_struct *c);
   /* calculate the brightness value of a pixel (0.0 ... 1.0) */
   /* x: x coordinate of pixel on screen */
   /* y: the number of the "scan"line to calculate */
   /* LBuf: depth information for the pixel */
   /* fields in calc_struct: */
   /* base: normalized base */
   /* sbase: specially normalized base */
   /* f, v: structures with the fractal and view information */
   /* returns brightness value */

#endif // __NVCC__
