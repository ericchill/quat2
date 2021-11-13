/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997,98 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
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
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>   /* sqrt */
#include "common.h"
#include "colors.h"
#include <stdlib.h> /* labs */
#include <algorithm>

int GetTrueColor(RealPalette& rp, double color, double* r, double* g, double* b);
int DimAndGammaTrueColor(double RR, double GG, double BB, double* R, double* G, double* B,
    double rgam, double ggam, double bgam, double bright);

double weightsum;

int GetTrueColor(RealPalette& rp,
    double color,
    double* r, double* g, double* b) {
    int i;
    double nc;

    nc = 0;
    if (color == 1) {
        color -= 0.000000000001;
    }
    for (i = 0; (nc < color) && (i < rp._nColors); i++) {
        nc += rp._cols[i].weight / weightsum;
    }
    if (nc > color) {
        i -= 1;
        nc -= rp._cols[i].weight / weightsum;
    }
    nc = (color - nc) / (rp._cols[i].weight / weightsum);
    *r = nc * (rp._cols[i].col2[0] - rp._cols[i].col1[0]) + rp._cols[i].col1[0];
    *g = nc * (rp._cols[i].col2[1] - rp._cols[i].col1[1]) + rp._cols[i].col1[1];
    *b = nc * (rp._cols[i].col2[2] - rp._cols[i].col1[2]) + rp._cols[i].col1[2];
    return 0;
}

int DimAndGammaTrueColor(double RR, double GG, double BB,
    double* R, double* G, double* B,
    double rgam, double ggam, double bgam,
    double bright) {
    double dr, dg, db;
    double dbright;

    if (bright > 1) {
        /* this is to get white as maximum brightness
           (for phong) bright=2 is white
        */
        dr = 1 - RR; dg = 1 - GG; db = 1 - BB;
        dbright = bright - 1;
        *R = pow(RR + dbright * dr, rgam);
        *G = pow(GG + dbright * dg, ggam);
        *B = pow(BB + dbright * db, bgam);
    } else {
        *R = pow(bright * RR, rgam);
        *G = pow(bright * GG, ggam);
        *B = pow(bright * BB, bgam);
    }
    return 0;
}

int FindNearestColor(disppal_struct* disppal,
    unsigned char r, unsigned char g, unsigned char b) {
    long dist1, dist, mindist;
    int i, index = 0, rshift, gshift, bshift;

    mindist = 200000;
    rshift = 8 - disppal->rdepth;
    gshift = 8 - disppal->gdepth;
    bshift = 8 - disppal->bdepth;
    dist = 1;
    for (i = 0; (i < disppal->maxcol) && (dist != 0); i++) {
        dist = 0;
        dist1 = (disppal->cols[i].r << rshift) - r;
        dist += labs(dist1); /*pow(fabs(dist1),0.6);*/
        dist1 = (disppal->cols[i].g << gshift) - g;
        dist += labs(dist1); /*pow(fabs(dist1),0.8)*/;
        dist1 = (disppal->cols[i].b << bshift) - b;
        dist += labs(dist1); /*pow(fabs(dist1),0.6); */
        if (dist < mindist) {
            mindist = dist;
            index = i;
        }
    }
    return index;
}

int CalcWeightsum(RealPalette& realpal) {
    /* Sum the weights in realpal */
    weightsum = 0;
    for (size_t i = 0; i < realpal._nColors; i++) {
        weightsum += realpal._cols[i].weight;
    }
    return 0;
}

int CreateDispPal(disppal_struct* disppal, RealPalette& realpal, int maxcol, double phongmax, int rdepth, int gdepth, int bdepth) {
    int j, requiredcols, maxdepth, maxbright, oswcols;
    int disppos;
    double weight;
    double r, g, b, rr, gg, bb;
    int rncol, gncol, bncol;

    /* how many colours did the user specify at least ? */
    requiredcols = 0;
    weight = realpal._cols[0].weight;
    oswcols = 1;
    for (size_t i = 0; i < realpal._nColors; i++) {
        if ((realpal._cols[i].col1[0] == realpal._cols[i].col2[0])
            && (realpal._cols[i].col1[1] == realpal._cols[i].col2[1])
            && (realpal._cols[i].col1[2] == realpal._cols[i].col2[2])) {
            requiredcols++;
            if (weight != realpal._cols[i].weight) {
                oswcols = 0;
            }
        } else {
            requiredcols += 2;
            oswcols = 0;
        }
    }
    /* how many brights fit into pal of 'maxcol' colors ? */
    disppal->brightnum = static_cast<int>(floor(sqrt(disppal->btoc * maxcol)));
    /* what is the maximum color-depth of r,g,b ? */
    maxdepth = 0;
    maxdepth = std::max(rdepth, maxdepth);
    maxdepth = std::max(gdepth, maxdepth);
    maxdepth = std::max(bdepth, maxdepth);
    if (oswcols == 1) {
        disppal->brightnum = maxcol / requiredcols;
    }
    /* is it possible to achieve 'brightnum' brights ? */
    maxbright = 1 << maxdepth;
    if (maxbright < disppal->brightnum) {
        disppal->brightnum = maxbright;   /* No. */
    }
    disppal->_nColors = (int)((double)maxcol / (double)disppal->brightnum);
    if (oswcols == 1 && disppal->_nColors > requiredcols) {
        disppal->_nColors = requiredcols;
    }
    /* It could all be so easy, if the user wasn't able to declare more */
    /* colours than we can display */
    if (disppal->_nColors < requiredcols) {
        return -1;
    }
    if (disppal->brightnum < 7) {
        return -1;
    }
    disppal->maxcol = disppal->_nColors * disppal->brightnum;

    CalcWeightsum(realpal);

    rncol = 1 << rdepth;  /* number of reds */
    gncol = 1 << gdepth;  /* number of greens */
    bncol = 1 << bdepth;  /* number of blues */

    disppos = 0;
    /* Go through every colour */
    for (size_t i = 0; i < disppal->_nColors; i++) {
        if (disppal->_nColors == 1) {
            GetTrueColor(realpal, 0.5, &r, &g, &b);
        } else {
            GetTrueColor(realpal, (double)i / (double)(disppal->_nColors - 1), &r, &g, &b);
        }
        /* Go through every possible brightness */
        for (j = 0; j < disppal->brightnum - 4; j++)  {
            DimAndGammaTrueColor(r, g, b, &rr, &gg, &bb, GAMMA, GAMMA, GAMMA, (double)j / (double)(disppal->brightnum - 5));
            disppal->cols[disppos].r = (int)((double)(rncol - 1) * rr);
            disppal->cols[disppos].g = (int)((double)(gncol - 1) * gg);
            disppal->cols[disppos].b = (int)((double)(bncol - 1) * bb);
            disppos++;
        }
        /* create phong colours */
        for (j = 0; j < 4; j++) {
            DimAndGammaTrueColor(r, g, b, &rr, &gg, &bb, 1.0, 1.0, 1.0, 1.0 + phongmax * (double)(j + 1) / 4.0);
            disppal->cols[disppos].r = (int)((double)(rncol - 1) * rr);
            disppal->cols[disppos].g = (int)((double)(gncol - 1) * gg);
            disppal->cols[disppos].b = (int)((double)(bncol - 1) * bb);
            disppos++;
        }
    }
    return 0;
}

int PixelvaluePalMode(int x1, int x2, int colmax, int brightmax, unsigned char* line, float* CBuf, float* BBuf) {
    int i;
    /* colmax+1 = _nColors, brightmax dto.  */

    for (i = x1; i <= x2; i++) {
        line[i] = static_cast<int>(floor(colmax * CBuf[i]) * (brightmax + 1));
        if (BBuf[i] <= 1) {
            line[i] += static_cast<int>(floor((brightmax - 4) * BBuf[i]));
        } else {
            line[i] += brightmax - 4 + static_cast<int>(floor(4.0 * (BBuf[i] - 1.0)));
        }
    }
    return 0;
}

int PixelvalueTrueMode(int x1, int x2, int rmax, int gmax, int bmax,
    RealPalette& realpal, unsigned char* line, float* CBuf, float* BBuf) {
    int i;
    double r, g, b;
    int ir, ig, ib;

    for (i = x1; i <= x2; i++) {
        if (BBuf[i] > 0.0001) {
            GetTrueColor(realpal, CBuf[i], &r, &g, &b);
            DimAndGammaTrueColor(r, g, b, &r, &g, &b, GAMMA, GAMMA, GAMMA, BBuf[i]);
            ir = static_cast<int>(floor(r * (float)rmax));
            ig = static_cast<int>(floor(g * (float)gmax));
            ib = static_cast<int>(floor(b * (float)bmax));
            line[3 * i] = ir;
            line[3 * i + 1] = ig;
            line[3 * i + 2] = ib;
        } else {
            line[3 * i] = 0; 
            line[3 * i + 1] = 0; 
            line[3 * i + 2] = 0;
        }
    }
    return 0;
}


