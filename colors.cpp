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


#include <math.h>   /* sqrt */
#include <stdlib.h> /* labs */
#include <algorithm>
#include "common.h"

#include "colors.h"


int dimAndGammaTrueColor(double RR, double GG, double BB,
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


col_struct::col_struct(const json::value jv) {
    const json::object obj = jv.as_object();
    weight = obj.at("weight").to_number<double>();
    const json::array& col1arr = obj.at("col1").as_array();
    for (size_t i = 0; i < 3; i++) {
        col1[i] = col1arr.at(i).to_number<double>();
    }
    const json::array& col2arr = obj.at("col2").as_array();
    for (size_t i = 0; i < 3; i++) {
        col2[i] = col2arr.at(i).to_number<double>();
    }

}
json::value col_struct::toJSON() const {
    return {
        { "weight", weight },
        { "col1", { col1[0], col1[1], col1[2] }},
        { "col2", { col2[0], col2[1], col2[2] }}
    };
}


RealPalette::RealPalette(const RealPalette& r) {
    for (size_t i = 0; i < r._nColors; i++) {
        _cols[i] = r._cols[i];
    }
    _nColors = r._nColors;
    _weightSum = r._weightSum;
}

void RealPalette::reset() {
    _cols[0].col1[0] = 0;
    _cols[0].col1[1] = 0;
    _cols[0].col1[2] = 1;
    _cols[0].col2[0] = 1;
    _cols[0].col2[1] = 0;
    _cols[0].col2[2] = 0;
    _cols[0].weight = 1;
    _nColors = 1;
}

RealPalette::RealPalette(const json::value& jv) {
    json::array colors = jv.as_object().at("colors").as_array();
    _weightSum = 0;
    for (size_t i = 0; i < colors.size(); i++) {
        _cols[i] = col_struct(colors.at(i));
        _weightSum += _cols[i].weight;
    }
    _nColors = colors.size();

}
json::value RealPalette::toJSON() const {
    json::array colors;
    for (size_t i = 0; i < _nColors; i++) {
        colors.push_back(_cols[i].toJSON());
    }
    return { { "colors", colors } };

}

RealPalette tag_invoke(const json::value_to_tag< RealPalette >&, json::value const& jv) {
    return RealPalette(jv);
}

double RealPalette::computeWeightSum() {
    _weightSum = 0;
    for (size_t i = 0; i < _nColors; i++) {
        _weightSum += _cols[i].weight;
    }
    return _weightSum;
}

void RealPalette::getTrueColor(
    double color,
    double* r, double* g, double* b) {
    int i;
 
    if (color == 1) {
        color -= 0.000000000001;
    }
    double nc = 0;
    for (i = 0; (nc < color) && (i < _nColors); i++) {
        nc += _cols[i].weight / _weightSum;
    }
    if (nc > color) {
        i -= 1;
        nc -= _cols[i].weight / _weightSum;
    }
    nc = (color - nc) / (_cols[i].weight / _weightSum);
    *r = nc * (_cols[i].col2[0] - _cols[i].col1[0]) + _cols[i].col1[0];
    *g = nc * (_cols[i].col2[1] - _cols[i].col1[1]) + _cols[i].col1[1];
    *b = nc * (_cols[i].col2[2] - _cols[i].col1[2]) + _cols[i].col1[2];
}

int RealPalette::pixelValue(
    int x1, int x2,
    int rmax, int gmax, int bmax,
    unsigned char* line,
    float* cBuf,
    float* bBuf) {
    int i;
    double r, g, b;
    int ir, ig, ib;

    for (i = x1; i <= x2; i++) {
        if (bBuf[i] > 0.0001) {
            getTrueColor(cBuf[i], &r, &g, &b);
            dimAndGammaTrueColor(r, g, b, &r, &g, &b, GAMMA, GAMMA, GAMMA, bBuf[i]);
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


