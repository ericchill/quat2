#pragma once

#include "json.h"


struct col_struct {
    double weight;   /* Weight in a palette from 0.0 to 1.0 */
                     /* 0 ... single color */
                     /* >0 ... smoothly shaded to next color in */
                     /* palette, the higher weight, the slower */
    double col1[3];  /* Red, green, blue from 0.0 to 1.0 */
    double col2[3];  /* dto. */

    col_struct() : weight(0), col1(), col2() {}
    col_struct(const json::value jv);
    json::value toJSON() const;
};


class RealPalette : public JSONSerializable {
public:
    RealPalette() { reset(); }
    RealPalette(const RealPalette& r);

    double computeWeightSum();

    void pixelValue(
        int x1, int x2,
        int rmax, int gmax, int bmax,
        unsigned char* line,
        float* CBuf, 
        float* BBuf);

    void fromArray(int nColors, const double (*colors)[3], bool wrap = false);
    void reset();
    void print();

    RealPalette(const json::value& jv);
    virtual json::value toJSON() const;

    static constexpr size_t maxColors = 100;

    struct col_struct _cols[maxColors];
    size_t _nColors;

private:

    double _weightSum;

    void getTrueColor(double color, double* r, double* g, double* b);
};

RealPalette tag_invoke(const json::value_to_tag< RealPalette >&, json::value const& jv);


constexpr double GAMMA = 1.0 / 2.2;
