#pragma once
#include "common.h"
#include "json.h"
#include "qmath.h"


class ViewBasis : public JSONSerializable {
public:

    Vec3 _O, _x, _y, _z;

    /* origin and 3 base vectors */
    ViewBasis() {
        _O = { 0, 0, 0 };
        _x = { 1, 0, 0 };
        _y = { 0, 1, 0 };
        _z = { 0, 0, 1 };
    }
    ViewBasis(const Vec3& O, const Vec3& x, const Vec3& y, const Vec3& z) {
        _O = O;
        _x = x;
        _y = y;
        _z = z;
    }
    ViewBasis& operator=(const ViewBasis& o) {
        _O = o._O;
        _x = o._x;
        _y = o._y;
        _z = o._z;
        return *this;
    }
    ViewBasis(json::value const& jv);
    virtual json::value toJSON() const;
};

ViewBasis tag_invoke(const json::value_to_tag< ViewBasis >&, json::value const& jv);


enum class WhichEye {
    Monocular,
    Right,
    Left
};
double whichEyeSign(WhichEye eye);


class FractalView : public JSONSerializable {
public:
    FractalView() {
        reset();
    }
    FractalView(const FractalView& f);

    bool isStereo() const {
        return _interocular > 0;
    }
    int renderedXRes() {
        int x = _xres;
        if (isStereo()) {
            x *= 2;
        }
        return x * _antialiasing;
    }
    int renderedYRes() {
        return _yres * _antialiasing;
    }

    /* calculates the _base and specially normalized base according to information in v */
   /* base: will be set by function. Origin and base vectors in 3d "float" space */
   /* sbase: will be set by function. Origin and base vectors specially normalized for 3d "integer" space */
   /*        may be NULL */
   /* v: view information from wich base and sbase will be calculated */
   /* flag:  0 ... common base */
   /*       -1 ... left eye base */
   /*        1 ... right eye base */
   /* returns -1 if view information makes no sense (a division by zero would occur) */
   /* returns 0 if successful */
    int calcbase(
        ViewBasis* base,
        ViewBasis* sbase,
        WhichEye viewType);

    int DoCalcbase(
        ViewBasis* base,
        ViewBasis* sbase,
        bool use_proj_up,
        Vec3 proj_up);

    /* Calculates brightness (diffuse, distance, phong) for point p */
    /* ls: vector origin to light source */
    /* p: vector origin to lightened point */
    /* n: normal vector (length 1 !!!) of object in lightened point */
    /* z: vector point to viewer (length 1 !!!) */
    /* returns brightness from 0.0 to 1.0 / or 255, if ls=p (=error) */
    double brightness(const Vec3& p, const Vec3& n, const Vec3& z) const;

    void reset();
    void print() const;

    FractalView(const json::value& jv);
    virtual json::value toJSON() const;

    Vec3 _s, _up, _light;
    double _Mov[2];
    double _LXR;
    int _xres, _yres, _zres;
    double _interocular;
    double _phongmax, _phongsharp, _ambient;
    int _antialiasing;
};

FractalView tag_invoke(const json::value_to_tag< FractalView >&, json::value const& jv);


class FractalSpec;

int formatExternToIntern(FractalSpec& spec, FractalView& view);

int formatInternToExtern(FractalSpec& frac, FractalView& view);