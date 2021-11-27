#pragma once

#include <cstring>
#include "common.h"
#include "qmath.h"
#include "json.h"

#include "colors.h"


/*
0..image from scratch, size view.xres* view.yres
1..ZBuffer from scratch, size view.xres* view.yres* AA ^ 2
2..image from ZBuffer, size img : xres * yres, buffer* AA ^ 2
for every case take into account that images can be stereo!
*/
enum class ZFlag {
    NewImage,
    NewZBuffer,
    ImageFromZBuffer
};

inline bool shouldCalculateDepths(ZFlag zflag) {
    return zflag != ZFlag::ImageFromZBuffer;
}
inline bool shouldCalculateImage(ZFlag zflag) {
    return zflag == ZFlag::NewImage || zflag == ZFlag::ImageFromZBuffer;
}


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
    ViewBasis(json::value const& jv);
    virtual json::value toJSON() const;
};

ViewBasis tag_invoke(const json::value_to_tag< ViewBasis >&, json::value const& jv);


class FractalSpec : public JSONSerializable {
public:
    FractalSpec() {
        reset();
    }
    FractalSpec(const FractalSpec& f) {
        _c = f._c;
        _bailout = f._bailout;
        _maxiter = f._maxiter;
        _lastiter = f._lastiter;
        _lTerm = f._lTerm;
        _formula = f._formula;
        for (size_t i = 0; i < numPowers; i++) {
            _p[i] = f._p[i];
        }
    }
    FractalSpec(
        const Quat& c,
        double bailout,
        int maxiter,
        double lvalue,
        int formula,
        const Quat* const p) {
        _c = c;
        _bailout = bailout;
        _maxiter = maxiter;
        _lTerm = lvalue;
        _formula = formula;
        for (size_t i = 0; i < numPowers; i++) {
            _p[i] = p[i];
        }
        _lastiter = 0;
    }
    void reset();
    void print() const;

    FractalSpec(const json::value& jv) {
        const json::object& obj = jv.as_object();
        _c = json::value_to<Quaternion<double> >(obj.at("c"));
        _bailout = obj.at("bailout").to_number<double>();
        _maxiter = obj.at("maxiter").to_number<int>();
        _lTerm = obj.at("lvalue").to_number<double>();
        _formula = obj.at("formula").to_number<int>();
        _p[0] = json::value_to<Quat>(obj.at("p1"));
        _p[1] = json::value_to<Quat>(obj.at("p2"));
        _p[2] = json::value_to<Quat>(obj.at("p3"));
        _p[3] = json::value_to<Quat>(obj.at("p4"));
        _lastiter = 0;
    }
    virtual json::value toJSON() const;

    Quat _c;
    double _bailout;
    int _maxiter;
    double _lastiter;
    double _lTerm;
    int _formula;
    static constexpr size_t numPowers = 4;
    Quat _p[numPowers];
};

FractalSpec tag_invoke(const json::value_to_tag< FractalSpec >&, json::value const& jv);


enum class WhichEye {
    Monocular,
    Right,
    Left
};
double whichEyeToSign(WhichEye eye);


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



class ColorScheme : public JSONSerializable {
    char _data[256];
public:
    ColorScheme() {
        reset();
    }
    ColorScheme(const char v[]) {
        set(v);
    }
    void reset();
    void set(const char* v) {
        strcpy_s(_data, sizeof(_data), v);
    }
    size_t size() const {
        return sizeof(_data);
    }
    const char* const get() const {
        return _data;
    }

    ColorScheme(const json::value& jv);

    virtual json::value toJSON() const;
};

ColorScheme tag_invoke(const json::value_to_tag< ColorScheme >&, json::value const& jv);


class CutSpec : public JSONSerializable {
public:
    CutSpec() {
        reset();
    }
    void reset();
    void print();
    size_t count() const;
    bool getPlane(size_t i, Vec3& normal, Vec3& point) const;
    bool setPlane(size_t i, const Vec3& normal, const Vec3& point);
    bool addPlane(const Vec3& normal, const Vec3& point);
    bool deletePlane(size_t i);

    bool cutaway(const Vec3 x) const;
    bool cutnorm(const Quat& x1, const Quat& x2, Quat& nq) const;

    CutSpec(const json::value& jv);
    virtual json::value toJSON() const;

    static constexpr size_t maxCuts = 6;
    static constexpr size_t cutBufSize = 140;
    /* intersection objects definitions (only planes for now) */
    /* every structure in this section must have */
    /* _a char "cut_type" as first element */

private:
    size_t _count;
    Vec3 _normal[maxCuts];
    Vec3 _point[maxCuts];
};

CutSpec tag_invoke(const json::value_to_tag< CutSpec >&, json::value const& jv);


class FractalPreferences : public JSONSerializable {
    FractalSpec _spec;
    FractalView _view;
    RealPalette _realpal;
    ColorScheme _colorScheme;
    CutSpec _cuts;
public:
    FractalPreferences() {
        reset();
    }

    FractalPreferences(const FractalPreferences& o) {
        _spec = o._spec;
        _view = o._view;
        _realpal = o._realpal;
        _colorScheme = o._colorScheme;
        _cuts = o._cuts;
    }

    void reset() {
        _spec.reset();
        _view.reset();
        _realpal.reset();
        _colorScheme.reset();
        _cuts.reset();
    }

    FractalSpec& fractal() {
        return _spec;
    }
    FractalView& view() {
        return _view;
    }
    RealPalette& realPalette() {
        return _realpal;
    }
    ColorScheme& colorScheme() {
        return _colorScheme;
    }
    CutSpec& cuts() {
        return _cuts;
    }

    const FractalSpec& fractal() const {
        return _spec;
    }
    const FractalView& view() const {
        return _view;
    }
    const RealPalette& realPalette() const {
        return _realpal;
    }
    const ColorScheme& colorScheme() const {
        return _colorScheme;
    }
    const CutSpec& cuts() const {
        return _cuts;
    }

    FractalPreferences(const json::value& jv);
    virtual json::value toJSON() const;
};

FractalPreferences tag_invoke(const json::value_to_tag< FractalPreferences >&, json::value const& jv);


struct iter_struct;

struct calc_struct {
    calc_struct(
        const FractalSpec& fractal,
        const FractalView& view,
        const CutSpec& _cuts,
        ViewBasis& base,
        ViewBasis& sbase,
        ZFlag zflag);
    ~calc_struct();
    Quat _xp, _xs, _xq, _xcalc;
    FractalSpec _f;
    FractalView _v;
    ViewBasis _sbase, _base, _aabase;
    double _absx, _absy, _absz;
    CutSpec _cuts;
    int (*_iterate_no_orbit) (iter_struct*);
    int (*_iterate) (iter_struct*);
    int (*_iternorm) (iter_struct* is, Vec3& norm);

    /* calculates a whole line of depths (precision: 1/20th of the base step in z direction), */
    /* Brightnesses and Colors */
    /* y: the number of the "scan"line to calculate */
    /* x1, x2: from x1 to x2 */
    /* LBuf: buffer for xres doubles. The depth information will be stored here */
    /* BBuf: buffer for xres floats. The brightnesses will be stored here */
    /* CBuf: buffer for xres floats. The colors will be stored here */
    /* fields in calc_struct: */
    /* sbase: the specially normalized _base of the screen's coordinate-system */
    /* f, v: structures with the fractal and view information */
    /* zflag: 0..calc image from scratch;
              1..calc ZBuffer from scratch;
              2..calc image from ZBuffer */
              /* if zflag==2, LBuf must be initialized with the depths from the ZBuffer */
    int calcline2(long x1, long x2, int y,
        double* LBuf, float* BBuf, float* CBuf,
        ZFlag zflag);
    int calcline(long x1, long x2, int y,
        double* LBuf, float* BBuf, float* CBuf,
        ZFlag zflag);
    void obj_distance_search_range(const Vec3& xStart, int& zFrom, int& zTo);
    double obj_distance(int zStart = 0);
    float colorizepoint();
    float brightpoint(long x, int y, double* LBuf);

    size_t _lBufSize;
    int(*_zLimits)[2];
    Quat* _xStarts;
    Quat* _manyOrbits;
    double* _distances;
    double* _lastIters;
};
