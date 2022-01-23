#pragma once

#include "common.h"
#include "qmath.h"
#include "json.h"

#include "colors.h"
#include "FractalView.h"
#include "CutSpec.h"


#include <cstring>
#include <tuple>

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

inline CUDA_CALLABLE bool shouldCalculateDepths(ZFlag zflag) {
    return zflag != ZFlag::ImageFromZBuffer;
}
inline CUDA_CALLABLE bool shouldCalculateImage(ZFlag zflag) {
    return zflag == ZFlag::NewImage || zflag == ZFlag::ImageFromZBuffer;
}


class FractalSpec : public JSONSerializable {
public:
    FractalSpec() {
        reset();
    }
    FractalSpec(const FractalSpec& f) {
        _c = f._c;
        _bailout = f._bailout;
        _maxiter = f._maxiter;
        _maxOrbit = f._maxOrbit;
        _lastiter = f._lastiter;
        _lTerm = f._lTerm;
        _formula = f._formula;
        for (size_t i = 0; i < numPowers; i++) {
            _p[i] = f._p[i];
        }
    }
    void reset();
    void print() const;

    FractalSpec(const json::value& jv) {
        _maxOrbit = 100;
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
    int _maxOrbit;
};

FractalSpec tag_invoke(const json::value_to_tag< FractalSpec >&, json::value const& jv);


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


class FractalPreferences {
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
    json::value toJSON(int saveChoices) const;
};

FractalPreferences tag_invoke(const json::value_to_tag< FractalPreferences >&, json::value const& jv);

