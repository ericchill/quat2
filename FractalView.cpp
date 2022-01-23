#include "FractalView.h"
#include "parameters.h"


ViewBasis::ViewBasis(json::value const& jv) {
    _O = json::value_to<Vec3>(jv.at("O"));
    _x = json::value_to<Vec3>(jv.at("x"));
    _y = json::value_to<Vec3>(jv.at("y"));
    _z = json::value_to<Vec3>(jv.at("z"));
}

json::value ViewBasis::toJSON() const {
    return {
            { "O", _O },
            { "x", _x },
            { "y", _y },
            { "z", _z }
    };
}

ViewBasis tag_invoke(const json::value_to_tag< ViewBasis >&, json::value const& jv) {
    return ViewBasis(jv);
}


FractalView::FractalView(const FractalView& f) {
    _s = f._s;
    _up = f._up;
    _light = f._light;
    _Mov[0] = f._Mov[0];
    _Mov[1] = f._Mov[1];
    _LXR = f._LXR;
    _xres = f._xres;
    _yres = f._yres;
    _zres = f._zres;
    _interocular = f._interocular;
    _phongmax = f._phongmax;
    _phongsharp = f._phongsharp;
    _ambient = f._ambient;
    _antialiasing = f._antialiasing;
}

void FractalView::reset() {
    _s = Vec3(0, 0, -2);
    _up = Vec3(0, -1, 0);
    _light = Vec3(-2, -5, -5);
    _Mov[0] = 0;
    _Mov[1] = 0;
    _LXR = 2.8;
    _xres = 640;
    _yres = 480;
    _zres = 480;
    _interocular = 0;
    _phongmax = 0.6f;
    _phongsharp = 30;
    _ambient = 0.04f;
    _antialiasing = 1;
}

void FractalView::print() const {
    std::cout << _s << ", " << _up << ", " << _light
        << "(" << _Mov[0] << "," << _Mov[1] << "), "
        << _LXR << ", "
        << "(" << _xres << "," << _yres << "," << _zres << "), "
        << _interocular << ", " << _phongmax << ", " << _phongsharp << ", "
        << _ambient << ", " << _antialiasing << std::endl;
}

FractalView::FractalView(const json::value& jv) {
    const json::object& obj = jv.as_object();
    _s = json::value_to<Vec3>(jv.at("s"));
    _s = json::value_to<Vec3>(obj.at("s"));
    _up = json::value_to<Vec3>(obj.at("up"));
    _light = json::value_to<Vec3>(obj.at("light"));
    _Mov[0] = obj.at("mov0").to_number<double>();
    _Mov[1] = obj.at("mov1").to_number<double>();
    _LXR = obj.at("LXR").to_number<double>();
    _xres = obj.at("xres").to_number<int>();
    _yres = obj.at("yres").to_number<int>();
    _zres = obj.at("zres").to_number<int>();
    _interocular = obj.at("interocular").to_number<float>();
    _phongmax = obj.at("phongmax").to_number<float>();
    _phongsharp = obj.at("phongsharp").to_number<float>();
    _ambient = obj.at("ambient").to_number<float>();
    _antialiasing = obj.at("antialiasing").to_number<int>();
}

json::value FractalView::toJSON() const {
    return {
        { "s", _s },
        { "up", _up },
        { "light", _light },
        { "mov0", _Mov[0] },
        { "mov1", _Mov[1] },
        { "LXR", _LXR },
        { "xres", _xres },
        { "yres", _yres },
        { "zres", _zres },
        { "interocular", _interocular },
        { "phongmax", _phongmax },
        { "phongsharp", _phongsharp },
        { "ambient", _ambient },
        { "antialiasing", _antialiasing }
    };
}


FractalView tag_invoke(const json::value_to_tag< FractalView >&, json::value const& jv) {
    return FractalView(jv);
}


int FractalView::DoCalcbase(
    ViewBasis* base,
    ViewBasis* sbase,
    bool use_proj_up,
    Vec3 proj_up) {

    if (0 == _xres || 0 == _yres || 0 == _zres) {
        return -1;
    }
    double leny = _LXR * _yres / _xres;

    base->_z = -_s.normalized();

    _up = _up.normalized();

    /* check whether up is linearly dependent on z */
    /* cross product != 0 */
    if (_up.cross(base->_z).magnitudeSquared() == 0) {
        return -1;
    }

    if (use_proj_up) {
        base->_y = proj_up;
    } else {
        /* Check whether up orthogonal to _z */
        if (base->_z.dot(_up) == 0.0) {
            /* Yes -> norm(up) = -y-Vec3 */
            base->_y = -_up;
        } else {
            /* calculate projection of up onto pi */
            double lambda = base->_z.dot(_s - _up);
            Vec3 ss = lambda * base->_z + _up;

            base->_y = (_s - ss).normalized();
        }
    }

    /* x orthogonal to y and z */
    base->_x = base->_y.cross(base->_z);

    /* calculate origin */
    leny /= 2;
    _LXR /= 2;
    base->_O = _s - leny * base->_y - _LXR * base->_x;

    /* ready with base, now: calculate a specially pseudo-normed base */
    /* where the length of the base vectors represent 1 pixel */

    if (nullptr == sbase) {
        return 0;
    }
    _LXR *= 2;
    leny *= 2;
    sbase->_O = base->_O;
    sbase->_x = _LXR * base->_x / _xres;
    sbase->_y = leny * base->_y / _yres;

    /* how deep into scene */
    sbase->_z = 2 * fabs(base->_z.dot(_s)) * base->_z / _zres;

    /* Only thing that has to be done: shift the plane */
    return 0;
}

double whichEyeSign(WhichEye eye) {
    switch (eye) {
    default:  // VS2019 compiler can't tell the rest of the cases are exhaustive.
    case WhichEye::Monocular:
        return 0;
    case WhichEye::Right:
        return 1;
    case WhichEye::Left:
        return -1;
    }
}

int FractalView::calcbase(ViewBasis* base, ViewBasis* sbase, WhichEye viewType) {
    ViewBasis commonbase;
    Vec3 y;

    int e = DoCalcbase(base, sbase, false, y);
    if (e != 0) {
        return e;
    }
    commonbase = *base;
    if (viewType != WhichEye::Monocular) {
        FractalView v2 = *this;
        v2._s += whichEyeSign(viewType) * _interocular / 2 * base->_x;
        y = base->_y;
        e = v2.DoCalcbase(base, sbase, true, y);
        if (e != 0) {
            return e;
        }
    }

    /* Shift plane in direction of common plane */
    base->_O += _Mov[0] * commonbase._x + _Mov[1] * commonbase._y;
    if (sbase != nullptr) {
        sbase->_O = base->_O;
    }
    return 0;
}




double FractalView::brightness(const Vec3& p, const Vec3& n, const Vec3& z) const {
    /* values >1 are for phong */
    Vec3 l, r;
    double absolute, result, a, b, c;

    /* vector point -> light source */
    l = _light - p;
    absolute = l.magnitudef();
    if (absolute == 0) {
        return 255.0;   /* light source and point are the same! */
    }
    l /= absolute;

    /* Lambert */
    a = n.dotf(l);

    /* Distance */
    b = 20 / absolute;
    if (b > 1) {
        b = 1;
    }

    /* Phong */
    if (a > 0) {
        r = 2 * a * n - l;
        /* r...reflected ray */
        c = r.dot(z);
        if (c < 0) {
            c = _phongmax * exp(_phongsharp * log(-c));
        } else {
            c = 0;
        }
    } else {
        c = 0;
    }

    if (a < 0) {
        result = _ambient;
    } else {
        result = a * b + _ambient;
        result = fmin(result, 1.0); /* Lambert and ambient together can't get bigger than 1 */
        result += c;                   /* additional phong to get white */
        result = fmax(result, 0);
    }
    return result;
}




int formatExternToIntern(FractalSpec& spec, FractalView& view) {
    /* This is to change the input format to internal format */
    /* Input: light relative to viewers position (for convenience of user) */
    /* Internal: light absolute in space (for convenience of programmer) */
    ViewBasis base;

    if (view.calcbase(&base, nullptr, WhichEye::Monocular) != 0) {
        return -1;
    }
    Vec3 ltmp = view._light[0] * base._x + view._light[1] * base._y + view._light[2] * base._z;
    view._light = view._s + view._Mov[0] * base._x + view._Mov[1] * base._y;
    /* Internal: square of bailout value (saves some time) */
    spec._bailout *= spec._bailout;
    return 0;
}

int formatInternToExtern(FractalSpec& frac, FractalView& view) {
    /* Reverse process of "formatExternToIntern". see above */
    ViewBasis base;

    if (view.calcbase(&base, nullptr, WhichEye::Monocular) != 0) {
        return -1;
    }
    frac._bailout = sqrt(frac._bailout);
    Vec3 ltmp = view._light - view._s;
    view._light = Vec3(
        ltmp.dot(base._x) - view._Mov[0],
        ltmp.dot(base._y) - view._Mov[1],
        ltmp.dot(base._z));
    return 0;
}
