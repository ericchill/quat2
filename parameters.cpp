#include "parameters.h"
#include <iostream>


base_struct::base_struct(json::value const& jv) {
    _O = json::value_to<vec3>(jv.at("O"));
    _x = json::value_to<vec3>(jv.at("x"));
    _y = json::value_to<vec3>(jv.at("y"));
    _z = json::value_to<vec3>(jv.at("z"));
}


json::value base_struct::toJSON() const {
    return {
            { "O", _O },
            { "x", _x },
            { "y", _y },
            { "z", _z }
    };
}

base_struct tag_invoke(const json::value_to_tag< base_struct >&, json::value const& jv) {
    return base_struct(jv);
}

void FractalSpec::reset() {
    _c = 0;
    _bailout = 16;
    _maxiter = 12;
    _lvalue = 0;
    _formula = 0;
    for (int i = 0; i < numPowers; i++) {
        _p[i] = 0;
    }
}

void FractalSpec::print() const {
    std::cout << "(" << _c[0] << "," << _c[1] << "," << _c[2] << "," << _c[3] << "), "
        << _bailout << ", " << _maxiter << ", " << _lvalue << ", " << _formula << std::endl;
    for (int j = 0; j < numPowers; ++j) {
        std::cout << "Parameter " << j << ": ";
        for (int i = 0; i < 4; ++i) {
            std::cout << _p[j][i] << "  ";
        }
        std::cout << std::endl;
    }
}

json::value FractalSpec::toJSON() const {
    return {
            {"c", _c },
            {"bailout", _bailout},
            {"maxiter", _maxiter},
            {"lastiter", _lastiter},
            {"lvalue", _lvalue},
            {"formula", _formula},
            {"p1", _p[0]},
            {"p2", _p[1]},
            {"p3", _p[2]},
            {"p4", _p[3]}
    };
}


FractalSpec tag_invoke(const json::value_to_tag< FractalSpec >&, json::value const& jv) {
    return FractalSpec(jv);
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
    _s = vec3(0, 0, -2);
    _up = vec3(0, -1, 0);
    _light = vec3(-2, -5, -5);
    _Mov[0] = 0;
    _Mov[1] = 0;
    _LXR = 2.8;
    _xres = 640;
    _yres = 480;
    _zres = 480;
    _interocular = 0;
    _phongmax = 0.6;
    _phongsharp = 30;
    _ambient = 0.04;
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
    _s = json::value_to<vec3>(jv.at("s"));
    _s = json::value_to<vec3>(obj.at("s"));
    _up = json::value_to<vec3>(obj.at("up"));
    _light = json::value_to<vec3>(obj.at("light"));
    _Mov[0] = obj.at("mov0").to_number<double>();
    _Mov[1] = obj.at("mov1").to_number<double>();
    _LXR = obj.at("LXR").to_number<double>();
    _xres = obj.at("xres").to_number<int>();
    _yres = obj.at("yres").to_number<int>();
    _zres = obj.at("zres").to_number<int>();
    _interocular = obj.at("interocular").to_number<double>();
    _phongmax = obj.at("phongmax").to_number<double>();
    _phongsharp = obj.at("phongsharp").to_number<double>();
    _ambient = obj.at("ambient").to_number<double>();
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
    std::cerr << jv << std::endl;
    json::array colors = jv.as_object().at("colors").as_array();
    for (size_t i = 0; i < colors.size(); i++) {
        _cols[i] = col_struct(colors.at(i));
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


void ColorScheme::reset() {
    strcpy_s(_data, sizeof(_data), "0");
}

ColorScheme::ColorScheme(const json::value& jv) {
    const json::value str = jv.at("formula");
    strcpy_s(_data, sizeof(_data), str.as_string().c_str());
}

json::value ColorScheme::toJSON() const {
    return { { "formula", std::string(_data) } };
}

ColorScheme tag_invoke(const json::value_to_tag< ColorScheme >&, json::value const& jv) {
    return ColorScheme(jv);
}


void CutSpec::reset() {
    _count = 0;
}

void CutSpec::print() {
    for (int i = 0; i < _count; i++) {
        std::cout << "CUT_PLANE (" << _normal[i][0] << "," << _normal[i][1] << "," << _normal[i][2] << "), ("
            << _point[i][0] << "," << _point[i][1] << "," << _point[i][2] << ")";
        std::cout << std::endl;
    }
    if (0 == _count) {
        std::cout << "empty." << std::endl;
    }

}

size_t CutSpec::count() const {
    return _count;
}

bool CutSpec::getPlane(size_t i, vec3& normal, vec3& point) const {
    if (i < _count) {
        normal = _normal[i];
        point = _point[i];
        return true;
    }
    return false;
}

bool CutSpec::setPlane(size_t i, const vec3& normal, const vec3 point) {
    if (i + 1 >= maxCuts) {
        return false;
    }
    _normal[i] = normal;
    _point[i] = point;
    if (i >= _count) {
        _count = i + 1;
    }
    return true;
}

bool CutSpec::addPlane(const vec3& normal, const vec3& point) {
    if (_count < maxCuts) {
        _normal[_count] = normal;
        _point[_count] = point;
        _count++;
        return true;
    }
    return false;
}

bool CutSpec::deletePlane(size_t i) {
    if (i < _count) {
        for (size_t j = i; j < _count - 1; j++) {
            _normal[j] = _normal[j + 1];
            _point[j] = _point[j + 1];
        }
        _count--;
        return true;
    }
    return false;
}

bool CutSpec::cutaway(const vec3 x) const {
    for (unsigned i = 0; i < _count; i++) {
        vec3 y = x - _point[i];
        if (_normal[i].dot(y) > 0) {
            return true;
        }
    }
    return false;
}

bool CutSpec::cutnorm(const Quat& x1, const Quat& x2, Quat& nq) const {
    vec3 n;

    for (size_t i = 0; i < _count; i++) {
        vec3 y1 = vec3(x1) - _point[i];
        Quat y2 = vec3(x2) - _point[i];
        int sign1 = (_normal[i].dot(y1) > 0) ? 1 : -1;
        int sign2 = (_normal[i].dot(y2) > 0) ? 1 : -1;
        if (sign1 != sign2) {
            return true;
        }
    }
    nq = n;
    return false;
}


CutSpec::CutSpec(const json::value& jv) {
    const json::array cuts = jv.as_array();
    for (size_t i = 0; i < cuts.size(); i++) {
        const json::value cut = cuts.at(i);
        _normal[i] = json::value_to<vec3>(cut.at("normal"));
        _point[i] = json::value_to<vec3>(cut.at("normal"));
    }
    _count = cuts.size();
}

json::value CutSpec::toJSON() const {
    json::array cuts;
    for (size_t i = 0; i < _count; i++) {
        cuts.push_back({
            { "normal", _normal[i] },
            { "point", _point[i] } });
    }
    return cuts;

}

CutSpec tag_invoke(const json::value_to_tag< CutSpec >&, json::value const& jv) {
    return CutSpec(jv);
}


FractalPreferences::FractalPreferences(const json::value& jv) {
    std::cerr << jv << std::endl;
    _spec = json::value_to<FractalSpec>(jv.at("spec"));
    _view = json::value_to<FractalView>(jv.at("view"));
    _realpal = json::value_to<RealPalette>(jv.at("palette"));
    _colorScheme = json::value_to<ColorScheme>(jv.at("colorscheme"));
    _cuts = json::value_to<CutSpec>(jv.at("cuts"));

}

json::value FractalPreferences::toJSON() const {
    return {
        { "spec", _spec },
        { "view", _view },
        { "palette", _realpal },
        { "colorscheme", _colorScheme },
        { "cuts", _cuts }
    };
}

FractalPreferences tag_invoke(const json::value_to_tag< FractalPreferences >&, json::value const& jv) {
    return FractalPreferences(jv);
}
