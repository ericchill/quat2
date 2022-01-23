#include "parameters.h"
#include <iostream>
#include "quatfiles.h"

void FractalSpec::reset() {
    _c = 0;
    _bailout = 16;
    _maxiter = 30;
    _maxOrbit = 100;
    _lTerm = 0;
    _formula = 0;
    for (int i = 0; i < numPowers; i++) {
        _p[i] = 0;
    }
}

void FractalSpec::print() const {
    std::cout << "(" << _c[0] << "," << _c[1] << "," << _c[2] << "," << _c[3] << "), "
        << _bailout << ", " << _maxiter << ", " << _lTerm << ", " << _formula << std::endl;
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
            {"maxorbit", _maxOrbit },
            {"lastiter", _lastiter},
            {"lvalue", _lTerm},
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




FractalPreferences::FractalPreferences(const json::value& jv) {
    try {
        const json::object& obj = jv.as_object();
        if (obj.contains("spec")) {
            _spec = json::value_to<FractalSpec>(jv.at("spec"));
        }
        if (obj.contains("view")) {
            _view = json::value_to<FractalView>(jv.at("view"));
        }
        if (obj.contains("palette")) {
            _realpal = json::value_to<RealPalette>(jv.at("palette"));
        }
        if (obj.contains("colorscheme")) {
            _colorScheme = json::value_to<ColorScheme>(jv.at("colorscheme"));
        }
        if (obj.contains("cuts")) {
            _cuts = json::value_to<CutSpec>(jv.at("cuts"));
        }
    }
    catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << jv << std::endl;
        std::rethrow_exception(std::current_exception());
    }
}

json::value FractalPreferences::toJSON(int saveChoices) const {
    json::object result;
    if (saveChoices & PS_OBJ) {
        result.emplace("spec", _spec.toJSON());
    }
    if (saveChoices & PS_VIEW) {
        result.emplace("view", _view.toJSON());
    }
    if (saveChoices & PS_COL) {
        result.emplace("palette", _realpal.toJSON());
        result.emplace("colorscheme", _colorScheme.toJSON());
    }
    if (saveChoices & PS_OTHER) {
        result.emplace("cuts", _cuts.toJSON());
    }
    return result;
}

FractalPreferences tag_invoke(const json::value_to_tag< FractalPreferences >&, json::value const& jv) {
    return FractalPreferences(jv);
}
