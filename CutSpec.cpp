#include "CutSpec.h"

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

bool CutSpec::getPlane(size_t i, Vec3& normal, Vec3& point) const {
    if (i < _count) {
        normal = _normal[i];
        point = _point[i];
        return true;
    }
    return false;
}

bool CutSpec::setPlane(size_t i, const Vec3& normal, const Vec3& point) {
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

bool CutSpec::addPlane(const Vec3& normal, const Vec3& point) {
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
        if (1 == _count) {
            _normal[0] = 0.0;
            _point[0] = 0.0;
        } else {
            for (size_t j = i; j < _count - 1; j++) {
                _normal[j] = _normal[j + 1];
                _point[j] = _point[j + 1];
            }
        }
        _count--;
        return true;
    }
    return false;
}

bool CutSpec::cutaway(const Vec3& x) const {
    for (unsigned i = 0; i < _count; i++) {
        Vec3 y = x - _point[i];
        if (_normal[i].dot(y) > 0) {
            return true;
        }
    }
    return false;
}

bool CutSpec::cutnorm(const Quat& x1, const Quat& x2, Quat& nq) const {
    Vec3 n;
    Vec3 vx1(x1);
    Vec3 vx2(x2);

    for (size_t i = 0; i < _count; i++) {
        Vec3 y1 = vx1 - _point[i];
        Vec3 y2 = vx2 - _point[i];
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
        _normal[i] = json::value_to<Vec3>(cut.at("normal"));
        _point[i] = json::value_to<Vec3>(cut.at("point"));
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