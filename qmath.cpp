#include "qmath.h"

void tag_invoke(const json::value_from_tag&, json::value& jv, Vec3 const& t) {
    jv = { t[0], t[1], t[2] };
}

Vec3 tag_invoke(const json::value_to_tag< Vec3 >&, json::value const& jv) {
    return Vec3(
        jv.as_array().at(0).to_number<double>(),
        jv.as_array().at(1).to_number<double>(),
        jv.as_array().at(2).to_number<double>());
}

std::ostream& operator<<(std::ostream& oo, const Vec3& v) {
    oo << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return oo;
}



template<>
void tag_invoke(const json::value_from_tag&, json::value& jv, Quaternion<double> const& t) {
    jv = { t[0], t[1], t[2], t[3] };
}

template<>
Quaternion<double> tag_invoke(const json::value_to_tag< Quaternion<double> >&, json::value const& jv) {
    const json::array& arr = jv.as_array();
    return Quaternion<double>(
        arr.at(0).to_number<double>(),
        arr.at(1).to_number<double>(),
        arr.at(2).to_number<double>(),
        arr.at(3).to_number<double>());
}
template<>
std::ostream& operator<< (std::ostream& oo, const Quaternion<double>& q) {
    oo << "(" << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << ")";
    return oo;
}

