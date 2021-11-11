/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "common.h"
#include "qmath.h"
#include <math.h>

void tag_invoke(const json::value_from_tag&, json::value& jv, vec3 const& t) {
    jv = { t[0], t[1], t[2] };
}

vec3 tag_invoke(const json::value_to_tag< vec3 >&, json::value const& jv) {
    return vec3(
        jv.as_array().at(0).to_number<double>(),
        jv.as_array().at(1).to_number<double>(),
        jv.as_array().at(2).to_number<double>());
}

std::ostream& operator<<(std::ostream& oo, const vec3& v) {
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

