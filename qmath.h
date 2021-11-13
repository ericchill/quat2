#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

#include "json.h"

#ifdef __NVCC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#include <intrin.h>
#endif

#define TOO_SMALL(x) (fabs(x) < 1e-100)
#define TOO_BIG(x)   (fabs(x) > 1e+100)


class alignas(16) vec3 {
    double _v[3];
    double _pad;
public:
    vec3() {
        memset(_v, 0, sizeof(_v));
    }
    vec3(const vec3& v) {
        memcpy(_v, v._v, sizeof(_v));
    }
    vec3(double x, double y, double z) {
        _v[0] = x;
        _v[1] = y;
        _v[2] = z;
    }
    vec3(double* p) {
        for (int i = 0; i < 3; i++) {
            _v[i] = p[i];
        }
    }
    double& operator[](size_t i) {
        return _v[i];
    }
    double operator[](size_t i) const {
        return _v[i];
    }
    vec3 operator+(const vec3& o) const {
        vec3 res = *this;
        for (int i = 0; i < 3; i++) {
            res[i] += o[i];
        }
        return res;
    }
    vec3 operator-() const {
        vec3 res = *this;
        for (int i = 0; i < 3; i++) {
            res[i] *= -1;
        }
        return res;
    }
    vec3 operator-(const vec3& o) const {
        vec3 res = *this;
        for (int i = 0; i < 3; i++) {
            res[i] -= o[i];
        }
        return res;
    }
    vec3 operator*(double s) const {
        vec3 res = *this;
        for (int i = 0; i < 3; i++) {
            res[i] *= s;
        }
        return res;
    }
    vec3 operator/(double s) const {
        vec3 res = *this;
        for (int i = 0; i < 3; i++) {
            res[i] /= s;
        }
        return res;
    }
    vec3 operator+=(vec3& o) {
        *this = *this + o;
        return *this;
    }
    vec3 operator/=(double s) {
        *this = *this / s;
        return *this;
    }
    double dot(const vec3& o) const {
        return _v[0] * o[0] + _v[1] * o[1] + _v[2] * o[2];
    }
    double magnitude() const {
        return sqrt(dot(*this));
    }
    vec3 normalize() const {
        double m = magnitude();
        if (m != 0) {
            return *this / m;
        } else {
            return *this;
        }
    }
    vec3 cross(const vec3& o) const {
        return vec3(
            _v[1] * o[2] - _v[2] * o[1],
            _v[2] * o[0] - _v[0] * o[2],
            _v[0] * o[1] - _v[1] * o[0]);
    }
};

inline vec3 operator*(double s, const vec3& v) {
    return v * s;
}

std::ostream& operator<<(std::ostream& oo, const vec3& v);

void tag_invoke(const json::value_from_tag&, json::value& jv, vec3 const& t);

vec3 tag_invoke(const json::value_to_tag< vec3 >&, json::value const& jv);



template<typename T>
class alignas(16) Quaternion {
    T _q[4];
public:
    CUDA_CALLABLE Quaternion() {
        memset(_q, 0, sizeof(_q));
    }
    CUDA_CALLABLE Quaternion(const Quaternion<T>& q) {
        memcpy(_q, q._q, sizeof(_q));
    }
    CUDA_CALLABLE Quaternion(T x) {
        memset(_q, 0, sizeof(_q));
        _q[0] = x;
    }
    CUDA_CALLABLE Quaternion(T a, T b, T c, T d) {
        _q[0] = a;
        _q[1] = b;
        _q[2] = c;
        _q[3] = d;
    }
    CUDA_CALLABLE Quaternion(const vec3& v) {
        for (int i = 0; i < 3; i++) {
            _q[i] = v[i];
        }
        _q[3] = 0;
    }

    CUDA_CALLABLE T& operator[](size_t i) {
        return _q[i];
    }

    CUDA_CALLABLE T operator[](size_t i) const {
        return _q[i];
    }

    CUDA_CALLABLE T real() const {
        return _q[0];
    }

    CUDA_CALLABLE operator T() const {
        return real();
    }
    
    CUDA_CALLABLE Quaternion<T> imag() const {
        Quaternion<T> res = *this;
        res[0] = 0;
        return res;
    }

    CUDA_CALLABLE Quaternion<T> conjugate() const {
        return real() - imag();
    }

    CUDA_CALLABLE T magnitudeSquared() const {
        T sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += _q[i] * _q[i];
        }
        return sum;
    }

    CUDA_CALLABLE T magnitude() const {
        return sqrt(magnitudeSquared());
    }

    CUDA_CALLABLE Quaternion<T> operator+(const Quaternion<T>& b) const {
        Quaternion<T> res = *this;
        for (int i = 0; i < 4; i++) {
            res[i] += b[i];
        }
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator+(T x) const {
        Quaternion<T> res = *this;
        res[0] += x;
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator-() const {
        Quaternion<T> res = *this;
        for (int i = 0; i < 4; i++) {
            res *= -1;
        }
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator-(const Quaternion<T>& b) const {
        Quaternion<T> res = *this;
        for (int i = 0; i < 4; i++) {
            res[i] -= b[i];
        }
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator*(T s) const {
        Quaternion<T> res = *this;
        for (int i = 0; i < 4; i++) {
            res[i] *= s;
        }
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator*(const Quaternion<T>& b) const {
        return Quaternion<T>(
            _q[0] * b[0] - _q[1] * b[1] - _q[2] * b[2] - _q[3] * b[3],
            _q[0] * b[1] + _q[1] * b[0] + _q[2] * b[3] - _q[3] * b[2],
            _q[0] * b[2] - _q[1] * b[3] + _q[2] * b[0] + _q[3] * b[1],
            _q[0] * b[3] + _q[1] * b[2] - _q[2] * b[1] + _q[3] * b[0]);
    }

    CUDA_CALLABLE Quaternion<T> operator/(T s) const {
        return *this * (1 / s);
    }

    CUDA_CALLABLE Quaternion<T> reciprocal() const {
        return conjugate() / magnitudeSquared();
    }

    CUDA_CALLABLE Quaternion<T> operator/(const Quaternion<T>& b) const {
        return *this * b.reciprocal();
    }

    CUDA_CALLABLE Quaternion<T> squared() const {
        return Quaternion<T>(
            _q[0] * _q[0] - _q[1] * _q[1] - _q[2] * _q[2] - _q[3] * _q[3],
            2 * _q[0] * _q[1],
            2 * _q[0] * _q[2],
            2 * _q[0] * _q[3]);
    }
    CUDA_CALLABLE Quaternion<T> componentMul(const Quaternion<T>& b) const {
        Quaternion<T> res = *this;
        for (int i = 0; i < 4; i++) {
            res[i] *= b[i];
        }
        return res;
    }

    CUDA_CALLABLE T sumComponents() const {
        T res = 0;
        for (int i = 0; i < 4; i++) {
            res += _q[i];
        }
        return res;
    }

    CUDA_CALLABLE Quaternion<T> operator+=(const Quaternion<T> o) {
        *this = *this + o;
        return *this;
    }

    CUDA_CALLABLE operator vec3() const {
        return vec3(_q[0], _q[1], _q[2]);
    }
};

template< typename T >
void tag_invoke(const json::value_from_tag&, json::value& jv, Quaternion<T> const& t);

template< typename T >
Quaternion<T> tag_invoke(const json::value_to_tag< Quaternion<T> >&, json::value const& jv);

template<class T>
std::ostream& operator<<(std::ostream& oo, const Quaternion<T>& q);

template<typename T>
CUDA_CALLABLE Quaternion<T> operator+(T x, const Quaternion<T>& q) {
    return q + x;
}

template<typename T>
CUDA_CALLABLE Quaternion<T> operator*(T s, const Quaternion<T>& q) {
    return q * s;
}

template<typename T>
CUDA_CALLABLE Quaternion<T> operator/(T s, const Quaternion<T>& q) {
    return q.reciprocal() * s;
}

template<typename T>
CUDA_CALLABLE T exp(const Quaternion<T>& x) {
    T ex = exp(x[0]);
    T n = x.imag().magnitude();
    T f = ex * sin(n);
    if (n != 0.0) {
        f /= n;
    }
    return Quaternion<T>(
        ex * cos(n),
        f * x[1],
        f * x[2],
        f * x[3]);
}

template<typename T>
CUDA_CALLABLE Quaternion<T> log(const Quaternion<T>& x) {
    T n = x.imag().magnitude();
    if (0 == n) {
        return Quaternion<T>(
            0.5 * log(x[0] * x[0]),
            atan2(0, x[0]),
            0, 0);
    } else {
        double f = atan2(n, x[0]) / n;
        return Quaternion<T>(
            0.5 * log(x[0] * x[0] + n * n),
            f * x[1],
            f * x[2],
            f * x[3]);
    }
}

template<typename T>
CUDA_CALLABLE Quaternion<T> pow(const Quaternion<T>& x, const Quaternion<T>& y) {
    T an = x.magnitudeSquared();
    Quaternion<T> yImag = y.imag();
    T bnp = yImag.magnitudeSquared();
    if (TOO_SMALL(an) && (y[0] > 0 || bnp != 0)) {
        return Quaternion<T>(0);
    }
    return exp(y * log(x));
}


typedef Quaternion<double> Quat;

template<>
std::ostream& operator<<(std::ostream& oo, const Quaternion<double>& q);


#ifndef __NVCC__
inline void intrinsicSquared(double* in, double* out) {
    alignas(32) double real2[4];
    register __m256d q = _mm256_load_pd(in);
    register __m256d re2 = _mm256_mul_pd(q, q);
    _mm256_store_pd(real2, re2);
    __m128d q0scalar = _mm_load_pd(in);
    __m256d q0 = _mm256_broadcastsd_pd(q0scalar);
    __m256d allByQ0 = _mm256_mul_pd(q0, q);
    allByQ0 = _mm256_add_pd(allByQ0, allByQ0);
    _mm256_store_pd(out, allByQ0);
    out[0] = real2[0] - real2[1] - real2[2] - real2[3];
}
#endif

template<>
Quaternion<double> Quaternion<double>::squared() const {
#ifndef __NVCC__
    double result[4];
    intrinsicSquared((double*)_q, result);
    return Quaternion(result);
#else
    return Quaternion<double>(
        _q[0] * _q[0] - _q[1] * _q[1] - _q[2] * _q[2] - _q[3] * _q[3],
        2 * _q[0] * _q[1],
        2 * _q[0] * _q[2],
        2 * _q[0] * _q[3]);
#endif
}