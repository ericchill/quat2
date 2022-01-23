#pragma once

#include "qmath.h"

struct iter_struct {
	Quat xstart;
	Quat c;
	double bailout;
	int maxiter;
	int maxOrbit;
	Quat p[4];
	Quat *orbit;
};

struct Iter0Op {
	_NODISCARD CUDA_CALLABLE Quat operator()(const Quat& z, const Quat& c, const Quat* /*p*/) const {
		return z.squared() - c;
	}
};

struct Iter1Op {
	_NODISCARD CUDA_CALLABLE Quat operator()(const Quat& z, const Quat& c, const Quat* /*p*/) const {
		return c * z * (1.0 - z);
	}
};

struct Iter2Op {
	_NODISCARD CUDA_CALLABLE Quat operator()(const Quat& z, const Quat& c, const Quat* /*p*/) const {
        return z * log(z) - c;
    }
};

struct Iter3Op {
	_NODISCARD CUDA_CALLABLE Quat operator()(const Quat& z, const Quat& c, const Quat* /*p*/) const {
		return z.squared() * z - c;
	}
};

struct Iter4Op {
	_NODISCARD CUDA_CALLABLE Quat operator()(const Quat& z, const Quat& c, const Quat* p) const {
		return pow(z, p[0]) - c;
	}
};


#ifndef __NVCC__

template<typename Op>
int basic_iterate(iter_struct* is) {
    Op op;
    Quat* orbit = is->orbit;
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;
    Quat z = is->xstart;
    orbit[0] = z;
    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxOrbit-1 && iter < maxiter) {
        z = op(z, is->c, is->p);
        orbit[++iter] = z;
    }
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = op(z, is->c, is->p);
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

template<typename Op>
int basic_iterate_sans_orbit(iter_struct* is) {
    Op op;
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    Quat z = is->xstart;
    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = op(z, is->c, is->p);
        ++iter;
    }
    return iter;
}

#endif // __NVCC__
