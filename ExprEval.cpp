#include "ExprEval.h"
#include <stdexcept>

#define _USE_MATH_DEFINES // To get M_PI &c.
#include <math.h>


std::unordered_map<std::string, int> EvalSymbol::_symbolMap;
std::mutex EvalSymbol::_mapMutex;

double signum(double x) {
    if (x < 0) {
        return -1;
    } else if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

void EvalContext::setupBasicVariables(EvalContext& ctx) {
    ctx.setVariable("pi", M_PI);
    ctx.setVariable("phi", (1 + sqrt(5)) / 2);
    ctx.setVariable("sqrt2", M_SQRT2);
    ctx.setVariable("sqrt3", sqrt(3));

    ctx.setBinaryFunc("min", fmin);
    ctx.setBinaryFunc("max", fmax);
    ctx.setUnaryFunc("abs", fabs);
    ctx.setUnaryFunc("floor", floor);
    ctx.setUnaryFunc("ceil", ceil);
    ctx.setUnaryFunc("trunc", trunc);
    ctx.setUnaryFunc("round", round);

    ctx.setUnaryFunc("exp", exp);
    ctx.setUnaryFunc("log", log);
    ctx.setUnaryFunc("log1p", log1p);
    ctx.setUnaryFunc("log2", log2);
    ctx.setUnaryFunc("log10", log10);

    ctx.setUnaryFunc("sqrt", sqrt);
    ctx.setUnaryFunc("cbrt", cbrt);
    ctx.setBinaryFunc("hypot", hypot);

    ctx.setUnaryFunc("sin", sin);
    ctx.setUnaryFunc("cos", cos);
    ctx.setUnaryFunc("tan", tan);
    ctx.setUnaryFunc("asin", asin);
    ctx.setUnaryFunc("acos", acos);
    ctx.setUnaryFunc("atan", atan);
    ctx.setBinaryFunc("atan2", atan2);

    ctx.setUnaryFunc("sinh", sinh);
    ctx.setUnaryFunc("cosh", cosh);
    ctx.setUnaryFunc("tanh", tanh);
    ctx.setUnaryFunc("asinh", asinh);
    ctx.setUnaryFunc("acosh", acosh);
    ctx.setUnaryFunc("atanh", atanh);

    ctx.setUnaryFunc("gamma", tgamma);
    ctx.setUnaryFunc("lgamma", lgamma);

    ctx.setUnaryFunc("sgn", signum);
}

double VariableExpr::eval(EvalContext& ctx) {
    try {
        return ctx.getVariable(_symbol);
    } catch (std::out_of_range&) {
        notFound();
        return 0;
    }
}


double NilaryFunction::eval(EvalContext& ctx) {
    try {
        return ctx.getNilaryFunc(_symbol)();
    } catch (std::out_of_range&) {
        notFound();
        return 0;
    }
}


double UnaryFunction::eval(EvalContext& ctx) {
    try {
        double arg = _arg->eval(ctx);
        return ctx.getUnaryFunc(_symbol)(arg);
    } catch (std::out_of_range&) {
        notFound();
        return 0;
    }
}


double BinaryFunction::eval(EvalContext& ctx) {
    try {
        double arg1 = _arg1->eval(ctx);
        double arg2 = _arg2->eval(ctx);
        return ctx.getBinaryFunc(_symbol)(arg1, arg2);
    } catch (std::out_of_range&) {
        notFound();
        return 0;
    }
}
