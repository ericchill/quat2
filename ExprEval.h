#pragma once
#include "common.h"
#include <functional>
#include <cmath>
#include <unordered_map>
#include <string>


typedef double (*NilaryFunctionPtr)();
typedef double (*UnaryFunctionPtr)(double);
typedef double (*BinaryFunctionPtr)(double, double);

class EvalException : public QuatException {
public:
    EvalException(const std::string& msg) : QuatException(msg) {}
};


class EvalSymbol {
    int _number;
    std::string _name;

    static std::unordered_map<std::string, int> _symbolMap;
public:
    EvalSymbol(const std::string& name) : _name(name), _number(nameToInt(name)) {}
    int number() { return _number; }
    std::string name() { return _name; }
private:
    int nameToInt(const std::string& name) {
        if (_symbolMap.count(name) == 0) {
            _symbolMap[name] = static_cast<int>(_symbolMap.size());
        }
        return _symbolMap[name];
    }
};


class EvalContext {
    std::unordered_map<int, double> _variables;
    std::unordered_map<int, NilaryFunctionPtr> _nilaryFuncs;
    std::unordered_map<int, UnaryFunctionPtr> _unaryFuncs;
    std::unordered_map<int, BinaryFunctionPtr> _binaryFuncs;
public:

    static void setupBasicVariables(EvalContext& ctx);

    EvalContext() {}
    virtual ~EvalContext() {}

    void setVariable(const std::string& name, double x) {
        _variables[EvalSymbol(name).number()] = x;
    }
    void setVariable(EvalSymbol& symbol, double x) {
        _variables[symbol.number()] = x;
    }
    void setNilaryFunc(const std::string& name, NilaryFunctionPtr f) {
        _nilaryFuncs[EvalSymbol(name).number()] = f;
    }
    void setUnaryFunc(const std::string& name, UnaryFunctionPtr f) {
        _unaryFuncs[EvalSymbol(name).number()] = f;
    }
    void setBinaryFunc(const std::string& name, BinaryFunctionPtr f) {
        _binaryFuncs[EvalSymbol(name).number()] = f;
    }

    double getVariable(EvalSymbol& sym) { return _variables.at(sym.number()); }
    NilaryFunctionPtr getNilaryFunc(EvalSymbol& sym) { return _nilaryFuncs.at(sym.number()); }
    UnaryFunctionPtr getUnaryFunc(EvalSymbol& sym) { return _unaryFuncs.at(sym.number()); }
    BinaryFunctionPtr getBinaryFunc(EvalSymbol& sym) { return _binaryFuncs.at(sym.number()); }
};


class Expression {
public:
    virtual ~Expression() {}
    virtual double eval(EvalContext& ctx) = 0;
};

template<typename Op>
class UnaryOp : public Expression {
    Expression* _expr;
public:
    UnaryOp(Expression* expr)
        : _expr(expr) {}
    ~UnaryOp() {
        delete _expr;
    }
    double eval(EvalContext& ctx) {
        Op o;
        double x = _expr->eval(ctx);
        return o(x);
    }
};

template<typename Op>
class BinaryOp : public Expression {
    Expression* _a;
    Expression* _b;
public:
    BinaryOp(Expression* a, Expression* b)
        : _a(a), _b(b) {}
    ~BinaryOp() {
        delete _a;
        delete _b;
    }
    virtual double eval(EvalContext& ctx) {
        Op o;
        double a = _a->eval(ctx);
        double b = _b->eval(ctx);
        return o(a, b);
    }
};

class ModuloOp : public Expression {
    Expression* _a;
    Expression* _b;
public:
    ModuloOp(Expression* a, Expression* b)
        : _a(a), _b(b) {}
    ~ModuloOp() {
        delete _a;
        delete _b;
    }
    virtual double eval(EvalContext& ctx) {
        double a = _a->eval(ctx);
        double b = _b->eval(ctx);
        return fmod(a, b);
    }
};


class RaisePowerOp : public Expression {
    Expression* _a;
    Expression* _b;
public:
    RaisePowerOp(Expression* a, Expression* b)
        : _a(a), _b(b) {}
    ~RaisePowerOp() {
        delete _a;
        delete _b;
    }
    virtual double eval(EvalContext& ctx) {
        double a = _a->eval(ctx);
        double b = _b->eval(ctx);
        return pow(a, b);
    }
};

class TernaryOpExpr : public Expression {
    Expression* _test;
    Expression* _ifTrue;
    Expression* _ifFalse;
public:
    TernaryOpExpr(Expression* test, Expression* ifTrue, Expression* ifFalse)
        : _test(test), _ifTrue(ifTrue), _ifFalse(ifFalse) {
    }
    ~TernaryOpExpr() {
        delete _test;
        delete _ifTrue;
        delete _ifFalse;
    }
    virtual double eval(EvalContext& ctx) {
        bool test = static_cast<bool>(_test->eval(ctx));
        if (test) {
            return _ifTrue->eval(ctx);
        } else {
            return _ifFalse->eval(ctx);
        }
    }
};

class ExprList : public Expression {
    Expression* _expr;
    ExprList* _list;
public:
    ExprList(Expression* expr, ExprList* list = nullptr)
        : _expr(expr), _list(list) {}
    ~ExprList() {
        delete _expr;
        if (nullptr != _list) {
            delete _list;
        }
    }
    virtual double eval(EvalContext& ctx) { 
        double value = _expr->eval(ctx);
        if (nullptr != _list) {
            return _list->eval(ctx);
        } else {
            return value;
        }
    }
};

class ConstantExpr : public Expression {
    double _x;
public:
    ConstantExpr(double x) : _x(x) {}
    virtual double eval(EvalContext&) { return _x;  }
};

class VariableExpr : public Expression {
    EvalSymbol _symbol;
public:
    VariableExpr(const std::string& name) : _symbol(name) {}
    virtual double eval(EvalContext& ctx);
    void notFound() {
        throw EvalException(std::string("Variable \"") + _symbol.name() + "\" not defined.");
    }
};

class FunctionExpr : public Expression {
protected:
    EvalSymbol _symbol;
public:
    FunctionExpr(const std::string& name) : _symbol(name) {}
    EvalSymbol& symbol() { return _symbol;  }
    void notFound() {
        throw EvalException(std::string("Function \"") + _symbol.name() + "\" not defined.");
    }
};

class NilaryFunction : public FunctionExpr {
public:
    NilaryFunction(const std::string& name) 
        : FunctionExpr(name) {}
    virtual double eval(EvalContext& ctx);
};

class UnaryFunction : public FunctionExpr {
    Expression* _arg;
public:
    UnaryFunction(std::string& name, Expression* arg) 
        : FunctionExpr(name), _arg(arg) {}
    ~UnaryFunction() {
        delete _arg;
    }
    double eval(EvalContext& ctx);
};

class BinaryFunction : public FunctionExpr {
    Expression* _arg1, * _arg2;
public:
    BinaryFunction(std::string& name, Expression* arg1, Expression* arg2) 
        : FunctionExpr(name), _arg1(arg1), _arg2(arg2) {}
    ~BinaryFunction() {
        delete _arg1;
        delete _arg2;
    }
    double eval(EvalContext& ctx);
};
