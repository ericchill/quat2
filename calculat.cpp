/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997,98 Dirk Meyer */
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

#include <string.h>
#include <stdio.h>    /* "sprintf" */
#include <ctype.h>    /* "tolower" */
#include <stdlib.h>
#include <math.h>
#include "calculat.h"
#include "memory.h"
#include <crtdbg.h>

char f_set(double* a, const double* b);
char f_add(double* a, const double* b);
char f_sub(double* a, const double* b);
char f_mul(double* a, const double* b);
char f_div(double* a, const double* b);
char f_pow(double* a, const double* b);
char f_sin(double* a, const double* b);
char f_cos(double* a, const double* b);
char f_sqr(double* a, const double* b);
char f_sqrt(double* a, const double* b);
char f_exp(double* a, const double* b);
char f_ln(double* a, const double* b);
char f_atan(double* a, const double* b);
char f_asin(double* a, const double* b);
char f_acos(double* a, const double* b);
char f_round(double* a, const double* b);
char f_trunc(double* a, const double* b);
char f_abs(double* a, const double* b);
char f_tan(double* a, const double* b);
char f_sinh(double* a, const double* b);
char f_cosh(double* a, const double* b);
char f_tanh(double* a, const double* b);
char f_random(double* a, const double* b);

constexpr char *operators = "+-*/^";
constexpr char *numbers = "0123456789.";
constexpr char *registers = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr char *prio1 = "^";
constexpr char *prio2 = "*/";
constexpr char *prio3 = "+-";

size_t strindex(const char* toSearch, char value) {
    return strchr(toSearch, value) - toSearch;
}


/* number of functions defined the line below */
constexpr char* functions[] = {
    "sin", "cos", "sqr", "sqrt", "exp", "ln",
    "round", "trunc", "abs", "tan", "random",
    "atan", "asin", "acos",
    "sinh", "cosh", "tanh"
};

constexpr int fncount = sizeof(functions) / sizeof(functions[0]);

/* Define functions for mathematical operations */
char f_set(double* a, const double* b) {
    *a = *b;
    return 0;
}

char f_add(double* a, const double* b) {
    *a = *a + (*b);
    return 0;
}

char f_sub(double* a, const double* b) {
    *a = *a - (*b);
    return 0;
}

char f_mul(double* a, const double* b) {
    *a = (*a) * (*b);
    return 0;
}

char f_div(double* a, const double* b) {
    if (fabs(*b) < 1E-30) {
        return 1;   /* Division by zero */
    }
    *a = (*a) / (*b);
    return 0;
}

char f_pow(double* a, const double* b) {
    if (0 == *a && *b <= 0) {
        return 1;   /* would be division by zero or 0^0 */
    }
    if (*a < 0 && ceil(*b) != *b) {
        return 1;   /* would be root of neg. number */
    }
    *a = pow(*a, *b);
    return 0;
}

char f_sin(double* a, const double* b) {
    *a = sin(*b);
    return 0;
}

char f_cos(double* a, const double* b) {
    *a = cos(*b);
    return 0;
}

char f_sqr(double* a, const double* b) {
    *a = (*b) * (*b);
    return 0;
}

char f_sqrt(double* a, const double* b) {
    if (*b < 0) {
        return 1;
    }
    *a = sqrt(*b);
    return 0;
}

char f_exp(double* a, const double* b) {
    *a = exp(*b);
    return 0;
}

char f_ln(double* a, const double* b) {
    if (*b <= 0) {
        return 1;
    }
    *a = log(*b);
    return 0;
}

char f_atan(double* a, const double* b) {
    *a = atan(*b);
    return 0;
}

char f_asin(double* a, const double* b) {
    *a = asin(*b);
    return 0;
}

char f_acos(double* a, const double* b) {
    *a = acos(*b);
    return 0;
}

char f_round(double* a, const double* b) {
    if (*b - floor(*b) >= 0.5) {
        *a = ceil(*b);
    } else {
        *a = floor(*b);
    }
    return 0;
}

char f_trunc(double* a, const double* b) {
    *a = floor(*b);
    return 0;
}

char f_abs(double* a, const double* b) {
    *a = fabs(*b);
    return 0;
}

char f_tan(double* a, const double* b) {
    if (cos(*b) == 0) {
        return 1;
    } else {
        *a = tan(*b);
    }
    return 0;
}

char f_sinh(double* a, const double* b) {
    *a = sinh(*b);
    return 0;
}

char f_cosh(double* a, const double* b) {
    *a = cosh(*b);
    return 0;
}

char f_tanh(double* a, const double* b) {
    *a = tanh(*b);
    return 0;
}

char f_random(double* a, const double* b) {
    *a = *b * rand() / RAND_MAX;
    return 0;
}


std::string getObjectBefore(const std::string& s, int p) {
    /*
       s ... string to look for object
       p ... position to start looking
    */

    int i = p - 1;
    while (i >= 0 && strchr(operators, s[i]) == NULL) --i;
    std::string result(s.substr(i + 1, p - i - 1));
    return result;
}

std::string getObjectAfter(const std::string& s, int p) {
    /*
       s ... string to look for object
       p ... position to start looking
    */

    unsigned int i = p + 1;
    while (i < s.size() && strchr(operators, s[i]) == NULL) {
        i++;
    }
    std::string result(s.substr(p + 1, i - p - 1));
    return result;
}


bool progtype::isVar(const std::string& s) {
    int i = 0;
    while (i < _varDef && s.compare(0, strlen(_varNames[i]), _varNames[i]) != 0) {
        i++;
    }
    return i < _varDef;
}

int progtype::varNumber(const std::string& s) {
    int i = 0;
    while (i < _varDef && s.compare(0, strlen(_varNames[i]), _varNames[i]) != 0) {
        i++;
    }
    if (i < _varDef) {
        return i;
    } else {
        return 255;
    }
}

bool progtype::isNumber(const std::string& s) {
    char* p;

    if (0 == s.size()) {
        return false;
    }
    (void)strtod(s.c_str(), &p);
    return '\0' == *p;
}

bool progtype::isRegister(const std::string& s) {
    if (0 == s.size()) {
        return 0;
    }
    if (strchr(registers, s[0]) != NULL) {
        int i = 1;
        while (isspace(s[i]) && i < s.size() - 1) {
            i++;
        }
        return i == 1 || isspace(s[i]);
    }
    return false;
}

int progtype::isFunction(const std::string& s)
/* return code >=0: number of integrated function */
/* >=0 and bit 7 set (>=128): number of declared function */
/* -1: error, none of the above */
{
    int i;

    for (i = 0;
        (i < _availFnCount
            && 0 != s.compare(0, strlen(_availFunctions[i].name), _availFunctions[i].name) != 0);
        i++);
    if (i == _availFnCount) {
        for (i = 0;
            (i < fncount && s.compare(0, strlen(functions[i]), functions[i]) != 0);
            i++);
        if (i == fncount) {
            return -1;
        } else {
            return i;
        }
    }
    return i + 128;
}

int progtype::declareFunction(const char* name, funcptr f)
/* returns -1 if too many functions declared */
{
    int i = isFunction(name);
    if (i >= 128) {
        _availFunctions[i - 128].func = f;
        return 0;
    }
    if (_availFnCount == 30) {
        return -1;
    }
    _availFunctions[_availFnCount].func = f;
    _availFunctions[_availFnCount++].name = name;
    return 0;
}

/* Set value of variable / create new variable */
int progtype::setVariable(
    const char* name,
    unsigned char* handle,
    double value) {

    if (*handle < _varDef) {
        _varValues[*handle] = value;
    } else {
        int i = varNumber(name);
        if (255 == i) {
            if (_varDef == 30) {
                return 100;
            }
            *handle = _varDef;
            strcpy_s(_varNames[_varDef], sizeof(_varNames[0]), name);
            _varValues[_varDef++] = value;
        } else {
            *handle = i;
            _varValues[i] = value;
        }
    }
    return -1;
}

void progtype::reset() {
    _varDef = 0;
    for (int i = 0; i < 30; i++) {
        _availFunctions[i].name = "";
    }
    _availFnCount = 0;
    _program[0] = nullptr;
}

double progtype::calculate(char* notdef) {
    if (NULL == _program[0]) {
        return 0.0;
    }
    *notdef = 0;
    int i;
    for (i = 0; _program[i] != NULL && i < 100 && 0 == *notdef; i++) {
        notdef += _program[i](_a[i], _b[i]);
    }
    return *(_a[i - 1]);
}

int progtype::doTranslate(char* errorMsg, size_t maxErrorLen, std::string& expr) {
    LexicallyScopedPtr<unsigned int> priori = new unsigned int[expr.size()];
    std::string suba;
    int error, fct;

    errorMsg[0] = '\0';
    if (strchr(prio3, expr[0]) != NULL) {
        expr.insert(0, 1, '0');
    }
    int bropen = -1;
    int brcount = 0;
    for (int j = 0; j < expr.size(); j++) {
        if (expr[j] == '(') {
            if (bropen == -1) {
                bropen = j;
            }
            brcount++;
        }
        if (expr[j] == ')') {
            brcount--;
        }
        if ((bropen != -1) && (0 == brcount)) {
            std::string suba = expr.substr(bropen + 1, j - (bropen + 1));
            error = doTranslate(errorMsg, maxErrorLen, suba);
            if (error != 0) {
                return error;
            }
            std::string obj1 = getObjectBefore(expr, bropen);
            std::string obj2 = getObjectAfter(expr, bropen);
            std::string subaFragment = suba.substr(0, j - (bropen + 1));
            expr = expr.replace(bropen, subaFragment.size(), subaFragment);
            expr.replace(bropen + subaFragment.size(), 2, 2, ' ');
            fct = isFunction(obj1);
            if (fct != -1) {

                if ((fct & 128) != 0) {
                    /* Create command */
                    _program[_oCount] = _availFunctions[fct & ~128].func;
                    _a[_oCount] = &_reg[strindex(registers, suba[0])];
                    _b[_oCount] = &_reg[strindex(registers, suba[0])];

                } else if (fct >= 0) {

                    /* Declare new function */
                    int k = 0;
                    switch (fct) {
                    case 0:  k = declareFunction("sin", f_sin); break;
                    case 1:  k = declareFunction("cos", f_cos); break;
                    case 2:  k = declareFunction("sqr", f_sqr); break;
                    case 3:  k = declareFunction("sqrt", f_sqrt); break;
                    case 4:  k = declareFunction("exp", f_exp); break;
                    case 5:  k = declareFunction("ln", f_ln); break;
                    case 6:  k = declareFunction("round", f_round); break;
                    case 7:  k = declareFunction("trunc", f_trunc); break;
                    case 8:  k = declareFunction("abs", f_abs); break;
                    case 9:  k = declareFunction("tan", f_tan); break;
                    case 10: k = declareFunction("random", f_random); break;
                    case 11: k = declareFunction("atan", f_atan); break;
                    case 12: k = declareFunction("asin", f_asin); break;
                    case 13: k = declareFunction("acos", f_acos); break;
                    case 14: k = declareFunction("sinh", f_sinh); break;
                    case 15: k = declareFunction("cosh", f_cosh); break;
                    case 16: k = declareFunction("tanh", f_tanh); break;
                    default: k = -255;
                    }

                    if (k == -1) {
                        sprintf_s(errorMsg, maxErrorLen, "%s", "Too many functions declared (max. 30)");
                        return -1;
                    }
                    if (k == -255) {
                        sprintf_s(errorMsg, maxErrorLen, "%s", "Internal error #1");
                        return -1;
                    }
                    /* Create command */
                    _program[_oCount] = _availFunctions[_availFnCount - 1].func;
                    _a[_oCount] = &_reg[strindex(registers, suba[0])];
                    _b[_oCount] = &_reg[strindex(registers, suba[0])];
                }
                _oCount++;
                bropen -= static_cast<int>(obj1.size());
            }
            expr.replace(bropen, suba.size() + 1, suba.size() + 1, ' ');
            expr[bropen] = suba[0];
            if ((1 == bropen) && (j == expr.size())) {
                sprintf_s(errorMsg, maxErrorLen, "Internal error #2");
                return 244;
            }
            bropen = -1;
            brcount = 0;
        }
    }
    memset(priori, 0, suba.size());
    for (int i = 0; i < expr.size(); i++) {
        if (strchr(prio3, expr[i]) != NULL) priori[i] = 3;
        else if (strchr(prio2, expr[i]) != NULL) priori[i] = 2;
        else if (strchr(prio1, expr[i]) != NULL) priori[i] = 1;
    }
    int sign = 0;
    for (unsigned int j = 1; j < 4; j++) {
        for (int i = 0; i < expr.size(); i++) {
            if (priori[i] == j) {
                sign = 1; /* string contains calculation signs */
                std::string obj1 = getObjectBefore(expr, i);
                std::string obj2 = getObjectAfter(expr, i);
                if (isNumber(obj1) || isVar(obj1)) {
                    /* create sequence "load register" */
                    if (i - obj1.size() < 0) {
                        sprintf_s(errorMsg, maxErrorLen, "Internal error #3");
                        return 1;
                    } else {
                        expr[i - obj1.size()] = registers[_regsCount];
                    }
                    _program[_oCount] = f_set;
                    _a[_oCount] = &_reg[_regsCount];
                    if (isNumber(obj1)) {
                        _z[_oCount] = atof(obj1.c_str());
                        _b[_oCount] = &_z[_oCount];
                    } else {
                        _b[_oCount] = &_varValues[varNumber(obj1)];
                    }
                    _a[_oCount + 1] = &_reg[_regsCount];
                    _oCount++;
                    _regsCount++;
                    if (_oCount > progtype::maxComplication) {
                        sprintf_s(errorMsg, maxErrorLen, "Formula too complex.");
                        return 1;
                    }
                } else if (!isRegister(obj1) && !isVar(obj1)) {
                    sprintf_s(errorMsg, maxErrorLen, "Unknown object on left of %c:  \"%s\" ", expr[i], obj1.c_str());
                    return 1;
                }
                /* create operation */
                switch (expr[i]) {
                case '+': _program[_oCount] = f_add; break;
                case '-': _program[_oCount] = f_sub; break;
                case '*': _program[_oCount] = f_mul; break;
                case '/': _program[_oCount] = f_div; break;
                case '^': _program[_oCount] = f_pow; break;
                }
                if (isRegister(obj1)) {
                    _a[_oCount] = &_reg[strindex(registers, obj1[0])];
                }
                if (isRegister(obj2)) {
                    _b[_oCount] = &_reg[strindex(registers, obj2[0])];
                }  else if (isNumber(obj2)) {
                    _z[_oCount] = atof(obj2.c_str());
                    _b[_oCount] = &_z[_oCount];
                } else if (isVar(obj2)) {
                    _b[_oCount] = &_varValues[varNumber(obj2)];
                } else {
                    sprintf_s(errorMsg, maxErrorLen, "Unknown object B %s", obj2.c_str());
                    return 1;
                }
                if (obj2.size() + obj1.size() > expr.size()) {
                    sprintf_s(errorMsg, maxErrorLen, "Internal error #4"); 
                    return 1;
                } else {
                    size_t numSpaces = obj1.size() + obj2.size();
                    expr.replace(i - obj1.size() + 1, numSpaces, numSpaces, ' ');
                }
                _oCount++;
                if (_oCount > progtype::maxComplication) {
                    sprintf_s(errorMsg, maxErrorLen, "Formula too complex.");
                    return 1;
                }
            }
        }
    }
    if (0 == sign) {
        if (isRegister(expr)) {
            errorMsg[0] = '\0';
            return 0;
        }
        if (isNumber(expr)) {
            _program[_oCount] = f_set;
            _a[_oCount] = &_reg[_regsCount];
            _b[_oCount] = &_z[_oCount];
            _z[_oCount] = atof(expr.c_str());
            size_t numSpaces = expr.size();
            expr.replace(0, numSpaces, numSpaces, ' ');
            expr[0] = registers[_regsCount];
            _oCount++; 
            _regsCount++;
        } else if (isVar(expr)) {
            _program[_oCount] = f_set;
            _a[_oCount] = &_reg[_regsCount];
            _b[_oCount] = &_varValues[varNumber(expr)];
            size_t numSpaces = expr.size();
            expr.replace(0, numSpaces, numSpaces, ' ');
            expr[0] = registers[_regsCount];
            _oCount++; _regsCount++;
        } else {
            sprintf_s(errorMsg, maxErrorLen, "Unknown object C %s", expr.c_str());
            return 1;
        }
        if (_oCount > progtype::maxComplication) {
            sprintf_s(errorMsg, maxErrorLen, "Formula too complex.");
            return 1;
        }
    }
    errorMsg[0] = '\0';
    return 0;
}

int progtype::compile(char* errorMsg, size_t maxErrorLen, const char* expr) {
    assert(_CrtCheckMemory());
    size_t exprLen = strlen(expr) + 1;
    std::string cleanExpr;

    errorMsg[0] = '\0';
    int j = 0;
    for (int i = 0; i < strlen(expr) && expr[i] != '#'; i++) {
        if (!isspace(expr[i])) {
            cleanExpr.push_back(tolower(expr[i]));
            j++;
        }
    }
    cleanExpr[j] = '\0';
    int brcount = 0;
    for (int i = 0; i < cleanExpr.size(); i++) {
        if (cleanExpr[i] == '(') brcount++;
        if (cleanExpr[i] == ')') brcount--;
        /* error with brackets */
        if (brcount < 0) {
            sprintf_s(errorMsg, maxErrorLen, "Too many closing parentheses.");
            return 255;
        }
    }
    if (brcount != 0) {
        sprintf_s(errorMsg, maxErrorLen, "Missing close parenthesis.");
        return 255;
    }
    _regsCount = 0; 
    _oCount = 0;
    int error = doTranslate(errorMsg, maxErrorLen, cleanExpr);
    if (0 != error) {
        assert(false);
    }
    if (strlen(errorMsg) == 0) {
        sprintf_s(errorMsg, maxErrorLen, "Parsing OK");
    }
    _program[_oCount] = NULL;
    //assert(_CrtCheckMemory());
    return error;
}
