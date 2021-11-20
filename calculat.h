#pragma once

#include <string>

typedef char (*funcptr) (double* a, const double* b);
/* applies operation on a and b and stores result in a */
/* e.g.: a = a/b; (operation = division) */
/* or: a = sin(b) */

struct function_decl {
    const char* name;  /* string, name of function e.g. "sin" */
    funcptr func;      /* pointer to function that performs operation */
};

class progtype {
public:

    /* Resets variables in program "prog" */
    void reset();

    /* function to compile a formula given as a string (aa) to a program (prog)  */
    /* for use with  "calculate". */
    /* Must be called before function "calculate"! */
    /* returns 0, if successful, else an error message (in wrong) */
    int compile(char* errorMSg, size_t maxErrorLen, const char* expr);

    /* function to calculate a value by a formula */
    /* The formula has to be translated to a program "prog" first (see "progtype::compile") */
    /* values for variables in the formula-string may be set by function */
    /* "setVariable" */
    /* The calculated value will be returned. */
    double calculate(char* notdef);

    /* Declares an user defined function */
    /* name: Name of function, e.g. "sin" */
    /* f: pointer to a function that is connected to name */
    /* note: it has a special calling syntax: f(double *a, double *b) */
    /*       the function has to calculate something with a and b and to store the result in a */
    int declareFunction(const char* name, funcptr f);

    /* Sets the variable referenced by "name" or "handle" to the value "value" */
    /* in the program "prog". */
    /* To define a new variable, set handle to 255 */
    /* in further calls use the returned handle, not the name (for speed reasons) */
    int setVariable(const char* name, unsigned char* handle, double value);

    bool isNumber(const std::string& s);
    bool isVar(const std::string& s);
    int varNumber(const std::string& s);
    bool isRegister(const std::string& s);
    int isFunction(const std::string& s);

private:

    static constexpr size_t maxComplication = 100;

    // Modifies expr (with register name?)
    int doTranslate(char* errorMsg, size_t maxErrorLength, std::string& expr);

    /* None of these members has to be referenced by the user, they are filled automatically */
    function_decl _availFunctions[30];   // pointers to defined functions (e.g. "sin")
    int _varDef;                         // how many defined variables
    int _availFnCount;                   // how many defined variables, and functions
    char _varNames[30][12];              // names of defined variables (max 30, max 12 char.)
    double _varValues[30];               // here the values of the variables are stored
    funcptr _program[maxComplication];   // 100 instructions (function calls) which _a program consists of */
    double* _a[maxComplication], * _b[maxComplication], _z[maxComplication];  // the data passed to the above functions

    int _regsCount;
    double _reg[26];

    int _oCount;
};



