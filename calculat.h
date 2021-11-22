#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

typedef char (*funcptr) (double* a, const double* b);
/* applies operation on a and b and stores result in a */
/* e.g.: a = a/b; (operation = division) */
/* or: a = sin(b) */


class progtype {
public:

    progtype();

    /* Resets variables in program "prog" */
    void reset();

    /* function to compile a formula given as a string (aa) to a program (prog)  */
    /* for use with  "calculate". */
    /* Must be called before function "calculate"! */
    /* returns 0, if successful, else an error message (in wrong) */
    int compile(std::ostream& errorMsg, const char* expr);

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
    void declareFunction(const char* name, funcptr f);

    /* Sets the variable referenced by "name" or "handle" to the value "value" */
    /* in the program "prog". */
    /* To define a new variable, set handle to 255 */
    /* in further calls use the returned handle, not the name (for speed reasons) */
    int setVariable(const char* name, size_t* handle, double value);

    static constexpr size_t nullHandle = static_cast<size_t>(-1);

    bool isNumber(const std::string& s);
    bool isVar(const std::string& s);
    int varNumber(const std::string& s);
    bool isRegister(const std::string& s);
    bool isFunction(const std::string& s);

private:

    static constexpr size_t maxComplication = 100;

    void initFunctions();

    // Modifies expr (with register name?)
    int doTranslate(std::ostream& errorMsg, std::string& expr);

    /* None of these members has to be referenced by the user, they are filled automatically */
    std::map<std::string, funcptr> _functions;
    std::vector<std::string> _varNames;           // names of defined variables (max 30, max 12 char.)
    std::vector<double> _varValues;               // here the values of the variables are stored
    funcptr _program[maxComplication];            // 100 instructions (function calls) which _a program consists of */
    double* _a[maxComplication], * _b[maxComplication], _z[maxComplication];  // the data passed to the above functions

    int _regsCount;
    double _reg[26];

    int _oCount;
};



