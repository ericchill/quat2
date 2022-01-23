#pragma once

#pragma warning ( push, 0 )
#include "grammar.h"
#pragma warning ( pop )
#include "common.h"

#include <string>
#include <sstream>
#include <unordered_map>


struct token;

class ParseException : public QuatException {
    size_t _line;
    size_t _col;
public:
    ParseException(const std::string& msg, size_t line, size_t col) 
        : QuatException(msg), _line(line), _col(col) {};
    ParseException(const std::string& msg, yy::parser::location_type& location)
        : QuatException(msg), _line(location.begin.line), _col(location.begin.column) {}
};


class ExprCompiler {
    yy::parser::location_type _curPos;
    yy::location::counter_type _line;
    yy::location::counter_type _col;
    std::istringstream& _input;

    typedef yy::parser::symbol_type (*SymbolMaker_t)(const std::string&, const yy::parser::location_type&);
    static std::unordered_map<std::string, SymbolMaker_t> _operatorDict;

public:
    static ExprCompiler* currentCompiler;

    static Expression* translate(const std::string& exprStr);

    ExprCompiler(std::istringstream& input) : _input(input), result(nullptr) {
        reset();
    }

    yy::parser::symbol_type yylex();

    void nextCol() {
        ++_col;
    }
    void nextLine() {
        _col = 0;
        ++_line;
    }
    void reset() {
        _line = 0;
        _col = 0;
    }
    void error(const std::string& msg) {
        std::ostringstream s;
        s << "Parse error: " << msg << " at line: " << _curPos.begin.line << " col: " << _curPos.begin.column;
        throw ParseException(s.str(), _curPos);
    }

    Expression* result;

private:
    char peek() { return static_cast<char>(_input.peek()); }
    char getNextChar();
    void skipBlanks();
    std::string consumeIdentifier();
    double parseNumber();
    yy::parser::symbol_type parseOperator();
};

yy::parser::symbol_type yylex(ExprCompiler& drv);

#define YY_NO_UNISTD_H 1
