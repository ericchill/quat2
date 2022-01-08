#include "ExprParse.h"
#include "grammar.h"

ExprCompiler* ExprCompiler::currentCompiler = nullptr;


Expression* ExprCompiler::translate(const std::string& exprStr) {
	std::istringstream input(exprStr);
	ExprCompiler* compiler = new ExprCompiler(input);
	ExprCompiler::currentCompiler = compiler;
	yy::parser parse(*compiler);
	if (0 == parse()) {
		return compiler->result;
	} else {
		return nullptr;
	}
}

constexpr char FloatChar[] = "0123456789.eE+-";
constexpr char FloatInitChar[] = "0123456789.";
constexpr char OpChar[] = "!%()*+,-/:<=>?^";
constexpr char TwoCharOpInit[] = "!<=>";

std::unordered_map<std::string, ExprCompiler::SymbolMaker_t> ExprCompiler::_operatorDict {
	std::make_pair("!=", yy::parser::make_NOT_EQUAL_TO),
	std::make_pair("%", yy::parser::make_MODULO),
	std::make_pair("(", yy::parser::make_OPEN_PAREN),
	std::make_pair(")", yy::parser::make_CLOSE_PAREN),
	std::make_pair("*", yy::parser::make_TIMES),
	std::make_pair("+", yy::parser::make_PLUS),
	std::make_pair(",", yy::parser::make_COMMA),
	std::make_pair("-", yy::parser::make_MINUS),
	std::make_pair("/", yy::parser::make_DIVIDE),
	std::make_pair(":", yy::parser::make_COLON),
	std::make_pair("<", yy::parser::make_LESS),
	std::make_pair("<=", yy::parser::make_LESS_EQUAL),
	std::make_pair("==", yy::parser::make_EQUAL_TO),
	std::make_pair(">", yy::parser::make_GREATER),
	std::make_pair(">=", yy::parser::make_GREATER_EQUAL),
	std::make_pair("?", yy::parser::make_QUESTION),
	std::make_pair("^", yy::parser::make_POWER)
};

yy::parser::symbol_type ExprCompiler::yylex() {
	skipBlanks();
	static std::string bogusFilename("-");
	_curPos = yy::parser::location_type(&bogusFilename, _line, _col);
	if (!_input || -1 == peek() || '#' == peek()) {
		return yy::parser::make_YYEOF(_curPos);
	}
	try {
		char next = peek();
		if (isalpha(next) || '_' == next) {
			return yy::parser::make_IDENTIFIER(consumeIdentifier(), _curPos);
		} else if (strchr(OpChar, next)) {
			return parseOperator();
		} else if (strchr(FloatInitChar, next)) {
			return yy::parser::make_FLOAT(parseNumber(), _curPos);
		} else {
			std::cerr << "Unexpected '" << next << "' (" << static_cast<int>(next) << ")." << std::endl;
			throw ParseException(std::string("Unexpected character"), _curPos);
		}
	} catch (ParseException& ex) {
		return yy::parser::symbol_type(
			yy::parser::token::YYerror,
			std::string(ex.what()),
			_curPos);
	}
}

char ExprCompiler::getNextChar() {
	nextCol();
	return static_cast<char>(_input.get());
}

void ExprCompiler::skipBlanks() {
	while (_input && isspace(peek())) {
		getNextChar();
	}
}

std::string ExprCompiler::consumeIdentifier() {
	std::string res;
	char c = peek();
	// We already know that the first character isn't a digit.
	while (isalpha(c) || isdigit(c) || c == '_') {
		res.push_back(getNextChar());
		c = peek();
	}
	return res;
}

double ExprCompiler::parseNumber() {
	std::string numstr;
	bool digits = false;
	bool exponent = false;
	bool decimal = false;

	while (strchr(FloatChar, peek()) != nullptr) {
		char c = getNextChar();
		if (isdigit(c)) {
			digits = true;
		} else if (c == '.') {
			if (exponent || decimal) {
				error("Bad number: Out of place '.'");
			}
			decimal = true;
		} else if (c == 'e' || c == 'E') {
			if (!digits) {
				error("Bad number: No digits before 'E'");
			}
			exponent = true;
			digits = true;
			decimal = false;
		} else if (c == '-' || c == '+') {
			if (!exponent) {
				_input.unget();
				_col--;
				break;
			}
			if (digits || decimal) {
				error("Bad number: Out of place +/-");
			}
			decimal = true;
		}
		numstr.push_back(c);
	}
	if (!digits) {
		error("Bad number: No digits");
	}
	return atof(numstr.c_str());
}

yy::parser::symbol_type ExprCompiler::parseOperator() {
	std::string opName;
	char c = getNextChar();
	opName.push_back(c);
	if (nullptr != strchr(TwoCharOpInit, c) && '=' == peek()) {
		opName.push_back(getNextChar());
	}
	try {
		SymbolMaker_t maker = _operatorDict.at(opName);
		return maker(opName, _curPos);
	} catch (std::out_of_range&) {
		error("Unknown symbol");
		return yy::parser::make_YYerror(_curPos);
	}
}

void
yy::parser::error(const location_type& l, const std::string& m) {
	std::cerr << l << ": " << m << '\n';
}


yy::parser::symbol_type yylex(ExprCompiler& drv) {
	return drv.yylex();
}
