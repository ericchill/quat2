%require "3.2"
%language "c++"
%output "grammar.cpp"
%header "grammar.h"
%define api.token.raw
%define api.token.constructor
%define api.value.type variant
%define api.location.file none
%define parse.assert
%locations
%define parse.trace
%define parse.error detailed
%define parse.lac full

%{
#include "ExprParse.h"
#include "ExprEval.h"
//#define yylex ExprCompiler::currentCompiler->yylex
%}

%code requires {
class Expression;
class ExprCompiler;
}

// The parsing context.
%param { ExprCompiler& drv }

%token <double> FLOAT
%token <std::string> IDENTIFIER
%token <std::string> QUESTION
%token <std::string> COMMA
%token <std::string> COLON
%token <std::string> GREATER
%token <std::string> GREATER_EQUAL
%token <std::string> LESS
%token <std::string> LESS_EQUAL
%token <std::string> EQUAL_TO
%token <std::string> NOT_EQUAL_TO
%token <std::string> PLUS
%token <std::string> MINUS
%token <std::string> TIMES
%token <std::string> DIVIDE
%token <std::string> MODULO
%token <std::string> POWER
%token <std::string> OPEN_PAREN
%token <std::string> CLOSE_PAREN
%token <std::string> OPEN_BRACE
%token <std::string> CLOSE_BRACE
%token <std::string> OPEN_BRACKET
%token <std::string> CLOSE_BRACKET

%left COMMA;
%left QUESTION;
%right COLON;
%right GREATER GREATER_EQUAL LESS LESS_EQUAL EQUAL_TO NOT_EQUAL_TO;
%left PLUS MINUS;
%left TIMES DIVIDE MODULO;
%right POWER;

%nterm <Expression*> expr.result;
%nterm <Expression*> expr;
%nterm <Expression*> function;
%nterm <Expression*> value;

%%
%start expr.result;

expr.result:
	expr
		{ drv.result = $1; }

expr:
		expr QUESTION expr COLON expr 
			{ $$ = new TernaryOpExpr($1, $3, $5); }
	|	expr GREATER expr
			{ $$ = new BinaryOp<std::greater<double>>($1, $3); }
	|	expr GREATER_EQUAL expr
			{ $$ = new BinaryOp<std::greater_equal<double>>($1, $3); }
	|	expr LESS expr
			{ $$ = new BinaryOp<std::less<double>>($1, $3); }
	|	expr LESS_EQUAL expr
			{ $$ = new BinaryOp<std::less_equal<double>>($1, $3); }
	|	expr EQUAL_TO expr
			{ $$ = new BinaryOp<std::equal_to<double>>($1, $3); }
	|	expr NOT_EQUAL_TO expr
			{ $$ = new BinaryOp<std::not_equal_to<double>>($1, $3); }
	|	expr PLUS expr	
			{ $$ = new BinaryOp<std::plus<double>>($1, $3); }
	|	expr MINUS expr
			{ $$ = new BinaryOp<std::minus<double>>($1, $3); }
	|	expr TIMES expr
			{ $$ = new BinaryOp<std::multiplies<double>>($1, $3); }
	|	expr DIVIDE expr
			{ $$ = new BinaryOp<std::divides<double>>($1, $3); }
	|	expr MODULO expr
			{ $$ = new ModuloOp($1, $3); }
	|	expr POWER expr
			{ $$ = new RaisePowerOp($1, $3); }
	|	function
			{ $$ = $1; }
	|	value
			{ $$ = $1; }

function :
		IDENTIFIER OPEN_PAREN expr CLOSE_PAREN
			{ $$ = new UnaryFunction($1, $3); }
	|	IDENTIFIER OPEN_PAREN expr COMMA expr CLOSE_PAREN
			{ $$ = new BinaryFunction($1, $3, $5); }
	|	IDENTIFIER OPEN_PAREN CLOSE_PAREN
			{ $$ = new NilaryFunction($1); }

//	|	lambda OPEN_PAREN exprlist CLOSE_PAREN
//			{ $$ = new LambdaInvokeExpr($1, $2); }
//	|	OPEN_BRACE OPEN_PAREN varlist CLOSE_PAREN COLON expr CLOSE_BRACE
//			{ $$ = new LambdaExpr($2, $1); }

value :
		IDENTIFIER
			{ $$ = new VariableExpr($1); }
	|	FLOAT
			{ $$ = new ConstantExpr($1); }
	|	OPEN_PAREN expr CLOSE_PAREN
			{ $$ = $2; }
	|	MINUS expr
			{ $$ = new UnaryOp<std::negate<double>>($2); }

//exprlist($$) :
//	expr
//		{ $$ = $1; }

//exprlist($$) ::= exprlist COMMA expr
//		{ $$ = new ExprList($1, $2); }

//varlist($$) ::= identifier
//		{ $$ = new VarList; }

//varlist($$) ::= varlist COMMA identifier
//		{ $$ = new VarList($1, $2); }
