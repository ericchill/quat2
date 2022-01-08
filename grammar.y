%include {
#include "lexer.h"
#include "ExprEval.h"
}

%name exprParser

%token_prefix TOKEN_

%token_type {LexerToken}

%type expr {Expression}
%type function {Expression}
%type IDENTIFIER {VariableExpr}
%type NUMBER {ConstantExpr}
//%type lambda {Expression}
//%type arglist {Expression}
//%type varlist {Expression}

%left COMMA.
%left QUESTION.
%right COLON.
%right GREATER LESS EQUALS NOT_EQUALS.
%left PLUS MINUS.
%left TIMES DIVIDE MODULO.
%right POWER.


program ::= expr.

expr(res) ::= expr(A) QUESTION expr(B) COLON expr(C).
	{ res = TernaryOpExpr(A, B, C); }

expr(res) ::= expr(A) GREATER expr(B).
	{ res = BinaryOp<std::greater<double>>(A, B); }

expr(res) ::= expr(A) LESS expr(B).
	{ res = BinaryOp<std::less<double>>(A, B); }

expr(res) ::= expr(A) EQUALS expr(B).
	{ res = BinaryOp<std::equal_to<double>>(A, B); }

expr(res) ::= expr(A) NOT_EQUALS expr(B).
	{ res = BinaryOp<std::not_equal_to<double>>(A, B); }

expr(res) ::= expr(A) PLUS expr(B).	
	{ res = BinaryOp<std::plus<double>>(A, B); }

expr(res) ::= expr(A) MINUS expr(B).
	{ res = BinaryOp<std::minus<double>>(A, B); }

expr(res) ::= expr(A) TIMES expr(B).
	{ res = BinaryOp<std::multiplies<double>>(A, B); }

expr(res) ::= expr(A) DIVIDE expr(B).
	{ res = BinaryOp<std::divides<double>>(A, B); }

expr(res) ::= expr(A) MODULO expr(B).
	{ res = BinaryOp<std::modulus<double>>(A, B) ;}

expr(res) ::= expr(A) POWER expr(B).
	{ res = RaisePowerExpr(A, B); }

expr(res) ::= function(A).
	{ res = A; }

expr(res) ::= value(A).
	{ res = A; }

function(res) ::= IDENTIFIER(A) OPEN_PAREN expr(B) CLOSE_PAREN.
	{ res = UnaryFunction(A, B); }

function(res) ::= IDENTIFIER(A) OPEN_PAREN expr(B) COMMA expr(C) CLOSE_PAREN.
	{ res = BinaryFunction(A, B, C); }

function(res) ::= IDENTIFIER(A) OPEN_PAREN CLOSE_PAREN.
	{ res = NilaryFunction(A); }

//function(res) ::= lambda(A) OPEN_PAREN exprlist(B) CLOSE_PAREN.
//		{ res = LambdaInvokeExpr(A, B); }

//lambda(res) ::= OPEN_BRACE OPEN_PAREN varlist(A) CLOSE_PAREN COLON expr(B) CLOSE_BRACE.
//		{ res = LambdaExpr(B, A); }

value(res) ::= IDENTIFIER(A).
		{ res = VariableExpr(A.name); }

value(res) ::= NUMBER(A).
		{ res = ConstantExpr(A.dValue); }

value(res) ::= OPEN_PAREN expr(A) CLOSE_PAREN.
		{ res = A; }

value(res) ::= MINUS expr(A).
		{ res = UnaryOp<std::negate<double>>(A); }

//exprlist(res) ::= expr(A).
//		{ res = A; }

//exprlist(res) ::= exprlist(A) COMMA expr(B).
//		{ res = ExprList(A, B); }

//varlist(res) ::= IDENTIFIER(A).
//		{ res = VarList(A); }

//varlist(res) ::= varlist(A) COMMA IDENTIFIER(B).
//		{ res = VarList(A, B); }
