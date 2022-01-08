%{
#include "ExprLexer.h"
#include <string.h>
#include <math.h>
#include "grammar.h"

LexerState lexerState;
%}

%%


-?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?)    {
    lexerState.nextCol(strlen(yytext));
    LexerToken token;
    token.dValue = atof(yytext);
    lexerState.setToken(token);
    return TOKEN_NUMBER;
    }

[ \t]   { lexerState.nextCol(strlen(yytext)); }               /* ignore but count white space */

[A-Za-z][A-Za-z0-9]*  {
    lexerState.nextCol(strlen(yytext));
    LexerToken token;
    token.name = yytext;
    lexerState.setToken(token);
    return TOKEN_IDENTIFIER;
}

","          {  return TOKEN_COMMA; }
"?"          {  return TOKEN_QUESTION; }
":"          {  return TOKEN_COLON; }
"<"          {  return TOKEN_LESS; }
">"          {  return TOKEN_GREATER; }
"=="         {  return TOKEN_EQUALS; }
"!="         {  return TOKEN_NOT_EQUALS; }
"("          {  return TOKEN_OPEN_PAREN; }
")"          {  return TOKEN_CLOSE_PAREN; }

"+"           {  return TOKEN_PLUS; }
"-"           {  return TOKEN_MINUS; }
"*"           {  return TOKEN_TIMES; }
"/"           {  return TOKEN_DIVIDE; }
"%"           {  return TOKEN_MODULO; }
"^"           {  return TOKEN_POWER; }


\n      { lexerState.nextLine(); /*return NEWLINE;*/ }

.       { lexerState.nextCol(strlen(yytext)); return yytext[0]; }
%%
/**
 * reset the line and column count
 */
void reset_lexer(void)
{
  lexerState.reset();
}

/**
 * yyerror() is invoked when the lexer or the parser encounter
 * an error. The error message is passed via *s
 *
 *
 */
void yyerror(char *s)
{
   lexerState.error(s); 

}

int yywrap(void)
{
  return 1;
}
