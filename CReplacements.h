#pragma once

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string>
//#include <iostream>
extern "C" {
#include <stdio.h>
#include <stdlib.h>
}

#ifndef NO_NAMESPACE
using namespace std;
#endif

class pathname: public string
{
public:
	pathname() {}
	pathname(const char*);
	pathname(const string&);
	void uppercase();
	string path() const;
	string file() const;
	void file(const char*);
	string filename() const;
	string ext() const;
	void ext(const char*);
	bool exists() const;
	void seperate(string& path, string& filename, string& ext) const;
	string next_name(int digits = 3) const;
	void auto_name(int digits = 3);
private:
	string _convert_slash(const char *) const;
};

inline void i_assert(const char *f, int l, const char *expr) {
  fprintf(stderr, "Assertion failed: file: %s, line %d, expr: %s\n",
	  f, l, expr);
  exit(1);
}

#ifdef assert
#undef assert
#endif

#ifdef __STDC__
#define assert(e)       ((e) ? (void)0 : i_assert(__FILE__, __LINE__, #e))
#else   /* PCC */
#define assert(e)       ((e) ? (void)0 : i_assert(__FILE__, __LINE__, "e"))
#endif


