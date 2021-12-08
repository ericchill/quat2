#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>


class pathname: public std::string
{
public:
	pathname() {}
	pathname(const char*);
	pathname(const std::string&);
	void uppercase();
	std::string path() const;
	std::string filename() const;
	void filename(const char*);
	std::string basename() const;
	std::string ext() const;
	void ext(const char*);
	bool exists() const;
	void seperate(std::string& path, std::string& filename, std::string& ext) const;
	std::string next_name(int digits = 3) const;
	void auto_name(int digits = 3);
private:
	std::string _convert_slash(const char *) const;
};


class AssertionException : public std::exception {
private:
	const char* _msg;
public:
	AssertionException(const char* msg) : _msg(msg) {}
	virtual const char* what() const noexcept { return _msg; }
};

inline void i_assert(const char *f, int l, const char *expr) {
	std::stringstream ss;
	ss << "Assertion failed: file: " << f << ",  line " << l << ", expr: \"" << expr << "\"" << std::endl;
#ifdef NDEBUG
	throw AssertionException(ss.str().c_str());
#else
	std::cerr << ss.str();
	*(int*)nullptr = 1;
#endif
}


#ifdef assert
#undef assert
#endif

#ifdef __STDC__
#define assert(e)       ((e) ? (void)0 : i_assert(__FILE__, __LINE__, #e))
#else   /* PCC */
#define assert(e)       ((e) ? (void)0 : i_assert(__FILE__, __LINE__, "e"))
#endif


