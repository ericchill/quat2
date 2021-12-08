
#include <sstream>
#include <iostream>
#include <fstream>
#include <cctype>	// toupper


#include "CReplacements.h"

pathname::pathname(const char* c) : std::string(_convert_slash(c)) {}
pathname::pathname(const std::string& s) : std::string(_convert_slash(s.c_str())) {}

std::string pathname::_convert_slash(const char* c) const {
	std::string s(c);
#ifdef WIN32
	for (unsigned int i = 0; i < s.length(); ++i) {
		if (s[i] == '\\') {
			s[i] = '/';
		}
	}
#endif
	return s;
}

void pathname::uppercase() {
	for (size_type i = 0; i < length(); ++i) {
		at(i) = ::toupper(at(i));
	}
}

std::string pathname::path() const {
	return substr(0, rfind("/")+1);
}

std::string pathname::filename() const {
	size_type last_slash = rfind("/");
	if (last_slash > length()) {
		return *this;
	}
	return substr(last_slash+1);
}

void pathname::filename(const char* c) {
	*this = path();
	*this += c;
}

std::string pathname::basename() const {
	size_type last_slash = rfind("/");
	size_type last_dot = rfind(".");
	if (last_slash < length() && last_dot < last_slash) {
		last_dot = (size_type)(-1);
	}
	return substr(last_slash+1, last_dot-last_slash-1);
}

std::string pathname::ext() const {
	size_type last_slash = rfind("/");
	size_type last_dot = rfind(".");
	if (last_slash < length() && last_dot < last_slash) {
		return std::string("");
	}
	if (last_dot >= length()) {
		return std::string("");
	}
	return substr(last_dot);
}

void pathname::ext(const char* c) {
	*this = path() + basename();
	if (c[0] != '.') {
		*this += '.';
	}
	*this += c;
}

bool pathname::exists() const {
	std::ifstream f;
	f.open(c_str(), std::ios::binary);
	if (!f) {
		return false;
	}
	f.close();
	return true;
}

void pathname::seperate(std::string& _path, std::string& _filename, std::string& _ext) const {
	_path = path();
	_filename = basename();
	_ext = ext();
}

std::string pathname::next_name(int digits) const {
	std::string _path = path();
	std::string _filename = basename();
	std::string _ext = ext();
	if (_filename.length() == 0) _filename = "Noname";
	size_type nr = _filename.find_last_not_of("0123456789");
	std::string base(_filename.substr(0, nr+1));
	size_type stellen = _filename.length() - nr - 1;
	int number = 0;
	if (stellen > 0) {
		std::stringstream ss;
		ss << _filename.substr(nr+1, stellen);
		ss >> number;
	}
	if (stellen == 0) {
		stellen = digits;
		number = 0;
		if (base[base.length()-1] != '-') base += '-';
	}
	++number;
	std::stringstream tmp;
	tmp << number;
	if (tmp.str().length() > stellen) ++stellen;
	std::stringstream s;
	s << _path << base;
	s.fill('0');
	s.width(stellen);
	s << number << _ext;
	return s.str();
}

void pathname::auto_name(int digits) {
	while (exists()) {
		*this = next_name(digits);
	}
}
