#pragma once


#include <ostream>
#include <boost/json.hpp>

namespace json = boost::json;

class JSONSerializable {
public:
    virtual json::value toJSON() const = 0;
};

inline void tag_invoke(const json::value_from_tag&, json::value& jv, JSONSerializable const& o) {
    jv = o.toJSON();
}


void pretty_print(std::ostream& os, json::value const& jv, std::string* indent = nullptr);
