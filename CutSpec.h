#pragma once

#include "json.h"
#include "qmath.h"

class CutSpec : public JSONSerializable {
public:
    CutSpec() {
        reset();
    }
    void reset();
    void print();
    size_t count() const;
    bool getPlane(size_t i, Vec3& normal, Vec3& point) const;
    bool setPlane(size_t i, const Vec3& normal, const Vec3& point);
    bool addPlane(const Vec3& normal, const Vec3& point);
    bool deletePlane(size_t i);
    const Vec3* normals() const { return _normal; }
    const Vec3* points() const { return _point; }

    bool cutaway(const Vec3& x) const;
    bool cutnorm(const Quat& x1, const Quat& x2, Quat& nq) const;

    CutSpec(const json::value& jv);
    virtual json::value toJSON() const;

    static constexpr size_t maxCuts = 6;
    /* intersection objects definitions (only planes for now) */
    /* every structure in this section must have */
    /* _a char "cut_type" as first element */

private:
    size_t _count;
    Vec3 _normal[maxCuts];
    Vec3 _point[maxCuts];
};

CutSpec tag_invoke(const json::value_to_tag< CutSpec >&, json::value const& jv);
