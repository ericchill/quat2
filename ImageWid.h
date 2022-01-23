#pragma once

#include "common.h"

#pragma warning(push, 0)
#include <FL/fl_draw.H>
#include <FL/Fl_Widget.H>
#pragma warning(pop)

#include <cstring>
#include <map>

class ImageWid : public Fl_Widget
{
public:
	ImageWid(int x, int y, int w, int h, const char *label = 0);
	virtual ~ImageWid();
#if 0
	bool newImage(int w, int h);
#endif
	bool valid() const { return _data != 0; }
	void gray(int level);
	void white();
	void fill(Fl_Color& col);
	void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
	void set_line_segment(int x1, int x2, int y, uint8_t *d);
	void set_line(int n, uint8_t *d);
	void get_line(int n, uint8_t *d) const;

	unsigned long type() const { return 0x0a0b0c0d; }
protected:
	void size(int, int) {}
	virtual void draw();
	virtual int handle(int);
private:
	uint8_t *_data;
	bool* _lineDrawn;
};
