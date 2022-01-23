#ifndef PIXWID_H
#define PIXWID_H

#pragma warning(push, 0)
#include <FL/Fl_Widget.H>
#include <FL/Fl_Pixmap.H>
#pragma warning(pop)

class PixWid : public Fl_Widget
{
public:
	PixWid(int x, int y, int w, int h, const char *label=0);
	virtual ~PixWid();
	void setPixmap(const char* const* data);
	static const unsigned long TYPE;
	unsigned long type() { return _type; }
protected:
	void draw();
private:
	Fl_Pixmap *_pixmap;
	const unsigned long _type;
};

#endif
