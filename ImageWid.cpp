/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */
/* */
/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */
/* */
/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */

#include "ImageWid.h"

#include "CReplacements.h" // assert
#include <new>


ImageWid::ImageWid(int x, int y, int w, int h, const char *label)
	: Fl_Widget(x, y, w, h, label), _data(0)
{
	_data = new uint8_t[w*h*3];
	_lineDrawn = new bool[h];
	fillArray<bool>(_lineDrawn, false, h);
}

ImageWid::~ImageWid() {
	delete _data;
	delete _lineDrawn;
}
#if 0
bool ImageWid::newImage(int w, int h) {
	operator delete[] (_data, std::nothrow);
	_data = new uint8_t[w*h*3];
	memset(_data, 0, w*h*3);
	Fl_Widget::size(w, h);
	return true;
}
#endif
void ImageWid::gray(int level) {
	memset(_data, level, w() * h() * 3);
	redraw();
}

void ImageWid::white() {
	memset(_data, 255, w()*h()*3);
	redraw();
}

void ImageWid::set_pixel(int _x, int _y, uint8_t r, uint8_t g, uint8_t b) {
	int pixIdx = 3 * (w() * _y + _x);
	_data[pixIdx]   = r;
	_data[pixIdx+1] = g;
	_data[pixIdx+2] = b;
	damage(FL_DAMAGE_CHILD, x()+_x, y()+_y, 1, 1);
}

int randomFoo = 0;
void ImageWid::set_line_segment(int x1, int x2, int _y, uint8_t *d) {
	if (_lineDrawn[_y]) {
		randomFoo = 1;
	}
	_lineDrawn[_y] = true;
	memcpy(_data+3*(w()*_y+x1), d, 3*(x2-x1+1));
	damage(FL_DAMAGE_CHILD, x()+x1, y()+_y, x2-x1+1, 1);
}

void ImageWid::set_line(int n, uint8_t *d) {
	memcpy(_data+3*w()*n, d, 3*w());
	damage(FL_DAMAGE_CHILD, x(), y()+n, w(), 1);
}

void ImageWid::get_line(int n, uint8_t *d) const {
	memcpy(d, _data+3*w()*n, 3*w());
}

void ImageWid::draw() {
	fl_draw_image(_data, x(), y(), w(), h());
}

int ImageWid::handle(int event) {
	switch (event) {
		case FL_ENTER:
		case FL_LEAVE:
			return 1;
		default:
			return Fl_Widget::handle(event);
	}
}
