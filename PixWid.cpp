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


#include "PixWid.h"
#pragma warning(push, 0)
#include <FL/fl_draw.H>
#pragma warning(pop)

constexpr unsigned long PixWid::TYPE = 0x0a0b0d0d;

PixWid::PixWid(int x, int y, int w, int h, const char *label)
	: Fl_Widget(x, y, w, h, label), _pixmap(0), _type(TYPE) {}

PixWid::~PixWid() {
	delete _pixmap;
}

void PixWid::setPixmap(const char * const * data) {
	delete _pixmap;
	_pixmap = new Fl_Pixmap(data);
}

void PixWid::draw() {
	if (_pixmap == 0) return;
	int X, Y, W, H;
	fl_clip_box(x(), y(), w(), h(), X, Y, W, H);
	_pixmap->draw(X, Y, W, H, X-x(), Y-y());
}

