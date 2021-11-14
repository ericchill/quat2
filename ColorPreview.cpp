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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <cstdio>
#include <cstring>

#include <FL/Fl.H>
#include <FL/fl_draw.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Output.H>

#include "ColorPreview.h"
#include "colors.h"

ColorPreview::ColorPreview(int x, int y, int w, int h, const char *label)
	: Fl_Valuator(x, y, w, h, label),
	_defined(true),
	_state(0),
	_idx(0)
{
	_pixmap = new unsigned char[3 * (w - 2 * h) * h];
	_pal.reset();
}

ColorPreview::~ColorPreview() {
	delete[] _pixmap;
}

void ColorPreview::draw() {
	if (!_defined) {
		draw_box();
	}
	int b = static_cast<int>(0.2 * static_cast<double>(h())) + 2;
	int odd = 0;
	if (h() % 2 == 0) {
		odd++;
	}
	Fl_Boxtype bt = FL_UP_BOX;
	if (_state & CP_LEFT) {
		bt = FL_DOWN_BOX;
	}
	fl_draw_box(bt, x(), y(), h(), h(), color());
	bt = FL_UP_BOX;
	if (_state & CP_RIGHT) {
		bt = FL_DOWN_BOX;
	}
	fl_draw_box(bt, x() + w() - h(), y(), h(), h(), color());
	fl_color(labelcolor());
	fl_polygon(x() + b, y() + h() / 2, x() + h() -b , y() + b + odd, x() + h() - b, y() + h() - b);
	fl_polygon(x() + w() - b, y() + h() / 2, x() + b + w() - h(), y() + b + odd, x() + b + w() - h(), y() + h() - b);
	fl_color(FL_BLACK);
	if (_defined) {
		fl_draw_image(_pixmap, x() + h(), y(), w() - 2 * h(), h());
		fl_color(FL_BLACK);
		fl_rect(x() + h() + _pos[_idx], y(), _pos[_idx+1] - _pos[_idx], h());
		fl_color(FL_WHITE);
		fl_rect(x() + h() + _pos[_idx] + 1, y() + 1, _pos[_idx+1] - _pos[_idx] - 2, h() - 2);
	} else {
		static char tmp[100];
		sprintf_s(tmp, sizeof(tmp), "%s%zi", "A weight is zero! Current color range: ", _idx+1);
		fl_draw(tmp, x() + h() + 5, y() + 14);
	}
	return;
}

int ColorPreview::handle(int event)
{
    size_t oldidx = _idx;
    int oldstate = _state;
    switch (event) {
	case FL_ENTER:
	case FL_LEAVE:
	    return 1;
	case (FL_PUSH):
	    if (FL_LEFT_MOUSE == Fl::event_button()) {
		    int mx = Fl::event_x();
			int my = Fl::event_y();
		    if (x() < mx && mx < x() + h() && y() < my && my < y() + h()) {
			    _state |= CP_LEFT;
				if (_idx > 0) {
					_idx--;
				}
			}
		    else _state &= ~CP_LEFT;
		    if (x() + w() - h() < mx && mx < x() + w() && y() < my && my < y() + h()) {
			    _state |= CP_RIGHT;
				if (_idx < _pal._nColors - 1) {
					_idx++;
				}
			} else {
				_state &= ~CP_RIGHT;
			}
		    if (_defined && x() + h() < mx && mx <x () + w() - h() && y() < my && my < y() + h()) {
			    for (_idx =0 ; _idx  <_pal._nColors && _pos[_idx+1] < mx - x() - h(); _idx++);
			    redraw();
			}
		}
		if (oldidx != _idx) {
			UpdateInputs();
		}
		if (oldstate != _state) {
			redraw();
		}
	    return 1;
	case (FL_RELEASE):
	    if (FL_LEFT_MOUSE == Fl::event_button()) {
		    _state &= ~(CP_LEFT | CP_RIGHT);
		}
		if (oldstate != _state) {
			redraw();
		}
	    return 1;
	default:
		return Fl_Valuator::handle(event);
	}
}

void ColorPreview::add()
{
	if (_pal._nColors >= RealPalette::maxColors) {
		return;
	}
    _pal._nColors++;
	if (_pal._nColors == RealPalette::maxColors) {
		_button_add->deactivate();
	} else {
		_button_add->activate();
	}
    _button_del->activate();
	for (size_t i = _pal._nColors; i > _idx; i--) {
		_pal._cols[i] = _pal._cols[i - 1];
	}
	if (_idx > 0) {
		for (int i = 0; i < 3; ++i) {
			//		_pal.cols[_idx].col2 = _pal.cols[_idx+1].col1;
			_pal._cols[_idx].col2[i] = _pal._cols[_idx + 1].col1[i];
			_pal._cols[_idx].col1[i] = _pal._cols[_idx - 1].col2[i];
		}
	} else {
		for (int i = 0; i < 3; ++i) {
			_pal._cols[0].col1[i] = _pal._cols[0].col2[i];
			_pal._cols[0].col2[i] = _pal._cols[1].col1[i];
		}
	}
    UpdateInputs();
    Update();
}

void ColorPreview::del() {
	if (_pal._nColors <= 1) {
		return;
	}
    _pal._nColors--;
	if (_pal._nColors == 1) {
		_button_del->deactivate();
	}
    _button_add->activate();
	if (_idx > _pal._nColors - 1) {
		_idx = _pal._nColors - 1;
	} else {
		for (size_t i = _idx; i < _pal._nColors; i++) {
			_pal._cols[i] = _pal._cols[i + 1];
		}
	}
    UpdateInputs();
    Update();
}

void ColorPreview::Update() {
    float BBuf = 0.0F, CBuf = 0.0F;
    unsigned char col[3] = { 0, 0, 0};

	double weightsum = _pal.computeWeightSum();
    _pos.clear(); 
	_pos.push_back(0);
	if (0 == weightsum) {
		_defined = false;
	} else {
	    _defined = true;
		BBuf = 1.0f;
	    for (int i = 0; i < w() - 2 * h(); i++) {
		    CBuf = static_cast<float>(i)/static_cast<float>(w()-2*h());
		    _pal.pixelValue(0, 0, 255, 255, 255, &col[0], &CBuf, &BBuf);
		    for (int j = 0; j < h(); j++) {
				int idx = 3 * (i + j * (w() - 2 * h()));
			    _pixmap[idx] = col[0];
			    _pixmap[idx+1] = col[1];
			    _pixmap[idx+2] = col[2];
			}
		}
	    double offs = 0.0;
	    for (int i = 0; i < _pal._nColors; i++) {
		    offs += _pal._cols[i].weight;
		    _pos.push_back(static_cast<int>(offs / weightsum * static_cast<double>(w() - 2 * h()) + 0.5));
		}
	}
    redraw();
}

void ColorPreview::UpdateInputs() {
    _input_weight->value(_pal._cols[_idx].weight);
    _input_red1->value(_pal._cols[_idx].col1[0]);
    _input_green1->value(_pal._cols[_idx].col1[1]);
    _input_blue1->value(_pal._cols[_idx].col1[2]);
    _input_red2->value(_pal._cols[_idx].col2[0]);
    _input_green2->value(_pal._cols[_idx].col2[1]);
    _input_blue2->value(_pal._cols[_idx].col2[2]);
    char buffer[1024];
    sprintf_s(buffer, sizeof(buffer), "%s%zi/%s%zi", (_idx + 1 < 10) ? " " : "", _idx + 1, 
	    (_pal._nColors < 10) ? " " : "", _pal._nColors);
    _output_range->value(buffer);
}

