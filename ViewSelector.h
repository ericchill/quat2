#pragma once

#pragma warning(push, 0)
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Value_Input.H>
#pragma warning(pop)

#include "common.h"

#include "parameters.h"

class ViewSelector : public Fl_Double_Window
{
public:
	ViewSelector(int x, int y, int w, int h, const char *label);
	~ViewSelector();
	void SetInputs(Fl_Value_Input *vx, Fl_Value_Input *vy) {
		_input_vx = vx;
		_input_vy = vy;
	}
	void Init(int s_x, int s_y, int s_dx, int s_dy);
	void viewpointre(double a) { _v._s[0] = a; redraw(); }
	void viewpointi(double a) { _v._s[1] = a; redraw(); }
	void viewpointj(double a) { _v._s[2] = a; redraw(); }
	void upre(double a) { _v._up[0] = a; redraw(); }
	void upi(double a) { _v._up[1] = a; redraw(); }
	void upj(double a) { _v._up[2] = a; redraw(); }
	void lightx(double a) { _v._light[0] = a; redraw(); }
	void lighty(double a) { _v._light[1] = a; redraw(); }
	void lightz(double a) { _v._light[2] = a; redraw(); }
	void LXR(double a) { _v._LXR = a; redraw(); }
	void movex(double a) { _v._Mov[0] = a; redraw(); }
	void movey(double a) { _v._Mov[1] = a; redraw(); }
	void xres(int a) { _v._xres = a; redraw(); }
	void yres(int a) { _v._yres = a; redraw(); }
protected:
	void draw();
	int handle(int event);
private:
	void calculateCoords();
	FractalView _v;
	int _x, _y, _dx, _dy;
	int _vx, _vy;
	Fl_Value_Input *_input_vx, *_input_vy;
	int _state;
};
