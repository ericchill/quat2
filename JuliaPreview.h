#ifndef JULIAPREVIEW_H
#define JULIAPREVIEW_H

#include "ImageWid.h"
#include "parameters.h"
#include "quat.h"

class Fl_Button;

class JuliaPreview : public ImageWid, public Quater
{
public:
	JuliaPreview(int x, int y, int w, int h, const char* label = 0);
	~JuliaPreview();
	void setbutton3d(Fl_Button *b) {
		button3d = b;
	}
	void putLine(long x1, long x2, long, int y, unsigned char *lineData, bool);
	void CalcImage3D();
	void ownView(bool v) { _ownView = v; }
	void stereo(bool v) { _stereo = v; }
	void set(const FractalPreferences& fractal) {
		_fractal = fractal;
	}
private:
	Fl_Button* button3d;
	bool _updated, _stereo, _ownView, _imagestereo;
	bool _pic_ownView;
	FractalPreferences _fractal;
	FractalPreferences _calcFractal;
	FractalPreferences _picFractal;
};

#endif
