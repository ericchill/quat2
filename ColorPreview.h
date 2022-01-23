#ifndef COLORPREVIEW_H
#define COLORPREVIEW_H


#include <vector>


#pragma warning(push, 0)
#include <FL/Fl_Valuator.H>
#include <FL/Fl_Button.H>
#pragma warning(pop)

class Fl_Value_Input;
class Fl_Output;
class Fl_Button;

#include "parameters.h"

class ColorPreview : public Fl_Valuator
{
static constexpr int CP_LEFT = 1;
static constexpr int CP_RIGHT = 2;
public:
	ColorPreview(int x, int y, int w, int h, const char *label);
	~ColorPreview();
	void SetInputs(Fl_Button *add, Fl_Button *del, Fl_Value_Input *weight,
		Fl_Value_Input *red1, Fl_Value_Input *green1, Fl_Value_Input *blue1,
		Fl_Value_Input *red2, Fl_Value_Input *green2, Fl_Value_Input *blue2,
		Fl_Output *out)
	{
		_button_add = add; _button_del = del; _input_weight = weight;
		_input_red1 = red1; _input_green1 = green1; _input_blue1 = blue1;
		_input_red2 = red2; _input_green2 = green2; _input_blue2 = blue2;
		_output_range = out;
		UpdateInputs();
	}
	void SetPal(const RealPalette &pal) {
		_pal = pal;
		_idx = 0;
		Update();
		UpdateInputs(); 
		if (_pal._nColors > 1) {
			_button_del->activate();
		}
	}
	void GetPal(RealPalette &pal) const { pal = _pal; }
	size_t current() const { return _idx; }
	void add();
	void del();
	void weight(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].weight = v; Update(); }
	void red1(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col1[0] = v; Update(); }
	void green1(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col1[1] = v; Update(); }
	void blue1(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col1[2] = v; Update(); }
	void red2(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col2[0] = v; Update(); }
	void green2(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col2[1] = v; Update(); }
	void blue2(size_t pos, double v) { if (pos>_idx) return; _pal._cols[pos].col2[2] = v; Update(); }
protected:
	virtual void draw();
	virtual int handle(int event);
private:
	void Update();
	void UpdateInputs();
	bool _defined;
	int _state;
	size_t _idx;
	unsigned char *_pixmap;
	std::vector<int> _pos;
	RealPalette _pal;
	Fl_Button *_button_add, *_button_del;
	Fl_Value_Input *_input_weight, *_input_red1, *_input_green1, *_input_blue1;
	Fl_Value_Input *_input_red2, *_input_green2, *_input_blue2;
	Fl_Output *_output_range;
};

#endif
