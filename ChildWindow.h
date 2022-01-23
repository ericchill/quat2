#ifndef CHILDWINDOW_H
#define CHILDWINDOW_H

#pragma warning(push, 0)
#include <FL/Fl_Window.H>
#pragma warning(pop)

class ChildWindow : public Fl_Window
{
public:
	ChildWindow(int x, int y, const char *label=0) : Fl_Window(0, 0, x, y, label) {}
};

#endif

