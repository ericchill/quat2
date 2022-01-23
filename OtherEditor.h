// generated by Fast Light User Interface Designer (fluid) version 1.0308

#ifndef OtherEditor_h
#define OtherEditor_h
class ChildWindow;
#include "parameters.h"
class ViewSelector;
#pragma warning(push, 0)
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Button.H>
#pragma warning(pop)

class OtherEditor : public Fl_Group {
public:
  OtherEditor(int X, int Y, int W, int H, const char *label) ;
  ChildWindow *win;
  Fl_Group *group_res;
  Fl_Value_Input *xres;
private:
  inline void cb_xres_i(Fl_Value_Input*, void*);
  static void cb_xres(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *yres;
private:
  inline void cb_yres_i(Fl_Value_Input*, void*);
  static void cb_yres(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *zres;
private:
  inline void cb_zres_i(Fl_Value_Input*, void*);
  static void cb_zres(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *phongmax;
private:
  inline void cb_phongmax_i(Fl_Value_Input*, void*);
  static void cb_phongmax(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *phongsharp;
private:
  inline void cb_phongsharp_i(Fl_Value_Input*, void*);
  static void cb_phongsharp(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *ambient;
private:
  inline void cb_ambient_i(Fl_Value_Input*, void*);
  static void cb_ambient(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *antialiasing;
private:
  inline void cb_antialiasing_i(Fl_Value_Input*, void*);
  static void cb_antialiasing(Fl_Value_Input*, void*);
public:
  Fl_Group *group_buttons;
  Fl_Button *res1;
private:
  inline void cb_res1_i(Fl_Button*, void*);
  static void cb_res1(Fl_Button*, void*);
public:
  Fl_Button *res1b;
private:
  inline void cb_res1b_i(Fl_Button*, void*);
  static void cb_res1b(Fl_Button*, void*);
public:
  Fl_Button *res2;
private:
  inline void cb_res2_i(Fl_Button*, void*);
  static void cb_res2(Fl_Button*, void*);
public:
  Fl_Button *res3;
private:
  inline void cb_res3_i(Fl_Button*, void*);
  static void cb_res3(Fl_Button*, void*);
public:
  Fl_Button *res4;
private:
  inline void cb_res4_i(Fl_Button*, void*);
  static void cb_res4(Fl_Button*, void*);
public:
  void setSelectors(ViewSelector *vsa, ViewSelector *vsb, ViewSelector *vsf);
  void set(const FractalView& v);
  void get(FractalView& v);
  void checkValidity();
private:
  FractalView _view; 
  ViewSelector *_vs[3]; 
};
#endif
