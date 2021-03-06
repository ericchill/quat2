// generated by Fast Light User Interface Designer (fluid) version 1.0308

#ifndef ViewEditor_h
#define ViewEditor_h
#include "parameters.h"
class ChildWindow;
class ViewSelector;
class ParameterEditor;
#pragma warning(push, 0)
#include <FL/Fl.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Round_Button.H>
#pragma warning(pop)

class ViewEditor : public Fl_Group {
public:
  ViewEditor(int X, int Y, int W, int H, const char *label) ;
  ChildWindow *win;
  Fl_Group *group_v;
  Fl_Value_Input *vre;
private:
  inline void cb_vre_i(Fl_Value_Input*, void*);
  static void cb_vre(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *vi;
private:
  inline void cb_vi_i(Fl_Value_Input*, void*);
  static void cb_vi(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *vj;
private:
  inline void cb_vj_i(Fl_Value_Input*, void*);
  static void cb_vj(Fl_Value_Input*, void*);
public:
  Fl_Group *group_up;
  Fl_Value_Input *upre;
private:
  inline void cb_upre_i(Fl_Value_Input*, void*);
  static void cb_upre(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *upi;
private:
  inline void cb_upi_i(Fl_Value_Input*, void*);
  static void cb_upi(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *upj;
private:
  inline void cb_upj_i(Fl_Value_Input*, void*);
  static void cb_upj(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *lxr;
private:
  inline void cb_lxr_i(Fl_Value_Input*, void*);
  static void cb_lxr(Fl_Value_Input*, void*);
public:
  Fl_Group *group_interocular;
  Fl_Value_Input *ocular_dist;
private:
  inline void cb_ocular_dist_i(Fl_Value_Input*, void*);
  static void cb_ocular_dist(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *ocular_angle;
private:
  inline void cb_ocular_angle_i(Fl_Value_Input*, void*);
  static void cb_ocular_angle(Fl_Value_Input*, void*);
public:
  Fl_Round_Button *radio1;
private:
  inline void cb_radio1_i(Fl_Round_Button*, void*);
  static void cb_radio1(Fl_Round_Button*, void*);
public:
  Fl_Round_Button *radio2;
private:
  inline void cb_radio2_i(Fl_Round_Button*, void*);
  static void cb_radio2(Fl_Round_Button*, void*);
public:
  Fl_Value_Input *lightx;
private:
  inline void cb_lightx_i(Fl_Value_Input*, void*);
  static void cb_lightx(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *lighty;
private:
  inline void cb_lighty_i(Fl_Value_Input*, void*);
  static void cb_lighty(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *lightz;
private:
  inline void cb_lightz_i(Fl_Value_Input*, void*);
  static void cb_lightz(Fl_Value_Input*, void*);
public:
  Fl_Group *group_move;
  Fl_Value_Input *movex;
private:
  inline void cb_movex_i(Fl_Value_Input*, void*);
  static void cb_movex(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *movey;
private:
  inline void cb_movey_i(Fl_Value_Input*, void*);
  static void cb_movey(Fl_Value_Input*, void*);
public:
  void setSelectors(ViewSelector *vsa, ViewSelector *vsb, ViewSelector *vsf, Fl_Light_Button *ster);
  void set(const FractalView& v);
  void get(FractalView& v);
private:
  FractalView _view; 
  ViewSelector *_vs[3]; 
  Fl_Light_Button *stereo; 
public:
  bool isParallel(const Vec3& a, const Vec3& b);
  double newangle();
  float newdist();
  void checkValidity();
private:
  ParameterEditor *_parentEditor; 
public:
  void setParentEditor(ParameterEditor* parentEditor);
};
#endif
