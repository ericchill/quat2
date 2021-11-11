// generated by Fast Light User Interface Designer (fluid) version 1.0108

#ifndef ColorEditor_h
#define ColorEditor_h
#include <FL/Fl.H>
#include <FL/Fl_Group.H>
class ColorPreview;
class ColorClipboard;
class ChildWindow;
#include <FL/Fl_Group.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Input.H>

class ColorScheme;
class RealPalette;

class ColorEditor : public Fl_Group {
public:
  ColorEditor(int X, int Y, int W, int H, const char *label) ;
  ChildWindow *win;
  ColorPreview *CP;
  ColorClipboard *clip;
  Fl_Output *range;
  Fl_Button *add;
private:
  void cb_add_i(Fl_Button*, void*);
  static void cb_add(Fl_Button*, void*);
public:
  Fl_Button *del;
private:
  void cb_del_i(Fl_Button*, void*);
  static void cb_del(Fl_Button*, void*);
public:
  Fl_Value_Input *weight;
private:
  void cb_weight_i(Fl_Value_Input*, void*);
  static void cb_weight(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *red1;
private:
  void cb_red1_i(Fl_Value_Input*, void*);
  static void cb_red1(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *green1;
private:
  void cb_green1_i(Fl_Value_Input*, void*);
  static void cb_green1(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *blue1;
private:
  void cb_blue1_i(Fl_Value_Input*, void*);
  static void cb_blue1(Fl_Value_Input*, void*);
public:
  Fl_Button *button_sel1;
private:
  void cb_button_sel1_i(Fl_Button*, void*);
  static void cb_button_sel1(Fl_Button*, void*);
public:
  Fl_Button *button_copy1;
private:
  void cb_button_copy1_i(Fl_Button*, void*);
  static void cb_button_copy1(Fl_Button*, void*);
public:
  Fl_Button *button_paste1;
private:
  void cb_button_paste1_i(Fl_Button*, void*);
  static void cb_button_paste1(Fl_Button*, void*);
public:
  Fl_Value_Input *red2;
private:
  void cb_red2_i(Fl_Value_Input*, void*);
  static void cb_red2(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *green2;
private:
  void cb_green2_i(Fl_Value_Input*, void*);
  static void cb_green2(Fl_Value_Input*, void*);
public:
  Fl_Value_Input *blue2;
private:
  void cb_blue2_i(Fl_Value_Input*, void*);
  static void cb_blue2(Fl_Value_Input*, void*);
public:
  Fl_Button *button_sel2;
private:
  void cb_button_sel2_i(Fl_Button*, void*);
  static void cb_button_sel2(Fl_Button*, void*);
public:
  Fl_Button *button_copy2;
private:
  void cb_button_copy2_i(Fl_Button*, void*);
  static void cb_button_copy2(Fl_Button*, void*);
public:
  Fl_Button *button_paste2;
private:
  void cb_button_paste2_i(Fl_Button*, void*);
  static void cb_button_paste2(Fl_Button*, void*);
public:
  Fl_Choice *choice_formula;
  static Fl_Menu_Item menu_choice_formula[];
private:
  void cb_0_i(Fl_Menu_*, void*);
  static void cb_0(Fl_Menu_*, void*);
  void cb_01_i(Fl_Menu_*, void*);
  static void cb_01(Fl_Menu_*, void*);
  void cb_02_i(Fl_Menu_*, void*);
  static void cb_02(Fl_Menu_*, void*);
  void cb_03_i(Fl_Menu_*, void*);
  static void cb_03(Fl_Menu_*, void*);
  void cb_04_i(Fl_Menu_*, void*);
  static void cb_04(Fl_Menu_*, void*);
  void cb_x_i(Fl_Menu_*, void*);
  static void cb_x(Fl_Menu_*, void*);
  void cb_sin_i(Fl_Menu_*, void*);
  static void cb_sin(Fl_Menu_*, void*);
  void cb_y_i(Fl_Menu_*, void*);
  static void cb_y(Fl_Menu_*, void*);
  void cb_05_i(Fl_Menu_*, void*);
  static void cb_05(Fl_Menu_*, void*);
  void cb_sqrt_i(Fl_Menu_*, void*);
  static void cb_sqrt(Fl_Menu_*, void*);
public:
  Fl_Input *ColorFormula;
  void set(const RealPalette& p, const ColorScheme& scheme);
  void get(RealPalette& p, ColorScheme& scheme);
  void deactiv();
};
#endif