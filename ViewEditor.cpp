// generated by Fast Light User Interface Designer (fluid) version 1.0308

#include "ViewEditor.h"
// Quat - A 3D fractal generation program
// Copyright (C) 1997-2000 Dirk Meyer
// (email: dirk.meyer@studserv.uni-stuttgart.de)
// mail:  Dirk Meyer
//        Marbacher Weg 29
//        D-71334 Waiblingen
//        Germany
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#include "ChildWindow.h"
#include "ViewSelector.h"
#pragma warning(push,0)
#include <FL/math.h>
#pragma warning(pop)
#include "CReplacements.h"
#include "ParameterEditor.h"

void ViewEditor::cb_vre_i(Fl_Value_Input* o, void*) {
  _view._s[0] = o->value();
for (int i=0; i<3; i++) { 
    _vs[i]->viewpointre(o->value());
}
if (radio2->value()) {
    ocular_dist->value(newdist());
     ocular_dist->do_callback();
}
if (radio1->value()) {
    ocular_angle->value(newangle());
}
checkValidity();
}
void ViewEditor::cb_vre(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_vre_i(o,v);
}

void ViewEditor::cb_vi_i(Fl_Value_Input* o, void*) {
  _view._s[1] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->viewpointi(o->value());
}
if (radio2->value()) {
    ocular_dist->value(newdist());
    ocular_dist->do_callback();
}
if (radio1->value()) {
    ocular_angle->value(newangle());
}
checkValidity();
}
void ViewEditor::cb_vi(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_vi_i(o,v);
}

void ViewEditor::cb_vj_i(Fl_Value_Input* o, void*) {
  _view._s[2] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->viewpointj(o->value());
}
if (radio2->value()) {
    ocular_dist->value(newdist());
    ocular_dist->do_callback();
}
if (radio1->value()) {
    ocular_angle->value(newangle());
}
checkValidity();
}
void ViewEditor::cb_vj(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_vj_i(o,v);
}

void ViewEditor::cb_upre_i(Fl_Value_Input* o, void*) {
  _view._up[0] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->upre(o->value());
}
checkValidity();
}
void ViewEditor::cb_upre(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_upre_i(o,v);
}

void ViewEditor::cb_upi_i(Fl_Value_Input* o, void*) {
  _view._up[1] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->upi(o->value());
}
checkValidity();
}
void ViewEditor::cb_upi(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_upi_i(o,v);
}

void ViewEditor::cb_upj_i(Fl_Value_Input* o, void*) {
  _view._up[2] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->upj(o->value());
}
checkValidity();
}
void ViewEditor::cb_upj(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_upj_i(o,v);
}

void ViewEditor::cb_lxr_i(Fl_Value_Input* o, void*) {
  _view._LXR = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->LXR(o->value());
}
checkValidity();
}
void ViewEditor::cb_lxr(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->user_data()))->cb_lxr_i(o,v);
}

void ViewEditor::cb_ocular_dist_i(Fl_Value_Input* o, void*) {
  _view._interocular = static_cast<float>(o->value());
ocular_angle->value(newangle());
checkValidity();
}
void ViewEditor::cb_ocular_dist(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_ocular_dist_i(o,v);
}

void ViewEditor::cb_ocular_angle_i(Fl_Value_Input*, void*) {
  _view._interocular = newdist();
ocular_dist->value(_view._interocular);
checkValidity();
}
void ViewEditor::cb_ocular_angle(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_ocular_angle_i(o,v);
}

void ViewEditor::cb_radio1_i(Fl_Round_Button*, void*) {
  ocular_angle->deactivate();
ocular_dist->activate();
}
void ViewEditor::cb_radio1(Fl_Round_Button* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_radio1_i(o,v);
}

void ViewEditor::cb_radio2_i(Fl_Round_Button*, void*) {
  ocular_dist->deactivate();
ocular_angle->activate();
}
void ViewEditor::cb_radio2(Fl_Round_Button* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_radio2_i(o,v);
}

void ViewEditor::cb_lightx_i(Fl_Value_Input* o, void*) {
  _view._light[0] = o->value();
}
void ViewEditor::cb_lightx(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_lightx_i(o,v);
}

void ViewEditor::cb_lighty_i(Fl_Value_Input* o, void*) {
  _view._light[1] = o->value();
}
void ViewEditor::cb_lighty(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_lighty_i(o,v);
}

void ViewEditor::cb_lightz_i(Fl_Value_Input* o, void*) {
  _view._light[2] = o->value();
}
void ViewEditor::cb_lightz(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_lightz_i(o,v);
}

void ViewEditor::cb_movex_i(Fl_Value_Input* o, void*) {
  _view._Mov[0] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->movex(o->value());
};
}
void ViewEditor::cb_movex(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_movex_i(o,v);
}

void ViewEditor::cb_movey_i(Fl_Value_Input* o, void*) {
  _view._Mov[1] = o->value();
for (int i=0; i<3; i++) {
    _vs[i]->movey(o->value());
};
}
void ViewEditor::cb_movey(Fl_Value_Input* o, void* v) {
  ((ViewEditor*)(o->parent()->parent()->user_data()))->cb_movey_i(o,v);
}

ViewEditor::ViewEditor(int X, int Y, int W, int H, const char *label) : Fl_Group(X,Y,W,H,label), _parentEditor(nullptr) {
  { win = new ChildWindow(415, 215);
    win->box(FL_FLAT_BOX);
    win->color(FL_BACKGROUND_COLOR);
    win->selection_color(FL_BACKGROUND_COLOR);
    win->labeltype(FL_NO_LABEL);
    win->labelfont(0);
    win->labelsize(14);
    win->labelcolor(FL_FOREGROUND_COLOR);
    win->user_data((void*)(this));
    win->align(Fl_Align(FL_ALIGN_TOP));
    win->when(FL_WHEN_RELEASE);
    { Fl_Group* o = new Fl_Group(5, 5, 260, 200, "QSpace coordinates (1, i, j)");
      o->tooltip("All parameters in this box are coordinates in Quaternion space.");
      o->box(FL_ENGRAVED_BOX);
      o->labelsize(12);
      o->align(Fl_Align(FL_ALIGN_TOP_LEFT|FL_ALIGN_INSIDE));
      o->end();
    } // Fl_Group* o
    { group_v = new Fl_Group(15, 30, 110, 75, "v");
      group_v->tooltip("The view point represents the center point of the screen in QSpace.\nYou can \
also use the three View Selectors to set the View Point with the mouse.");
      group_v->labeltype(FL_NO_LABEL);
      { vre = new Fl_Value_Input(15, 45, 110, 20, "View Point");
        vre->tooltip("Real part of View Point.");
        vre->labelsize(12);
        vre->minimum(-1e+20);
        vre->maximum(1e+20);
        vre->textsize(12);
        vre->callback((Fl_Callback*)cb_vre);
        vre->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      } // Fl_Value_Input* vre
      { vi = new Fl_Value_Input(15, 65, 110, 20, "value:");
        vi->tooltip("1st imaginary (i) part of View Point.");
        vi->labeltype(FL_NO_LABEL);
        vi->labelsize(12);
        vi->minimum(-1e+20);
        vi->maximum(1e+20);
        vi->textsize(12);
        vi->callback((Fl_Callback*)cb_vi);
      } // Fl_Value_Input* vi
      { vj = new Fl_Value_Input(15, 85, 110, 20, "value:");
        vj->tooltip("2nd imaginary (j) part of View Point.");
        vj->labeltype(FL_NO_LABEL);
        vj->labelsize(12);
        vj->minimum(-1e+20);
        vj->maximum(1e+20);
        vj->textsize(12);
        vj->callback((Fl_Callback*)cb_vj);
      } // Fl_Value_Input* vj
      group_v->end();
    } // Fl_Group* group_v
    { group_up = new Fl_Group(15, 120, 110, 75, "up");
      group_up->tooltip("Needed for orientation of the screen in QSpace.\nDefines the vertical directi\
on of screen.");
      group_up->labeltype(FL_NO_LABEL);
      { upre = new Fl_Value_Input(15, 135, 110, 20, "Up (Orientation)");
        upre->tooltip("Real part of Up.");
        upre->labelsize(12);
        upre->minimum(-1e+20);
        upre->maximum(1e+20);
        upre->textsize(12);
        upre->callback((Fl_Callback*)cb_upre);
        upre->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      } // Fl_Value_Input* upre
      { upi = new Fl_Value_Input(15, 155, 110, 20, "value:");
        upi->tooltip("1st imaginary (i) part of Up.");
        upi->labeltype(FL_NO_LABEL);
        upi->labelsize(12);
        upi->minimum(-1e+20);
        upi->maximum(1e+20);
        upi->textsize(12);
        upi->callback((Fl_Callback*)cb_upi);
      } // Fl_Value_Input* upi
      { upj = new Fl_Value_Input(15, 175, 110, 20, "value:");
        upj->tooltip("2nd imaginary (j) part of Up.");
        upj->labeltype(FL_NO_LABEL);
        upj->labelsize(12);
        upj->minimum(-1e+20);
        upj->maximum(1e+20);
        upj->textsize(12);
        upj->callback((Fl_Callback*)cb_upj);
      } // Fl_Value_Input* upj
      group_up->end();
    } // Fl_Group* group_up
    { lxr = new Fl_Value_Input(145, 65, 110, 20, "Length of View\nPlane\'s X-Axis");
      lxr->tooltip("Use this to scale the screen (image).");
      lxr->labelsize(12);
      lxr->minimum(-1e+20);
      lxr->maximum(1e+20);
      lxr->textsize(12);
      lxr->callback((Fl_Callback*)cb_lxr);
      lxr->align(Fl_Align(FL_ALIGN_TOP_LEFT));
    } // Fl_Value_Input* lxr
    { group_interocular = new Fl_Group(145, 115, 110, 85, "Interocular");
      group_interocular->labelsize(12);
      group_interocular->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      { ocular_dist = new Fl_Value_Input(165, 135, 90, 20, "Distance");
        ocular_dist->tooltip("Distance (in QSpace) between the two eyes. If nonzero, you\'ll get a stereo i\
mage.");
        ocular_dist->labelsize(12);
        ocular_dist->maximum(1e+20);
        ocular_dist->textsize(12);
        ocular_dist->callback((Fl_Callback*)cb_ocular_dist);
        ocular_dist->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      } // Fl_Value_Input* ocular_dist
      { ocular_angle = new Fl_Value_Input(165, 175, 90, 20, "Angle (\260)");
        ocular_angle->tooltip("Angle between the two eyes. If nonzero, you\'ll get a stereo image.");
        ocular_angle->labelsize(12);
        ocular_angle->maximum(360);
        ocular_angle->textsize(12);
        ocular_angle->callback((Fl_Callback*)cb_ocular_angle);
        ocular_angle->align(Fl_Align(FL_ALIGN_TOP_LEFT));
        ocular_angle->deactivate();
      } // Fl_Value_Input* ocular_angle
      { radio1 = new Fl_Round_Button(145, 130, 20, 30, "button");
        radio1->tooltip("If selected: Interocular distance constant while moving the View Point.");
        radio1->type(102);
        radio1->down_box(FL_DOWN_BOX);
        radio1->value(1);
        radio1->selection_color((Fl_Color)2);
        radio1->labeltype(FL_NO_LABEL);
        radio1->labelsize(12);
        radio1->callback((Fl_Callback*)cb_radio1);
      } // Fl_Round_Button* radio1
      { radio2 = new Fl_Round_Button(145, 170, 20, 30, "button");
        radio2->tooltip("If selected: Angle is constant while moving the View Point.");
        radio2->type(102);
        radio2->down_box(FL_DOWN_BOX);
        radio2->selection_color((Fl_Color)2);
        radio2->labeltype(FL_NO_LABEL);
        radio2->labelsize(12);
        radio2->callback((Fl_Callback*)cb_radio2);
      } // Fl_Round_Button* radio2
      group_interocular->end();
    } // Fl_Group* group_interocular
    { Fl_Group* o = new Fl_Group(275, 5, 130, 200, "Viewplane coord.");
      o->tooltip("All coordinates in this box are measured relative to the view plane.\nX=horiz\
ontal, Y=vertical, Z=perpendicular to screen.");
      o->box(FL_ENGRAVED_BOX);
      o->labelsize(12);
      o->align(Fl_Align(FL_ALIGN_TOP_LEFT|FL_ALIGN_INSIDE));
      o->end();
    } // Fl_Group* o
    { Fl_Group* o = new Fl_Group(285, 30, 110, 75, "light");
      o->tooltip("Measured from the View Point.");
      o->labeltype(FL_NO_LABEL);
      { lightx = new Fl_Value_Input(285, 45, 110, 20, "Light Source");
        lightx->tooltip("X coordinate of Light Source.");
        lightx->labelsize(12);
        lightx->minimum(-1e+20);
        lightx->maximum(1e+20);
        lightx->textsize(12);
        lightx->callback((Fl_Callback*)cb_lightx);
        lightx->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      } // Fl_Value_Input* lightx
      { lighty = new Fl_Value_Input(285, 65, 110, 20, "value:");
        lighty->tooltip("Y coordinate of Light Source.");
        lighty->labeltype(FL_NO_LABEL);
        lighty->labelsize(12);
        lighty->minimum(-1e+20);
        lighty->maximum(1e+20);
        lighty->textsize(12);
        lighty->callback((Fl_Callback*)cb_lighty);
      } // Fl_Value_Input* lighty
      { lightz = new Fl_Value_Input(285, 85, 110, 20, "value:");
        lightz->tooltip("Z coordinate of Light Source.");
        lightz->labeltype(FL_NO_LABEL);
        lightz->labelsize(12);
        lightz->minimum(-1e+20);
        lightz->maximum(1e+20);
        lightz->textsize(12);
        lightz->callback((Fl_Callback*)cb_lightz);
      } // Fl_Value_Input* lightz
      o->end();
    } // Fl_Group* o
    { group_move = new Fl_Group(285, 120, 110, 55, "move");
      group_move->tooltip("Move the view plane parallel to itself.");
      group_move->labeltype(FL_NO_LABEL);
      { movex = new Fl_Value_Input(285, 135, 110, 20, "Move");
        movex->tooltip("Movement in X direction.");
        movex->labelsize(12);
        movex->minimum(-1e+20);
        movex->maximum(1e+20);
        movex->textsize(12);
        movex->callback((Fl_Callback*)cb_movex);
        movex->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      } // Fl_Value_Input* movex
      { movey = new Fl_Value_Input(285, 155, 110, 20, "value:");
        movey->tooltip("Movement in Y direction.");
        movey->labeltype(FL_NO_LABEL);
        movey->labelsize(12);
        movey->minimum(-1e+20);
        movey->maximum(1e+20);
        movey->textsize(12);
        movey->callback((Fl_Callback*)cb_movey);
      } // Fl_Value_Input* movey
      group_move->end();
    } // Fl_Group* group_move
    win->clear_border();
    win->end();
  } // ChildWindow* win
  end(); // VERY IMPORTANT!
  win->position(X+2, Y+2);
  // DON'T delete win in destructor (or elsewhere) 
  // it's automatically deleted by Fl_Group
}

void ViewEditor::setSelectors(ViewSelector *vsa, ViewSelector *vsb, ViewSelector *vsf, Fl_Light_Button *ster) {
  _vs[0] = vsa;
  _vs[1] = vsb;
  _vs[2] = vsf;
  stereo = ster;
}

void ViewEditor::set(const FractalView& v) {
  for (int i=0; i<3; i++) {
      assert(_vs[i] != nullptr);
  }
  assert(stereo != nullptr);
  _view = v;
  vre->value(v._s[0]);
  vi->value(v._s[1]);
  vj->value(v._s[2]);
  upre->value(v._up[0]);
  upi->value(v._up[1]);
  upj->value(v._up[2]);
  lxr->value(v._LXR);
  ocular_dist->value(v._interocular);
  lightx->value(v._light[0]);
  lighty->value(v._light[1]);
  lightz->value(v._light[2]);
  movex->value(v._Mov[0]);
  movey->value(v._Mov[1]);
  
  // Set Angle, set ViewSelectors and checkValidity:
  ocular_dist->do_callback();
  vre->do_callback();
  vi->do_callback();
  vj->do_callback();
  upre->do_callback();
  upi->do_callback();
  upj->do_callback();
  lxr->do_callback();
  movex->do_callback();
  movey->do_callback();
}

void ViewEditor::get(FractalView& v) {
  // Don't do v=view, because there are other members
  // which aren't in the ViewEditor
  v._s = _view._s;
  v._up = _view._up;
  v._LXR = _view._LXR;
  v._interocular = _view._interocular;
  v._light = _view._light;
  v._Mov[0] = _view._Mov[0];
  v._Mov[1] = _view._Mov[1];
}

bool ViewEditor::isParallel(const Vec3& a, const Vec3& b) {
  return a.cross(b).magnitude() == 0;
}

double ViewEditor::newangle() {
  double a = _view._s.magnitude();
      if (a == 0.0 && _view.isStereo()) {
          return 180.0;
      } else if (a == 0.0) {
          return ocular_angle->value();
      } else {
          return atan(_view._interocular / (2.0 * a)) * 360.0 / M_PI;
      }
}

float ViewEditor::newdist() {
  return static_cast<float>(2.0 * _view._s.magnitude() * tan(M_PI * ocular_angle->value() / 360.0));
}

void ViewEditor::checkValidity() {
  const Fl_Color okc = FL_WHITE;
      const Fl_Color ndefc = FL_RED;
      Fl_Color vre_c = okc, vi_c = okc, vj_c = okc,
          upre_c = okc, upi_c = okc, upj_c = okc,
          lxr_c = okc, od_c = okc, oa_c = okc;
  
      if (_view._s.magnitude() == 0.0) {
          vre_c = ndefc; vi_c = ndefc; vj_c = ndefc;
      } else if (isParallel(_view._s, _view._up)) {
          upre_c = ndefc; upi_c = ndefc; upj_c = ndefc;
      }
  
      if (_view._LXR <= 0.0) lxr_c = ndefc;
  
      if (_view._interocular < 0.0) {
          od_c = ndefc; oa_c = ndefc;
      }
  
      if (vre_c != vre->color()) { vre->color(vre_c); vre->redraw(); }
      if (vi_c != vi->color()) { vi->color(vi_c); vi->redraw(); }
      if (vj_c != vj->color()) { vj->color(vj_c); vj->redraw(); }
  
      if (upre_c != upre->color()) { upre->color(upre_c); upre->redraw(); }
      if (upi_c != upi->color()) { upi->color(upi_c); upi->redraw(); }
      if (upj_c != upj->color()) { upj->color(upj_c); upj->redraw(); }
  
      if (lxr_c != lxr->color()) { lxr->color(lxr_c); lxr->redraw(); }
  
      if (od_c != ocular_dist->color()) { ocular_dist->color(od_c); ocular_dist->redraw(); }
      if (oa_c != ocular_angle->color()) { ocular_angle->color(oa_c); ocular_angle->redraw(); }
  
      if (!_view.isStereo() && stereo->active()) stereo->deactivate();
      if (_view.isStereo() && !stereo->active()) stereo->activate();
      
      _parentEditor->updateView(_view);
}

void ViewEditor::setParentEditor(ParameterEditor* parentEditor) {
  _parentEditor = parentEditor;
}
