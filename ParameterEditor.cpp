// generated by Fast Light User Interface Designer (fluid) version 1.0304

#include "ParameterEditor.h"
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
#include "MainWindow.h"
#include "ObjectEditor.h"
#include "ViewEditor.h"
#include "ColorEditor.h"
#include "IntersecEditor.h"
#include "OtherEditor.h"
#include "JuliaPreview.h"
#include "ViewSelector.h"
#include "quat.h"
#include <FL/fl_ask.H>
#include "dragWindow.h"
#include <cmath>    // sqrt

void ParameterEditor::cb_ok_button_i(Fl_Return_Button*, void*) {
  dialog->hide();
window_preview->hide();
_result = 1;
}
void ParameterEditor::cb_ok_button(Fl_Return_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_ok_button_i(o,v);
}

void ParameterEditor::cb_cancel_button_i(Fl_Button*, void*) {
  dialog->hide();
window_preview->hide();
_result = 0;
}
void ParameterEditor::cb_cancel_button(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_cancel_button_i(o,v);
}

void ParameterEditor::cb_button_reset_i(Fl_Button*, void*) {
  if (fl_ask("This will reset all parameters to their default values. Continue?")) {
    FractalPreferences f;
    set(f);
};
}
void ParameterEditor::cb_button_reset(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_button_reset_i(o,v);
}

void ParameterEditor::cb_button_import_i(Fl_Button*, void*) {
  FractalPreferences fractal;
get(fractal);
if (MainWindow::mainWindowPtr->Parameters_ReadPNG(fractal, _state==2)) {
    set(fractal);
};
}
void ParameterEditor::cb_button_import(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_button_import_i(o,v);
}

void ParameterEditor::cb_button_read_i(Fl_Button*, void*) {
  FractalPreferences fractal;
get(fractal);
if (MainWindow::mainWindowPtr->Parameters_ReadINI(fractal)) {
    set(fractal);
};
}
void ParameterEditor::cb_button_read(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_button_read_i(o,v);
}

void ParameterEditor::cb_Write_i(Fl_Button*, void*) {
  FractalPreferences fractal;
get(fractal);
MainWindow::mainWindowPtr->Parameters_SaveAs(fractal);
}
void ParameterEditor::cb_Write(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_Write_i(o,v);
}

void ParameterEditor::cb_button_show_i(Fl_Light_Button* o, void*) {
  if (o->value()) {
	window_preview->show();
} else {
	window_preview->hide();
	dialog->show();  // raise (Windows bug)
};
}
void ParameterEditor::cb_button_show(Fl_Light_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_button_show_i(o,v);
}

void ParameterEditor::cb_DoPreview_i(Fl_Button*, void*) {
  if (!window_preview->shown()) {
    button_show->value(1);
    window_preview->show();
}
calcPreview();
Preview->redraw();
}
void ParameterEditor::cb_DoPreview(Fl_Button* o, void* v) {
  ((ParameterEditor*)(o->parent()->user_data()))->cb_DoPreview_i(o,v);
}

ParameterEditor::ParameterEditor() : _state(0), _result(0) {
  { dialog = new Fl_Double_Window(585, 355, "Parameter Editor");
    dialog->user_data((void*)(this));
    { tabs = new Fl_Tabs(10, 0, 420, 240);
      { Object = new ObjectEditor(10, 20, 420, 220, "Object");
        Object->box(FL_NO_BOX);
        Object->color(FL_BACKGROUND_COLOR);
        Object->selection_color(FL_BACKGROUND_COLOR);
        Object->labeltype(FL_NORMAL_LABEL);
        Object->labelfont(0);
        Object->labelsize(14);
        Object->labelcolor(FL_FOREGROUND_COLOR);
        Object->align(Fl_Align(FL_ALIGN_CENTER));
        Object->when(FL_WHEN_RELEASE);
        Object->hide();
      } // ObjectEditor* Object
      { View = new ViewEditor(10, 20, 420, 220, "View");
        View->box(FL_NO_BOX);
        View->color(FL_BACKGROUND_COLOR);
        View->selection_color(FL_BACKGROUND_COLOR);
        View->labeltype(FL_NORMAL_LABEL);
        View->labelfont(0);
        View->labelsize(14);
        View->labelcolor(FL_FOREGROUND_COLOR);
        View->align(Fl_Align(FL_ALIGN_CENTER));
        View->when(FL_WHEN_RELEASE);
        View->hide();
      } // ViewEditor* View
      { Color = new ColorEditor(10, 20, 420, 220, "Color");
        Color->box(FL_NO_BOX);
        Color->color(FL_BACKGROUND_COLOR);
        Color->selection_color(FL_BACKGROUND_COLOR);
        Color->labeltype(FL_NORMAL_LABEL);
        Color->labelfont(0);
        Color->labelsize(14);
        Color->labelcolor(FL_FOREGROUND_COLOR);
        Color->align(Fl_Align(FL_ALIGN_CENTER));
        Color->when(FL_WHEN_RELEASE);
        Color->hide();
      } // ColorEditor* Color
      { Intersection = new IntersecEditor(10, 20, 420, 220, "Intersection");
        Intersection->box(FL_NO_BOX);
        Intersection->color(FL_BACKGROUND_COLOR);
        Intersection->selection_color(FL_BACKGROUND_COLOR);
        Intersection->labeltype(FL_NORMAL_LABEL);
        Intersection->labelfont(0);
        Intersection->labelsize(14);
        Intersection->labelcolor(FL_FOREGROUND_COLOR);
        Intersection->align(Fl_Align(FL_ALIGN_CENTER));
        Intersection->when(FL_WHEN_RELEASE);
        Intersection->hide();
      } // IntersecEditor* Intersection
      { Other = new OtherEditor(10, 20, 420, 220, "Other");
        Other->box(FL_NO_BOX);
        Other->color(FL_BACKGROUND_COLOR);
        Other->selection_color(FL_BACKGROUND_COLOR);
        Other->labeltype(FL_NORMAL_LABEL);
        Other->labelfont(0);
        Other->labelsize(14);
        Other->labelcolor(FL_FOREGROUND_COLOR);
        Other->align(Fl_Align(FL_ALIGN_CENTER));
        Other->when(FL_WHEN_RELEASE);
      } // OtherEditor* Other
      tabs->end();
    } // Fl_Tabs* tabs
    { info = new Fl_Box(450, 20, 120, 50, "For information only.\nTo edit values, please\nclose image first.");
      info->labelsize(12);
      info->labelcolor((Fl_Color)1);
      info->align(Fl_Align(FL_ALIGN_TOP_LEFT|FL_ALIGN_INSIDE));
      info->hide();
    } // Fl_Box* info
    { ok_button = new Fl_Return_Button(450, 20, 130, 30, "OK");
      ok_button->shortcut(0xff0d);
      ok_button->labeltype(FL_ENGRAVED_LABEL);
      ok_button->labelsize(12);
      ok_button->callback((Fl_Callback*)cb_ok_button);
    } // Fl_Return_Button* ok_button
    { cancel_button = new Fl_Button(450, 70, 130, 30, "Cancel");
      cancel_button->shortcut(0xff1b);
      cancel_button->labeltype(FL_ENGRAVED_LABEL);
      cancel_button->labelsize(12);
      cancel_button->callback((Fl_Callback*)cb_cancel_button);
    } // Fl_Button* cancel_button
    { button_reset = new Fl_Button(480, 120, 50, 20, "Reset");
      button_reset->tooltip("Reset all parameters in the editor.");
      button_reset->labelsize(12);
      button_reset->callback((Fl_Callback*)cb_button_reset);
      button_reset->window()->hotspot(button_reset);
    } // Fl_Button* button_reset
    { button_import = new Fl_Button(530, 120, 50, 20, "Import");
      button_import->tooltip("Read parameters from a PNG into the editor.");
      button_import->labelsize(12);
      button_import->callback((Fl_Callback*)cb_button_import);
    } // Fl_Button* button_import
    { button_read = new Fl_Button(480, 140, 50, 20, "Read");
      button_read->tooltip("Read parameters from an INI file into editor.");
      button_read->labelsize(12);
      button_read->callback((Fl_Callback*)cb_button_read);
    } // Fl_Button* button_read
    { Fl_Button* o = new Fl_Button(530, 140, 50, 20, "Write");
      o->tooltip("Write current parameters to INI file.");
      o->labelsize(12);
      o->callback((Fl_Callback*)cb_Write);
    } // Fl_Button* o
    { Fl_Box* o = new Fl_Box(10, 250, 350, 100, "Preview");
      o->tooltip("Preview section.");
      o->box(FL_ENGRAVED_BOX);
      o->align(Fl_Align(FL_ALIGN_TOP_LEFT|FL_ALIGN_INSIDE));
    } // Fl_Box* o
    { own = new Fl_Light_Button(20, 290, 50, 20, "&own");
      own->tooltip("On: Use the current view and color parameters for the preview.\nOff: Use defa\
ults in case you\'re lost.");
      own->shortcut(0x8006f);
      own->value(1);
      own->labelsize(12);
    } // Fl_Light_Button* own
    { stereo = new Fl_Light_Button(80, 290, 60, 20, "S&tereo");
      stereo->tooltip("Enable for a stereoscopic preview.");
      stereo->shortcut(0x80074);
      stereo->labelsize(12);
      stereo->deactivate();
    } // Fl_Light_Button* stereo
    { button_show = new Fl_Light_Button(20, 320, 50, 20, "&show");
      button_show->tooltip("Show / Hide Julia Preview Window.");
      button_show->shortcut(0x80073);
      button_show->labelsize(12);
      button_show->callback((Fl_Callback*)cb_button_show);
    } // Fl_Light_Button* button_show
    { DoPreview = new Fl_Button(80, 320, 60, 20, "&Calc");
      DoPreview->tooltip("Calculate a preview from the current parameters.");
      DoPreview->shortcut(0x80063);
      DoPreview->labelsize(12);
      DoPreview->callback((Fl_Callback*)cb_DoPreview);
    } // Fl_Button* DoPreview
    { preview_size = new Fl_Value_Slider(160, 320, 170, 20, "Preview size");
      preview_size->type(1);
      preview_size->labelsize(12);
      preview_size->minimum(4800);
      preview_size->maximum(76800);
      preview_size->step(10);
      preview_size->value(4800);
      preview_size->align(Fl_Align(FL_ALIGN_TOP_LEFT));
    } // Fl_Value_Slider* preview_size
    { Fl_Box* o = new Fl_Box(370, 260, 70, 20, "from beside");
      o->labelsize(12);
      o->align(Fl_Align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE));
    } // Fl_Box* o
    { vsbeside = new ViewSelector(370, 280, 100, 70, "from beside");
      vsbeside->tooltip("Drag to move the view point.");
      vsbeside->box(FL_EMBOSSED_FRAME);
      vsbeside->color(FL_BACKGROUND_COLOR);
      vsbeside->selection_color(FL_BACKGROUND_COLOR);
      vsbeside->labeltype(FL_NO_LABEL);
      vsbeside->labelfont(0);
      vsbeside->labelsize(12);
      vsbeside->labelcolor(FL_FOREGROUND_COLOR);
      vsbeside->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      vsbeside->when(FL_WHEN_RELEASE);
    } // ViewSelector* vsbeside
    { Fl_Box* o = new Fl_Box(480, 260, 60, 20, "from front");
      o->labelsize(12);
      o->align(Fl_Align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE));
    } // Fl_Box* o
    { vsfront = new ViewSelector(480, 280, 100, 70, "from front");
      vsfront->tooltip("Drag to move the view point.");
      vsfront->box(FL_EMBOSSED_FRAME);
      vsfront->color(FL_BACKGROUND_COLOR);
      vsfront->selection_color(FL_BACKGROUND_COLOR);
      vsfront->labeltype(FL_NO_LABEL);
      vsfront->labelfont(0);
      vsfront->labelsize(12);
      vsfront->labelcolor(FL_FOREGROUND_COLOR);
      vsfront->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      vsfront->when(FL_WHEN_RELEASE);
    } // ViewSelector* vsfront
    { Fl_Box* o = new Fl_Box(480, 170, 70, 20, "from above");
      o->labelsize(12);
      o->align(Fl_Align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE));
    } // Fl_Box* o
    { vsabove = new ViewSelector(480, 190, 100, 70, "from above");
      vsabove->tooltip("Drag to move the view point.");
      vsabove->box(FL_EMBOSSED_FRAME);
      vsabove->color(FL_BACKGROUND_COLOR);
      vsabove->selection_color(FL_BACKGROUND_COLOR);
      vsabove->labeltype(FL_NO_LABEL);
      vsabove->labelfont(0);
      vsabove->labelsize(12);
      vsabove->labelcolor(FL_FOREGROUND_COLOR);
      vsabove->align(Fl_Align(FL_ALIGN_TOP_LEFT));
      vsabove->when(FL_WHEN_RELEASE);
    } // ViewSelector* vsabove
    dialog->end();
  } // Fl_Double_Window* dialog
  { window_preview = new dragWindow(200, 80, "Julia Preview Window");
    window_preview->box(FL_EMBOSSED_BOX);
    window_preview->color(FL_BACKGROUND_COLOR);
    window_preview->selection_color(FL_BACKGROUND_COLOR);
    window_preview->labeltype(FL_NO_LABEL);
    window_preview->labelfont(0);
    window_preview->labelsize(14);
    window_preview->labelcolor(FL_FOREGROUND_COLOR);
    window_preview->user_data((void*)(this));
    window_preview->align(Fl_Align(FL_ALIGN_TOP));
    window_preview->when(FL_WHEN_RELEASE);
    { Preview = new JuliaPreview(2, 2, 200, 80, "label");
      Preview->tooltip("Julia set preview window.");
      Preview->box(FL_FLAT_BOX);
      Preview->color(FL_BACKGROUND_COLOR);
      Preview->selection_color(FL_BACKGROUND_COLOR);
      Preview->labeltype(FL_NO_LABEL);
      Preview->labelfont(0);
      Preview->labelsize(12);
      Preview->labelcolor(FL_FOREGROUND_COLOR);
      Preview->align(Fl_Align(FL_ALIGN_CENTER));
      Preview->when(FL_WHEN_RELEASE);
    } // JuliaPreview* Preview
    window_preview->set_non_modal();
    window_preview->clear_border();
    window_preview->end();
  } // dragWindow* window_preview
  vsabove->SetInputs(View->vre, View->vj); 
  vsfront->SetInputs(View->vre, View->vi); 
  vsbeside->SetInputs(View->vj, View->vi); 
  vsabove->Init(0, 2, 1, -1);
  vsbeside->Init(2, 1, -1, 1);
  vsfront->Init(0, 1, 1, 1);
  View->setSelectors(vsabove, vsbeside, vsfront, stereo);
  View->setParentEditor(this);
  Other->setSelectors(vsabove, vsbeside, vsfront);
  Preview->setbutton3d(DoPreview);
  window_preview->resize(dialog->x()+150, 
  	dialog->y()+250, window_preview->w(),
  	window_preview->h());
}

int ParameterEditor::run() {
  tabs->value(Object);
  dialog->show();
  while (dialog->shown() && MainWindow::mainWindowPtr->shown()) {
      Fl::wait();
      for (;;) {
          Fl_Widget *o = Fl::readqueue();
          if (!o) break;
      }
  }
  if (dialog->shown()) {
      dialog->hide();
  }
  return _result;
}

void ParameterEditor::set(const FractalPreferences& fractal) {
  Object->set(fractal.fractal());
  View->set(fractal.view());
  Color->set(fractal.realPalette(), fractal.colorScheme());
  Intersection->set(fractal.cuts(), fractal.view());
  Other->set(fractal.view());
}

void ParameterEditor::get(FractalPreferences& fractal) {
  Object->get(fractal.fractal());
  View->get(fractal.view());
  Color->get(fractal.realPalette(), fractal.colorScheme());
  Intersection->get(fractal.cuts());
  Other->get(fractal.view());
}

void ParameterEditor::SetState(int state) {
  if (_state == state) {
      return;
  }
  _state = state;
  if (state==1) {
  	vsbeside->deactivate();
  	vsabove->deactivate();
  	vsfront->deactivate();
  	button_reset->deactivate();
  	button_read->deactivate();
  	button_import->deactivate();
  //	tabs->deactivate();
  	Object->deactiv();
  	View->deactivate();
  	Color->deactiv();
  	Intersection->deactiv();
  	Other->deactivate();
  //	deactiv();
  //	Object->deactiv->deactivate();
  //	View->deactivate();
  //	Color->deactiv();
  //	Intersection->deactiv->deactivate();
  //	Other->deactivate();
  	ok_button->hide();
  	info->show();
  }
  if (state==2) {
  	vsbeside->deactivate();
  	vsfront->deactivate();
  	vsabove->deactivate();
  	button_reset->deactivate();
  //	Object->deactiv->deactivate();
  	Object->deactiv();
  	View->group_v->deactivate();
  	View->group_up->deactivate();
  	View->lxr->deactivate();
  	View->group_interocular->deactivate();
  	View->group_move->deactivate();
  //	Intersection->deactiv->deactivate();
  	Intersection->deactiv();
  	Other->group_res->deactivate();
  	Other->antialiasing->deactivate();
  	Other->group_buttons->deactivate();
  }
}

ParameterEditor::~ParameterEditor() {
  delete dialog;
  delete window_preview;
}

void ParameterEditor::calcPreview() {
  FractalPreferences fractal;
  
      bool ster = stereo->value() && stereo->active() && own->value();
  
      get(fractal);
  
      double rx = fractal.view()._xres;
      double ry = fractal.view()._yres;
      if (!own->value()) {
          rx = 80.0;
          ry = 60.0;
      }
      double ratio = rx / ry;
      int x = (int)(sqrt(ratio * preview_size->value()) + 0.5);
      int y = (int)((double)x / ratio + 0.5);
      if (ster) {
          x *= 2;
      }
  
      if (Preview->w() != x || Preview->h() != y) {
          window_preview->remove(Preview);
          delete Preview;
          window_preview->resize(window_preview->x(), window_preview->y(),
              x + Fl::box_dw(window_preview->box()) - 1, y + Fl::box_dh(window_preview->box()) - 1);
          window_preview->begin();
          Preview = new JuliaPreview(Fl::box_dx(window_preview->box()), Fl::box_dy(window_preview->box()), x, y);
          window_preview->end();
      }
  
      Preview->setbutton3d(DoPreview);
      Preview->ownView(own->value());
      Preview->stereo(ster);
      Preview->set(fractal);
      Preview->CalcImage3D();
}

void ParameterEditor::updateView(const FractalView& view) {
  Intersection->setView(view);
}
