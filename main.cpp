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


#include <stdlib.h>
#pragma warning(push, 0)
#include <FL/Fl.H>
#include <FL/Fl_File_Icon.H>
#pragma warning(pop)

#include "MainWindow.h"
#include "kernel.h"

int main(int argc, char **argv) {
	if (!initGPU(argc, argv)) {
		std::cerr << "No GPU availables, sorry." << std::endl;
		exit(1);
	}
	Fl::visual(FL_DOUBLE|FL_INDEX);
	Fl_File_Icon::load_system_icons();
	MainWindow win(argc, argv);
	return Fl::run();
}
