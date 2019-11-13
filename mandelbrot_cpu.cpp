///////////////////////////////////////////////////////////////////////////////
// File:		  mandelbrot_cpu.cpp
// Revision:	  1
// Date Creation: 28.11.2018
// Last Change:	  28.11.2018
// Author:		  Christian Steinbrecher
// Descrition:	  For calculation of a fractual carpet on the cpu
///////////////////////////////////////////////////////////////////////////////


#include "mandelbrot_cpu.h"
#include "mandelbrot.h"

void mandelbrot_cpu(Vertex hp_vertexBuffer[], 
	size_t const height, size_t const width, 
	complex_t const& z, complex_t const& c, double const unit)
{
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x)
		{
			auto const res = mandelbrot_iterate_color(z, get_index_value(x, y, width, height, c, unit));
			hp_vertexBuffer[y * width + x].color = {
				static_cast<float>(res.R) / 256.0f,
				static_cast<float>(res.G) / 256.0f,
				static_cast<float>(res.B) / 256.0f 
			};
		}
	}
}



