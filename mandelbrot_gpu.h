///////////////////////////////////////////////////////////////////////////////
// File:		  mandelbrot.h
// Revision:	  1
// Date Creation: 29.11.2018
// Last Change:	  29.11.2018
// Author:		  Christian Steinbrecher
// Descrition:	  For calculation of a mandelbrot on the gpu with cuda
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "mandelbrot.h"





// do not use const memory
void mandelbrot_gpu(Vertex dp_vertexBuffer[],
	size_t const height, size_t const width,
	complex_t const& z, complex_t const& c, double const unit);







