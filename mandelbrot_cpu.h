///////////////////////////////////////////////////////////////////////////////
// File:		  mandelbrot_cpu.h
// Revision:	  1
// Date Creation: 28.11.2018
// Last Change:	  28.11.2018
// Author:		  Christian Steinbrecher
// Descrition:	  For calculation of a mandelbrot on the cpu
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "mandelbrot.h"

/// <summary>
/// Calculates a picture of a mandelbrot
/// </summary>
/// <param name="bmp">bitmap with an allocated picture</param>
/// <param name="z">z variable of the fractual carpet</param>
/// <param name="c">c variable of the fractual carpet</param>
/// <param name="center">center of the complex plane</param>
/// <param name="unit">zomm factor in the complex plane </param>
void mandelbrot_cpu(Vertex hp_vertexBuffer[],
	size_t const height, size_t const width, 
	complex_t const & z, complex_t const & c, double const unit);


