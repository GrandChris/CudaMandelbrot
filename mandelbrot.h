///////////////////////////////////////////////////////////////////////////////
// File:		  mandelbrot.h
// Revision:	  1
// Date Creation: 28.11.2018
// Last Change:	  28.11.2018
// Author:		  Christian Steinbrecher
// Descrition:	  Cacluates a mandelbrot
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined __CUDACC__
#define GPU_ENABLED __device__ __forceinline__
#else
#define GPU_ENABLED inline
#endif

#include <pfc/pfc_complex.h>

#include <FixedPointValue.h>

using Fixed = FixedPointValue<4, 8>;

using complex_t = pfc::complex<Fixed>;




/// <summary>
/// Maximum number of the magnitude of the complex variable, before breaking up the for loop
/// </summary>
constexpr size_t g_iter_max = 4;

/// <summary>
/// Maximum number of iteratios of the for loop
/// </summary>
//constexpr size_t g_iter_limit = 1024;
constexpr size_t g_iter_limit = 256;
//constexpr size_t g_iter_limit = 128;


/// <summary>
/// Type of a mapped color
/// </summary>
struct color_t
{
	color_t() = default;
	GPU_ENABLED constexpr color_t(unsigned char R, unsigned char G, unsigned char B, unsigned char Alpha = 0)
		: B(B), G(G), R(R), Alpha(Alpha) {}

	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char Alpha;
};


/// <summary>
/// Calcualtes the formula of a fractual carpet
/// </summary>
GPU_ENABLED complex_t mandelbrot(complex_t const& z, complex_t const& c);


/// <summary>
/// Iterates with the fromula fo a fractual carpet and looks, if the
/// function converges or diverges
/// </summary>
/// <returns>The number of iterations</returns>
GPU_ENABLED size_t mandelbrot_iterate(complex_t z, complex_t const& c);


/// <summary>
/// Iterates with the fromula fo a fractual carpet and looks, if the
/// function converges or diverges
/// </summary>
/// <returns>A color corresponding to the number of iterations</returns>
GPU_ENABLED color_t mandelbrot_iterate_color(complex_t const& z, complex_t const& c);


/// <summary>
/// Returns a complex value, corresponding to a pixel of an image
/// </summary>
/// <param name="x">x index of the picture</param>
/// <param name="y">y index of the picture</param>
/// <param name="x_max">maximum x value</param>
/// <param name="y_max">maximum y value</param>
/// <param name="center">complex value of the center of the mandelbrot</param>
/// <param name="unit">unit of one pixel in the mandelbrot</param>
/// <returns></returns>
GPU_ENABLED complex_t get_index_value(size_t const x, size_t const y, size_t const x_max, size_t const y_max,
	complex_t const& center, double const unit);






// #######+++++++ Impelementation +++++++#######

GPU_ENABLED complex_t mandelbrot(complex_t const& z, complex_t const& c)
{
	return z * z + c;
}


GPU_ENABLED size_t mandelbrot_iterate(complex_t z, complex_t const& c)
{
	int i = 0;

	do {

		z = mandelbrot(z, c);

	} while (++i < g_iter_limit && norm(z) < g_iter_max);

	return i;
}


//GPU_ENABLED pfc::complex<double> mandelbrot(pfc::complex<double> const& z, pfc::complex<double> const& c)
//{
//	return z * z + c;
//}
//
//
//GPU_ENABLED size_t mandelbrot_iterate(pfc::complex<double> z, pfc::complex<double> const& c)
//{
//	int i = 0;
//
//	do {
//
//		z = mandelbrot(z, c);
//
//	} while (++i < g_iter_limit && norm(z) < g_iter_max);
//
//	return i;
//}


//GPU_ENABLED complex_t get_index_value(size_t const x, size_t const y, 
//	size_t const x_max, size_t const y_max,
//	complex_t const& center, double const unit)
//{
//	// cast zomm to desired data type
//	using elem_t = decltype(center.real);
//	elem_t const unitValue = static_cast<elem_t>(unit);
//
//	auto const centerX2 = center.real;
//	auto const centerY2 = center.imag;
//
//
//	auto const x_val = static_cast<elem_t>(x - x_max * static_cast<double>(0.5)) * unitValue - centerX2;
//	auto const y_val = static_cast<elem_t>(y - y_max * static_cast<double>(0.5)) * unitValue - centerY2;;
//
//	return { x_val,  y_val };
//}


GPU_ENABLED complex_t get_index_value(size_t const x, size_t const y,
	size_t const x_max, size_t const y_max,
	complex_t const& center, double const unit)
{
	// cast zomm to desired data type
	using elem_t = decltype(center.real);

	auto const centerX2 = center.real;
	auto const centerY2 = center.imag;

	double const middleX = x - x_max * static_cast<double>(0.5);
	double const middleY = y - y_max * static_cast<double>(0.5);

	//elem_t const scaleX = unit * middleX;
	//elem_t const scaleY = unit * middleY;

	//auto const x_val = scaleX - centerX2;
	//auto const y_val = scaleY - centerY2;


	double const scaleX = unit * middleX;
	double const scaleY = unit * middleY;

	// does not work correctly
	elem_t const scaleX_elem_t = elem_t(scaleX);
	elem_t const scaleY_elem_t = elem_t(scaleY);
	// <<

	auto const x_val = scaleX_elem_t - centerX2;
	auto const y_val = scaleY_elem_t - centerY2;

	return { x_val,  y_val };
}

GPU_ENABLED color_t map_colors(size_t  index)
{
	index = index * index / 4;

	constexpr unsigned char const max_color = std::numeric_limits<unsigned char>::max();
	size_t const range = max_color + 1;

	unsigned char const color = static_cast<unsigned char>(index);
	unsigned char const inverted_color = max_color - color;

	// white
	if (index < range * 1)
	{	// yellow
		return color_t(max_color, max_color, inverted_color);
	}
	else if (index < range * 2)
	{	// red
		return color_t(max_color, inverted_color, 0);
	}
	else if (index < range * 3)
	{	// pink
		return color_t(max_color, 0, color);
	}
	else if (index < range * 4)
	{	// light blue
		return color_t(inverted_color, color, max_color);
	}
	else if (index < range * 5)
	{	// blue
		return color_t(0, inverted_color, max_color);
	}
	else if (index < range * 6)
	{	// green
		return color_t(0, color, inverted_color);
	}
	else
	{	// black
		return color_t(0, 0, 0);
	}
}


GPU_ENABLED constexpr color_t map_colors2(size_t  index)
{
	constexpr unsigned char const max_color = std::numeric_limits<unsigned char>::max();
	size_t const range = max_color + 1;

	unsigned char  color = static_cast<unsigned char>(index);
	unsigned char  inverted_color = max_color - color;

	// white
	if (index < range / 32)
	{	// yellow
		index = index * 32;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(max_color, max_color, inverted_color);
	}
	else if (index < range / 16)
	{	// red
		index = index * 16;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(max_color, inverted_color, 0);
	}
	else if (index < range / 8)
	{	// pink
		index = index * 8;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(max_color, 0, color);
	}
	else if (index < range / 4)
	{	// light blue
		index = index * 4;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(inverted_color, color, max_color);
	}
	else if (index < range / 2)
	{	// blue
		index = index * 2;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(0, inverted_color, max_color);
	}
	else if (index < range - 1)
	{	// green
		index = index;
		color = static_cast<unsigned char>(index);
		inverted_color = max_color - color;
		return color_t(0, color, inverted_color);
	}
	else
	{	// black
		return color_t(0, 0, 0);
	}
}

GPU_ENABLED color_t mandelbrot_iterate_color(complex_t const& z, complex_t const& c)
{
	//pfc::complex<double> z2;
	//z2.real = z.real.toDouble();
	//z2.imag = z.imag.toDouble();

	//pfc::complex<double> c2;
	//c2.real = c.real.toDouble();
	//c2.imag = c.imag.toDouble();

	size_t const count = mandelbrot_iterate(z, c);
	return map_colors2(count);
}


#include "extern/glm-0.9.9.5/glm/glm/glm.hpp"

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;
};