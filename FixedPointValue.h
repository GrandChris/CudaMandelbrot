///////////////////////////////////////////////////////////////////////////////
// File:		  FixedPointValue.h
// Revision:	  1
// Date Creation: 07.11.2019
// Last Change:	  07.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  Same as float, but with fixed exponent 
///////////////////////////////////////////////////////////////////////////////

#pragma once

template<size_t nrUnsignedInts = 2, size_t exponentSize = 2>
class FixedPointValue
{
public:

	constexpr FixedPointValue operator+(FixedPointValue const& right) const;
	constexpr FixedPointValue operator*(FixedPointValue const& right) const;


	

private:
	unsigned int mValue[nrUnsignedInts] = { 0 };
};


// #######+++++++ Implementation +++++++#######

template<size_t nrUnsignedInts, size_t exponentSize>
inline constexpr FixedPointValue<nrUnsignedInts, exponentSize>
FixedPointValue<nrUnsignedInts, exponentSize>::operator+(FixedPointValue const& right) const
{
	FixedPointValue result;

	unsigned long lastRes = 0;
	for (size_t i = 0; i < nrUnsignedInts; ++i)
	{
		unsigned long res = static_cast<unsigned long>(mValue[i]) + right.mValue[i];
		result.mValue[i] = static_cast<unsigned int>(res) + static_cast<unsigned int>(res >> sizeof(unsigned int) * 8);
	}

	return result;
}

template<size_t nrUnsignedInts, size_t exponentSize>
inline constexpr FixedPointValue<nrUnsignedInts, exponentSize> 
	FixedPointValue<nrUnsignedInts, exponentSize>::operator*(FixedPointValue const& right) const
{



	return FixedPointValue();
}


