//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function LZ5.
 * 
 * The function is described in
 * 
 * H. Li and Q. Zhang. 
 * Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
 * IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
 * 
 * 
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ5_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ5_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {namespace benchmarks{
/*! \brief Multi-objective optimization benchmark function LZ5.
*
*  The function is described in
*
*  H. Li and Q. Zhang. 
*  Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
*  IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009.
* \ingroup benchmarks	
*/
struct LZ5 : public MultiObjectiveFunction
{
	LZ5(std::size_t numVariables = 0) : m_handler(SearchPointType(numVariables,-1),SearchPointType(numVariables,1) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LZ5"; }
	
	std::size_t numberOfObjectives()const{
		return 2;
	}
	
	std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		SearchPointType lb(numberOfVariables,-1);
		SearchPointType ub(numberOfVariables, 1);
		lb(0) = 0;
		ub(0) = 1;
		m_handler.setBounds(lb, ub);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2, 0 );

		std::size_t counter1 = 0, counter2 = 0;
		for( std::size_t i = 1; i < x.size(); i++ ) {
			if( i % 2 == 0 ) {
				counter2++;
				value[1] += sqr(
					x(i) - (
					(0.3*x(0)*x(0)*::cos(24*M_PI*x(0)+4*i*M_PI/x.size()) + 0.6*x(0))*
					::cos( 6 * M_PI * x( 0 ) + i*M_PI/x.size()  )
					)
					);
			} else {
				counter1++;
				value[0] += sqr(
					x(i) - (
					(0.3*x(0)*x(0)*::cos(24*M_PI*x(0)+4*i*M_PI/x.size()) + 0.6*x(0))*
					::sin( 6 * M_PI * x( 0 ) + i*M_PI/x.size()  )
					)
					);
			}
		}

		value[0] *= 2./counter1;
		value[0] += x( 0 );

		value[1] *= 2./counter2;
		value[1] += 1 - ::sqrt( x( 0 ) );

		return value;
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

}}
#endif
