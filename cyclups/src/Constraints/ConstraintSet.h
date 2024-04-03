#pragma once
#include <vector>
#include "../customTypes.h"
#include "Constraint.h"
#include "../Engine/OptimiserProperties.h"
namespace cyclups::constraint
{
	
	/*
		WARNING

		The constraint set is a finnicky object. In order to deal with Constraint heterogeneity, it contains a vector of std::unique_ptr<Constraint>, which can point to subclasses of the Constraint type. 
		This means that copying a ConstraintSet is impossible. The copy constructor defined below necessarily deletes all the unique_ptrs in the original entity, leaving it in a impotent (but still technically valid) state.
		In addition to this, constraints are also stateful objects.
		In short: the code is written (hopefully) to be robust against simple, linear usage of the ConstraintObject -- and where there are errors, it can detect them, rather than failing silently. 
		This guarantee collapses if you attempt to use a ConstraintSet across a parallel implementation. Do not do this! Always make sure a ConstraintSet is initialised *within* the asynchronous portion of the code, not shared across the boundary.
	
	*/
	class ConstraintSet
	{
		public:
			bool IsConstant = false; //true if TransformDimension==0, i.e. only exact constraints present.
			int Dimension; //the total dimension of c, including Optimising and Non-optimising parameters.
			int TransformDimension; //the total dimension of W -- i.e. the number of optimisable parameters. Often (but not always) equal to the sum of Dimension of the non-constant constraints.
			
			Matrix B; //the (constant) constraint Matrix B.
			
			


			ConstraintSet();

			/*
				This is a dangerous function! 
				It deletes the data within copy, without rendering it in a invalid state.
				We implement our own flag which is applied to copy (see PotencyCheck()), and store a pointer, such that when this object is destroyed, copy gets its data back -- so the invalid state lasts only as long as the calling object
			*/
			ConstraintSet(ConstraintSet & copy);
			
			//need a custom destructor in order to pass the data taken from 'copy' (see above) back to its original state
			~ConstraintSet();


			//Templated function for initialising a ConstraintSet with a single Constraint::Subclass object. Membership of the Constraint::Subclass family enforced by the behaviour of the Add function.
			template <class T>
			ConstraintSet(T s)
			{
				Add(s);
			}

			//Templated function for storing a pointer to a provided Constraint::Subclass object. This method generates a copy of the input constraint on the heap, so the Constraint is fully reusable.
			//Membership of Constraint::Subclass is enforced by the fact that Constraints is an object of type std::vector<std::unique_ptr<Constraint>>, and so can only accept push_backs if T is of a suitable type. 
			template<class T>
			void Add(T s)
			{
				auto y = std::make_unique<T>(s);
				Constraints.push_back(std::move(y));
				
			}

			//transfer (destructively) all constraints within c into the present object. Useful for concatenating several objects.
			void Add(ConstraintSet c);

			//removes the final element from the constraint list
			void Remove();

			//The meat of the destructor function. Used to prematurely 'kill' the present object and return the data within it back to the original owners. 
			void Retire();
			
			//Sets the initial position (in W) of all Optimising constraints
			void SetPosition(Vector c);

			//Calls the initialisation functions on all constraints -- they use t to infer their own dimensions and build their matrix and vectors, which are then collated together
			void Initialise(const std::vector<double> & t);

			//Constructs the current value of c, based on the current optimisation state. If IsConstant ==true, then simply returns c.
			Vector & C();

			//assembles the block-gradients from each individual constraint
			Matrix & Gradient();
			
			//Slices the gradient up, and gives them to each Constraint in turn, in order to take an optimisation step
			void Step(Vector & grad,int l,const OptimiserProperties & op);

			//forces each Constraint to make a copy of their current value of w (usually because it is the most optimal position thus far)
			void SavePosition();
			
			//forces each Constraint to set their value of w to that assigned during SavePosition. Usually called at the end of the optimisation to ensure the final value is the most-optimal one.
			void RecoverPosition();
			
		private:
			//flags for dealing with the problems imposed by unique_ptr
			bool WasCopyConstructed = false; 
			bool Impotent = false; //true if this object has been used to copy-construct another object, or was used as the argument to an Add() call. If this is true and the current object is used in any fashion, throws a unrecoverable error.
			ConstraintSet * DataOrigin; // a pointer to the object which was used as the copy-constructor -- so that the data can be returned when Retire() is called.
			
			Vector c;
			std::vector<std::unique_ptr<Constraint>> Constraints;
			Matrix Grad;

			//Inverse of add -- moves (destructively) all constraints within the current object into the target object. Used within the Destructor to return pointers back to where they came from.
			void Transfer(ConstraintSet * c);

			//called as part of most operations. If Impotent==true, then this throws an unrecoverable error.
			void PotencyCheck();
			
	};


	//some cute operator overloads for the Add function

	inline ConstraintSet operator +(ConstraintSet & c1,  ConstraintSet & c2)
	{
		ConstraintSet out = c1;
		out.Add(c2);
		return out;
	}
	inline ConstraintSet operator +=(ConstraintSet & c1, ConstraintSet & c2)
	{
		c1.Add(c2);
		return c1;
	}
	template< class T, class S>
	inline ConstraintSet operator +(T & c1, S & c2)
	{
		ConstraintSet out = c1;

		out.Add(c2);
		return out;
	}
	
}