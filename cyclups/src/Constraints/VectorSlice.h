#pragma once

namespace cyclups
{
	struct VectorSlice
	{
		Vector & V;
		int SliceStart;
		int SliceSize;

		VectorSlice(Vector &v): V(v)
		{
			SliceStart = 0;
			SliceSize = v.size();
		}
		VectorSlice(Vector & v, int start,int size): V(v)
		
		{
			SliceStart = start;
			SliceSize = size;
		}
		double & operator[](int i)
		{
			return V[SliceStart + i];
		}
	};

}