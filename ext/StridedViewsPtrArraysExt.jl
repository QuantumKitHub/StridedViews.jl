module StridedViewsPtrArraysExt

using StridedViews
using PtrArrays

StridedViews._normalizeparent(A::PtrArray, offset::Int) = PtrArray(A.ptr, length(A)), offset

end
