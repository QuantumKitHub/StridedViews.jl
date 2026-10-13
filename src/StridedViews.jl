module StridedViews

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims
const SliceIndex = Union{RangeIndex, Colon}

using LinearAlgebra
export StridedView, sreshape, sview, isstrided
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :unsafe_stridedview))
end

include("auxiliary.jl")
include("stridedview.jl")
include("precompile.jl")

end
