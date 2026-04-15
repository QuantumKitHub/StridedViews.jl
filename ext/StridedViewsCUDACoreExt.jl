module StridedViewsCUDACoreExt

using StridedViews
using CUDACore
using CUDACore: Adapt, CuPtr

const CuStridedView{T, N, A <: CuArray{T}} = StridedView{T, N, A}

function Base.pointer(x::CuStridedView{T}) where {T}
    return Base.unsafe_convert(CuPtr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{CuPtr{T}}, a::CuStridedView{T}) where {T}
    return convert(CuPtr{T}, pointer(a))
end

function Base.print_array(io::IO, X::CuStridedView)
    return Base.print_array(io, Adapt.adapt(Array, X))
end

end # module
