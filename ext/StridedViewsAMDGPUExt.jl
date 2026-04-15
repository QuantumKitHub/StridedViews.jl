module StridedViewsAMDGPUExt

using StridedViews
using AMDGPU
using AMDGPU: Adapt

const ROCStridedView{T, N, A <: ROCArray{T}} = StridedView{T, N, A}

function Base.pointer(x::ROCStridedView{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{Ptr{T}}, a::ROCStridedView{T}) where {T}
    return convert(Ptr{T}, pointer(a))
end

function Base.print_array(io::IO, X::ROCStridedView)
    return Base.print_array(io, Adapt.adapt(Array, X))
end

end # module
