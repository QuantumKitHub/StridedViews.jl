module StridedViewsMetalExt

using StridedViews
using Metal
using Metal: Adapt, MtlPtr

const MtlStridedView{T, N, A <: MtlArray{T}} = StridedView{T, N, A}

function Base.pointer(x::MtlStridedView{T}) where {T}
    return Base.unsafe_convert(MtlPtr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{MtlPtr{T}}, a::MtlStridedView{T}) where {T}
    return convert(MtlPtr{T}, pointer(a))
end

function Base.print_array(io::IO, X::MtlStridedView)
    return Base.print_array(io, Adapt.adapt(Array, X))
end

end # module
