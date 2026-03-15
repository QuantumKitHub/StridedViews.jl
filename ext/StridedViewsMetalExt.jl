module StridedViewsMetalExt

using StridedViews
using Metal
using Metal: Adapt, MtlPtr

const MtlStridedView{T, N, A <: MtlArray{T}} = StridedView{T, N, A}

function Adapt.adapt_structure(to, A::MtlStridedView)
    return StridedView(
        Adapt.adapt_structure(to, parent(A)),
        A.size, A.strides, A.offset, A.op
    )
end

function Base.pointer(x::MtlStridedView{T}) where {T}
    return Base.unsafe_convert(MtlPtr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{MtlPtr{T}}, a::MtlStridedView{T}) where {T}
    return convert(MtlPtr{T}, pointer(a))
end

function Base.print_array(io::IO, X::MtlStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

end # module
