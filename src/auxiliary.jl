# Auxiliary routines
#---------------------
# Errors are thrown from `@noinline` functions, so that building the exception and its
# message does not end up in the code of the (often inlined) callers. Messages with
# interpolation are `LazyString`s, so that they are only built when they are shown.
@noinline function _throw_reshape_dimensionmismatch(oldsize::Dims, newsize::Dims)
    throw(DimensionMismatch(lazy"Cannot reshape a StridedView of size $oldsize (length $(prod(oldsize))) to size $newsize (length $(prod(newsize)))"))
end
@noinline function _throw_invalidperm(N::Int, p)
    throw(ArgumentError(lazy"Invalid permutation of length $N: $p"))
end
@noinline function _throw_reinterpret_eltype(S::Type, T::Type)
    throw(ArgumentError(lazy"Cannot create StridedView with reinterpretation from $S to $T"))
end
@noinline function _throw_reinterpret_op()
    throw(ArgumentError("Cannot create StridedView with reinterpretation from view with non-identity operation"))
end

# Check whether p is a valid permutation of length N
# For N <= 64, the entries are recorded as bits in a mask: N entries that are all in `1:N` form
# a permutation if and only if together they set all N bits. This takes O(N) operations
# without branches or allocations, whereas `Base.isperm` compares all pairs for tuples
# (O(N^2)) and allocates for vectors. For N <= 4, `Base.isperm` is just as fast for tuples.
@inline function _isperm(N::Integer, p::NTuple{M, Integer}) where {M}
    M == N || return false
    # `M` is known at compile time, so only one of the branches remains
    (M <= 4 || M > 64) && return isperm(p)
    return _isperm_mask(p, M)
end
function _isperm(N::Integer, p::AbstractVector{<:Integer})
    length(p) == N || return false
    N > 64 && return isperm(p)
    return _isperm_mask(p, Int(N))
end
_isperm(N::Integer, p::AbstractVector) = length(p) == N && isperm(p)
_isperm(N::Integer, p) = false

@inline function _isperm_mask(p, N::Int)
    used = zero(UInt64)
    valid = true
    for x in p
        valid &= (1 <= x <= N)
        # `% Int` and `& 63` keep the shift well defined (and branch free) for invalid entries,
        # which are rejected through `valid`
        used |= one(UInt64) << (((x % Int) - 1) & 63)
    end
    full = N == 64 ? typemax(UInt64) : (one(UInt64) << N) - one(UInt64)
    return valid & (used == full)
end

# Compute the memory index given a list of cartesian indices and corresponding strides
@inline function _computeind(indices::NTuple{N, Int}, strides::NTuple{N, Int}) where {N}
    return (indices[1] - 1) * strides[1] + _computeind(tail(indices), tail(strides))
end
_computeind(indices::Tuple{}, strides::Tuple{}) = 1

# 'Simplify' the dimensions of a `StridedView` to represent it with the largest possible
# contiguous dimensions (thereby potentially fusing subsequent dimensions), without changing
# the order of the elements. For type stability, we do not remove dimensions but replace
# them with dimensions of size 1 and move all of those to the end.
_simplifydims(size::Tuple{}, strides::Tuple{}) = size, strides
_simplifydims(size::Dims{1}, strides::Dims{1}) = size, strides
function _simplifydims(size::Dims{N}, strides::Dims{N}) where {N}
    tailsize, tailstrides = _simplifydims(tail(size), tail(strides))
    if size[1] == 1
        return (tailsize..., 1), (tailstrides..., 1)
    elseif size[1] * strides[1] == tailstrides[1]
        return (size[1] * tailsize[1], tail(tailsize)..., 1),
            (strides[1], tail(tailstrides)..., 1)
    else
        return (size[1], tailsize...), (strides[1], tailstrides...)
    end
end

# 'Normalize' the strides of a `StridedView`, i.e. strides associated with dimensions of
# size 1 have no intrinsic meaning and can be changed arbitrarily. They are set to the extent
# of the preceding dimension (stride times size), or 1 for leading dimensions of size 1, such
# that a size 1 dimension never prevents fusing its neighbours. If one of the dimensions has
# size zero, then the whole array has length zero, and all strides are ambiguous; they are
# set to those of a contiguous array.
# The strides are computed with a fully unrolled scan that carries the extent of the
# preceding dimension.
@inline function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
    any(iszero, size) && return _contiguousstrides(size)
    return first(_normalizestrides_scan(size, strides, 1))
end
_normalizestrides_scan(::Tuple{}, ::Tuple{}, carry::Int) = ((), carry)
@inline function _normalizestrides_scan(size::Dims, strides::Dims, carry::Int)
    s = ifelse(size[1] == 1, carry, strides[1])
    rest, _ = _normalizestrides_scan(tail(size), tail(strides), size[1] * s)
    return (s, rest...), carry
end
_contiguousstrides(::Tuple{}) = ()
@inline _contiguousstrides(size::Dims) = (1, Base.front(cumprod(size))...)

# 'Normalize' the layout of a DenseArray, in order to reduce the number of required
# specializations in functions. The offset is adjusted to the start of `A` in the
# normalized parent.
@inline _normalizeparent(A, offset::Int) = A, offset
@static if isdefined(Core, :Memory)
    @inline _normalizeparent(A::Array, offset::Int) =
        A.ref.mem, offset + Base.memoryrefoffset(A.ref) - 1
else
    @inline _normalizeparent(A::Array, offset::Int) = reshape(A, length(A)), offset
end

# Auxiliary methods for `sview`
#------------------------------
# Compute the new dimensions of a strided view given the original size and the view slicing
# indices
@inline function _computeviewsize(oldsize::NTuple{N, Int}, I::NTuple{N, SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
_computeviewsize(::Tuple{}, ::Tuple{}) = ()

# Compute the new strides of a (strided) view given the original strides and the view
# slicing indices
@inline function _computeviewstrides(
        oldstrides::NTuple{N, Int},
        I::NTuple{N, SliceIndex}
    ) where {N}
    if isa(I[1], Integer)
        return _computeviewstrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else
        return (
            oldstrides[1] * step(I[1]),
            _computeviewstrides(tail(oldstrides), tail(I))...,
        )
    end
end
_computeviewstrides(::Tuple{}, ::Tuple{}) = ()

# Compute the additional offset of a (strided) view given the original strides and the view
# slicing indices
@inline function _computeviewoffset(
        strides::NTuple{N, Int},
        I::NTuple{N, SliceIndex}
    ) where {N}
    if isa(I[1], Colon)
        return _computeviewoffset(tail(strides), tail(I))
    else
        return (first(I[1]) - 1) * strides[1] + _computeviewoffset(tail(strides), tail(I))
    end
end
_computeviewoffset(::Tuple{}, ::Tuple{}) = 0

# Auxiliary methods for `sreshape`
#----------------------------------
# Compute the new strides of a (strided) reshape given the original strides and new and
# original sizes, or `nothing` if the reshape is not possible without copying.
# Callers guarantee that `prod(newsize) == prod(oldsize)`, which is preserved in the
# recursion: once one of the sizes is exhausted, the remaining dimensions of the other one
# all have size 1.
_computereshapestrides(newsize::Tuple{}, oldsize::Tuple{}, strides::Tuple{}) = strides
_computereshapestrides(newsize::Tuple{}, oldsize::Dims, strides::Dims) = ()
_computereshapestrides(newsize::Dims, oldsize::Tuple{}, strides::Tuple{}) = newsize
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, strides::Dims{N}) where {N}
    d, r = divrem(oldsize[1], newsize[1])
    if r == 0
        s1 = strides[1]
        if d == 1
            # not shrinking the following tuples helps type inference
            oldsize = (tail(oldsize)..., 1)
            strides = (tail(strides)..., 1)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        else
            oldsize = (d, tail(oldsize)...)
            strides = (newsize[1] * s1, tail(strides)...)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        end
    else
        return nothing
    end
end
