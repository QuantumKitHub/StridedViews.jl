using Test, StridedViews

# A dense suballocation with conservative storage identities and a precise parent
# alias check, like JLArray and CuArray. This keeps the test independent of GPUs.
struct AliasTestArray <: DenseArray{Float64, 1}
    data::Vector{Float64}
    region::UnitRange{Int}
end
Base.size(A::AliasTestArray) = (length(A.region),)
Base.strides(::AliasTestArray) = (1,)
Base.IndexStyle(::Type{AliasTestArray}) = IndexLinear()
Base.getindex(A::AliasTestArray, i::Int) = A.data[A.region[i]]
Base.dataids(A::AliasTestArray) = Base.dataids(A.data)
Base.mightalias(A::AliasTestArray, B::AliasTestArray) =
    A.data === B.data && !isdisjoint(A.region, B.region)

@testset "StridedView alias detection" begin
    data = zeros(16)
    left = StridedView(AliasTestArray(data, 1:8))
    right = StridedView(AliasTestArray(data, 9:16))
    overlap = StridedView(AliasTestArray(data, 5:12))
    other = StridedView(AliasTestArray(copy(data), 1:8))
    @test Base.dataids(left) == Base.dataids(right)
    @test Base.mightalias(left, left)
    @test !Base.mightalias(left, right)
    @test !Base.mightalias(right, left)
    @test Base.mightalias(left, overlap)
    @test Base.mightalias(overlap, left)
    @test Base.mightalias(right, overlap)
    @test !Base.mightalias(left, other)
    @test !Base.mightalias(other, left)
    @test !Base.mightalias(transpose(sreshape(left, (2, 4))), transpose(sreshape(right, (2, 4))))
    @test Base.mightalias(conj(left), overlap)
    @test !Base.mightalias(sview(left, 1:2:8), right)
    @test Base.mightalias(sview(left, 1:2:8), overlap)
    @test !Base.mightalias(sview(left, 3:2), left)
    @test !Base.mightalias(left, sview(left, 3:2))

    # For ordinary arrays, preserve conservative overlap detection and the empty
    # array behavior even when the normalized parent itself is nonempty.
    ordinary = StridedView(data)
    @test Base.mightalias(ordinary, sview(ordinary, 1:2:16))
    @test !Base.mightalias(ordinary, StridedView(copy(data)))
    @test !Base.mightalias(sview(ordinary, 3:2), ordinary)
    @test !Base.mightalias(ordinary, sview(ordinary, 3:2))
end
