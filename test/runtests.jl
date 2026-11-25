using Test
using Aqua
using LinearAlgebra
using Random
using StridedViews
using Adapt
using PtrArrays, JLArrays, CUDACore, AMDGPU, Metal

Random.seed!(1234)

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    @testset "construction of StridedView" begin
        @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A1 = randn(T1, (60, 60))
            B1 = StridedView(A1)
            C1 = StridedView(B1)
            @test isstrided(A1)
            @test isstrided(B1)
            @test C1 === B1
            @test parent(B1) == reshape(A1, :)
            @test Base.elsize(B1) == Base.elsize(A1)
            for op1 in (identity, conj, transpose, adjoint)
                if op1 == transpose || op1 == adjoint
                    @test op1(A1) == op1(B1) == StridedView(op1(A1))
                else
                    @test op1(A1) == op1(B1)
                end
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A1)) == op2(op1(B1))
                end
            end

            A2 = view(A1, 1:36, 1:20)
            @test isstrided(A2)
            B2 = StridedView(A2)
            for op1 in (identity, conj, transpose, adjoint)
                if op1 == transpose || op1 == adjoint
                    @test op1(A2) == op1(B2) == StridedView(op1(A2))
                else
                    @test op1(A2) == op1(B2)
                end
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A2)) == op2(op1(B2))
                end
            end

            A3 = reshape(A1, 360, 10)
            @test isstrided(A3)
            @test !isstrided(reshape(A1', 10, 360))
            B3 = StridedView(A3)
            @test size(A3) == size(B3)
            @test strides(A3) == strides(B3)
            @test stride(A3, 1) == stride(B3, 1)
            @test stride(A3, 2) == stride(B3, 2)
            @test stride(A3, 3) == stride(B3, 3)
            for op1 in (identity, conj, transpose, adjoint)
                if op1 == transpose || op1 == adjoint
                    @test op1(A3) == op1(B3) == StridedView(op1(A3))
                else
                    @test op1(A3) == op1(B3)
                end
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A3)) == op2(op1(B3))
                end
            end

            A4 = reshape(A1', (6, 10, 5, 12))
            @test isstrided(A4)
            B4 = StridedView(A4)
            B4[2, 2, 2, 2] = 3
            @test A4[2, 2, 2, 2] == 3
            for op1 in (identity, conj)
                @test op1(A4) == op1(B4)
                for op2 in (identity, conj)
                    @test op2(op1(A4)) == op2(op1(B4))
                end
            end

            A4 = reshape(view(A1, 1:36, 1:20), (6, 6, 5, 4))
            @test isstrided(A4)
            B4 = StridedView(A4)
            B4[2, 2, 2, 2] = 4
            @test A4[2, 2, 2, 2] == 4
            for op1 in (identity, conj)
                @test op1(A4) == op1(B4)
                for op2 in (identity, conj)
                    @test op2(op1(A4)) == op2(op1(B4))
                end
            end

            A4 = reshape(view(A1', 1:36, 1:20), (6, 6, 5, 4))
            B4 = StridedView(A4)
            B4[2, 2, 2, 2] = 5
            @test A4[2, 2, 2, 2] == 5
            for op1 in (identity, conj)
                @test op1(A4) == op1(B4)
                for op2 in (identity, conj)
                    @test op2(op1(A4)) == op2(op1(B4))
                end
            end

            A5 = PermutedDimsArray(reshape(view(A1, 1:36, 1:20), (6, 6, 5, 4)), (3, 1, 2, 4))
            @test isstrided(A5)
            B5 = StridedView(A5)
            for op1 in (identity, conj)
                @test op1(A5) == op1(B5)
                for op2 in (identity, conj)
                    @test op2(op1(A5)) == op2(op1(B5))
                end
            end

            A6 = reshape(view(A1, 1:36, 1:20), (6, 120))
            @test !isstrided(A6)
            @test_throws StridedViews.ReshapeException StridedView(A6)
            try
                StridedView(A6)
            catch ex
                println("Printing error message:")
                show(ex)
                println("")
            end

            # Array with Array elements
            A7 = [randn(T1, (5, 5)) for i in 1:5, j in 1:5]
            @test isstrided(A7)
            B7 = StridedView(A7)
            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A7) == op1(B7) == StridedView(op1(A7))
                for op2 in (identity, conj, transpose, adjoint)
                    @test op2(op1(A7)) == op2(op1(B7))
                end
            end

            # Zero-dimensional array
            A8 = randn(T1, ())
            B8 = StridedView(A8)
            @test stride(B8, 1) == stride(B8, 5) == 1
            for op1 in (identity, conj)
                @test op1(A8) == op1(B8) == StridedView(op1(A8))
                for op2 in (identity, conj)
                    @test op2(op1(A8)) == op2(op1(B8))
                end
            end
            @test reshape(B8, (1, 1, 1)) == reshape(A8, (1, 1, 1)) ==
                StridedView(reshape(A8, (1, 1, 1))) == sreshape(A8, (1, 1, 1))
            @test reshape(B8, ()) == reshape(A8, ())
        end

        @test !isstrided(Diagonal([0.5, 1.0, 1.5]))
    end

    @testset "arrays not starting at the beginning of their memory" begin
        v = collect(1.0:6.0)
        popfirst!(v)
        @test StridedView(v) == v
        @test pointer(StridedView(v)) == pointer(v)
        M = reshape(deleteat!(collect(1.0:8.0), 1:2), 2, 3)
        @test StridedView(M) == M
        @test StridedView(M)[:, 2:3] == M[:, 2:3]
        @test sview(M, 2, :) == M[2, :]
        StridedView(M)[2, 1] = 0
        @test M[2, 1] == 0
    end

    @testset "transpose and adjoint with vector StridedView" begin
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A = randn(T, (60,))

            @test sreshape(transpose(A), (1, length(A))) == transpose(A)
            @test sreshape(adjoint(A), (1, length(A))) == adjoint(A)
            @test isstrided(transpose(A))
            @test isstrided(adjoint(A))
        end
    end

    @testset "elementwise conj, transpose and adjoint" begin
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A = [rand(T) for i in 1:5, b in 1:4, c in 1:3, d in 1:2]
            B = StridedView(A)

            @test conj(B) == conj(A)
            @test conj(B) == map(conj, B)
            @test map(transpose, B) == map(transpose, A)
            @test map(adjoint, B) == map(adjoint, A)
        end
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A = [randn(T, (3, 3)) for i in 1:5, b in 1:4, c in 1:3, d in 1:2]
            B = StridedView(A)
            @test Base.elsize(B) == Base.elsize(A)

            @test conj(B) == conj(A)
            @test conj(B) == map(conj, B)
            @test map(transpose, B) == map(transpose, A)
            @test map(adjoint, B) == map(adjoint, A)
        end
    end

    @testset "reshape and permutedims with StridedView" begin
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A0 = randn(T, 10)
            @test permutedims(StridedView(A0), (1,)) == A0

            @testset "reshape and permutedims with $N-dimensional arrays" for N in 2:6
                let dims = ntuple(n -> rand(1:div(60, N)), N)
                    A = rand(T, dims)
                    B = StridedView(A)
                    @test conj(A) == conj(B)
                    p = randperm(N)
                    B2 = permutedims(B, p)
                    A2 = permutedims(A, p)
                    @test B2 == A2
                end

                let dims = ntuple(n -> 6, N)
                    A = rand(T, dims)
                    B = StridedView(A)
                    @test conj(A) == conj(B)
                    p = randperm(N)
                    B2 = permutedims(B, p)
                    A2 = permutedims(A, p)
                    @test B2 == A2

                    B2 = sreshape(B, (2, 3, ntuple(n -> 6, N - 2)..., 3, 2))
                    A2 = sreshape(A, (2, 3, ntuple(n -> 6, N - 2)..., 3, 2)...)
                    A3 = reshape(A, size(A2))
                    @test B2 == A3
                    @test B2 == A2
                    p = randperm(N + 2)
                    @test conj(permutedims(B2, p)) == conj(permutedims(A3, p))
                end
            end

            @testset "more reshape" begin
                A = randn(4, 0)
                B = StridedView(A)
                @test sreshape(B, size(B)) === B
                @test_throws DimensionMismatch sreshape(B, (4, 1))
                C = sreshape(B, (2, 1, 2, 0, 1))
                @test strides(C) == (1, 2, 2, 4, 0)
                @test sreshape(C, (4, 0)) == A

                A = randn(4, 1, 2)
                B = StridedView(A)
                @test_throws DimensionMismatch sreshape(B, (4, 4))
                @test sreshape(B, size(B)) === B
                @test strides(permutedims(B, (2, 1, 3))) == (1, 1, 4)
                @test strides(B[1:2:3, :, :]) == (2, 4, 4)
                @test StridedViews.unsafe_stridedview(
                    eltype(B), B.parent, size(B), strides(B), B.offset, B.op
                ) === B
                @test @inferred(StridedView(A)) == A
                C = sreshape(B, (2, 1, 1, 4, 1, 1))
                @test strides(C) == (1, 2, 2, 2, 8, 8)
                @test C == reshape(A, (2, 1, 1, 4, 1, 1))
                @test sreshape(C, (4, 1, 2)) == A
            end
        end
    end

    @testset "stride normalization" begin
        # reference implementation (previous version of `_normalizestrides`)
        function normalizestrides_ref(size::Dims{N}, strides::Dims{N}) where {N}
            for i in 1:N
                if size[i] == 1
                    newstride = i == 1 ? 1 : strides[i - 1] * size[i - 1]
                    strides = Base.setindex(strides, newstride, i)
                elseif size[i] == 0
                    return (1, Base.front(cumprod(size))...)
                end
            end
            return strides
        end
        @test @inferred(StridedViews._normalizestrides((), ())) == ()
        for N in 1:6, _ in 1:200
            sz = ntuple(_ -> rand((0, 1, 1, 2, 3)), N)
            st = ntuple(_ -> rand(-5:20), N)
            @test @inferred(StridedViews._normalizestrides(sz, st)) ==
                normalizestrides_ref(sz, st)
        end
        @test StridedViews._normalizestrides((1, 1, 3, 1, 2), (7, 7, 5, 7, 2)) ==
            (1, 1, 5, 15, 2)
    end

    @testset "unsafe_stridedview with a StridedView" begin
        A = randn(ComplexF64, 4, 3, 2)
        B = conj(StridedView(A))
        @test StridedViews.unsafe_stridedview(B) === B
        C = @inferred StridedViews.unsafe_stridedview(B, (4, 6), (1, 4))
        @test C isa StridedView{ComplexF64, 2, typeof(B.parent), typeof(conj)}
        @test C == conj(reshape(A, (4, 6)))
        # strides are used as given, e.g. zero strides for broadcasting
        D = StridedViews.unsafe_stridedview(B, (4, 5), (1, 0), StridedViews.offset(B) + 4)
        @test D == repeat(conj(A[:, 2, 1]), 1, 5)
        @test strides(D) == (1, 0)
    end

    @testset "permutation check" begin
        mask(p) = StridedViews._isperm_mask(p, length(p))
        # exhaustive for small lengths, including out of range entries
        for N in 0:5, t in Iterators.product(ntuple(_ -> -1:(N + 1), N)...)
            @test mask(t) == isperm(t)
            @test StridedViews._isperm(N, t) == isperm(t)
            @test StridedViews._isperm(N, collect(Int, t)) == isperm(t)
        end
        for N in (6, 10, 17, 63, 64, 65), _ in 1:50
            p = randperm(N)
            @test StridedViews._isperm(N, Tuple(p))
            @test StridedViews._isperm(N, p)
            @test StridedViews._isperm(N, UInt8.(p))
            q = copy(p)
            q[rand(1:N)] = rand((0, -1, N + 1, typemin(Int), p[mod1(rand(1:N), N)]))
            isperm(q) && continue # the replacement can be the original entry
            @test !StridedViews._isperm(N, Tuple(q))
            @test !StridedViews._isperm(N, q)
        end
        @test !StridedViews._isperm(3, (1, 2))
        @test !StridedViews._isperm(2, [1, 2, 3])
        @test !StridedViews._isperm(2, (UInt(2)^63 + 1, 1))
        @test !StridedViews._isperm(2, "ab")
    end

    @testset "bounds checks when slicing" begin
        A = randn(3, 4, 5)
        B = StridedView(A)
        @test_throws BoundsError B[1:4, :, :]
        @test_throws BoundsError B[0:2, :, 1]
        @test_throws BoundsError B[:, 5, :]
        @test_throws BoundsError view(B, :, :, 2:6)
        @test_throws BoundsError sview(B, :, 1:2:5, 1)
        @test_throws BoundsError sview(B, 1:61)
        @test_throws BoundsError sview(A, 3:4, :, :)
        @test B[3:-1:1, 4:-2:1, 1:0] == A[3:-1:1, 4:-2:1, 1:0]
        @test sview(B, 60:-7:1) == vec(A)[60:-7:1]
    end

    @testset "errors" begin
        B = StridedView(randn(2, 3, 4))
        err = try
            permutedims(B, (1, 1, 2))
        catch e
            e
        end
        @test err isa ArgumentError
        @test err.msg == "Invalid permutation of length 3: (1, 1, 2)"
        @test_throws ArgumentError permutedims(B, (1, 2))
        @test_throws ArgumentError permutedims(B, [3, 1])
        @test_throws DimensionMismatch sreshape(B, (5, 5))
        err = try
            sreshape(B, (5, 5))
        catch e
            e
        end
        @test sprint(showerror, err) ==
            "DimensionMismatch: Cannot reshape a StridedView of size (2, 3, 4) (length 24) to size (5, 5) (length 25)"
        @test_throws DimensionMismatch sreshape(StridedView(zeros(2, 0)), (3, 1))
        @test_throws DimensionMismatch sreshape(B, (2, 3, 4, 0))
        @test_throws StridedViews.ReshapeException sreshape(B[:, 1:2, :], (2, 8))
        @test_throws ArgumentError StridedView(reinterpret(Float32, randn(4)))
        A = randn(ComplexF64, 4)
        @test_throws ArgumentError StridedView(reinterpret(NTuple{2, Float64}, conj(StridedView(A))))
    end

    @testset "views with StridedView" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = randn(T, (10, 10, 10, 10))
            B = StridedView(A)
            @test isa(view(B, :, 1:5, 3, 1:5), StridedView)
            @test isa(view(B, :, [1, 2, 3], 3, 1:5), Base.SubArray)
            @test isa(sview(B, :, 1:5, 3, 1:5), StridedView)
            @test_throws MethodError sview(B, :, [1, 2, 3], 3, 1:5)

            @test view(A, 1:38) == view(B, 1:38) == sview(A, 1:38) == sview(B, 1:38)

            @test view(B, :, 1:5, 3, 1:5) == view(A, :, 1:5, 3, 1:5) == sview(A, :, 1:5, 3, 1:5)
            @test view(B, :, 1:5, 3, 1:5) === sview(B, :, 1:5, 3, 1:5) === B[:, 1:5, 3, 1:5]
            @test view(B, :, 1:5, 3, 1:5) == StridedView(view(A, :, 1:5, 3, 1:5))
            @test pointer(view(B, :, 1:5, 3, 1:5)) ==
                pointer(StridedView(view(A, :, 1:5, 3, 1:5)))
            @test StridedViews.offset(view(B, :, 1:5, 3, 1:5)) == 2 * stride(B, 3)
        end
    end

    @testset "reinterpretarray" begin
        struct MyComplex
            re::Float64
            im::Float64
        end
        a = MyComplex.(randn(5, 5, 5), randn(5, 5, 5))
        b = reinterpret(ComplexF64, a)
        sb = StridedView(b)
        csb = conj(sb)
        for index in CartesianIndices(a)
            @test sb[index] == a[index].re + im * a[index].im
            @test csb[index] == a[index].re - im * a[index].im
        end
        b2 = permutedims(reshape(sb, (25, 5)), (2, 1))
        cb2 = b2'
        a2 = permutedims(reshape(a, (25, 5)), (2, 1))
        for j in 1:size(b2, 2)
            for i in 1:size(b2, 1)
                @test b2[i, j] == a2[i, j].re + im * a2[i, j].im
                @test cb2[j, i] == a2[i, j].re - im * a2[i, j].im
            end
        end
    end

    @testset "PtrArrays with StridedView" begin
        @testset for T in (Float64, ComplexF64)
            A = randn!(PtrArrays.malloc(T, 10, 10, 10, 10))
            @test isstrided(A)
            B = StridedView(A)
            @test B isa StridedView
            @test B == A
            PtrArrays.free(A)
        end
    end

    @testset "JLArrays with StridedView" begin
        @testset "alias detection" begin
            buffer = JLArray{Float32}(undef, 16)
            left = view(buffer, 1:8)
            right = view(buffer, 9:16)
            overlap = view(buffer, 5:12)
            # Preserve the parents' alias check, including byte-range specializations.
            for other in (left, right, overlap)
                @test Base.mightalias(StridedView(left), StridedView(other)) ==
                    Base.mightalias(left, other)
            end
            empty = sview(StridedView(left), 3:2)
            @test !Base.mightalias(empty, StridedView(left))
            @test !Base.mightalias(StridedView(left), empty)
        end
        @testset for T in (Float64, ComplexF64)
            Araw = randn(T, 10, 10, 10, 10)
            A = JLArray(Araw)
            @test isstrided(A)
            B = StridedView(A)
            @test B isa StridedView
            JLArrays.@allowscalar begin
                @test B == A
            end
            Bvec = adapt(Vector{T}, B)
            @test Bvec == StridedView(Araw)
            @test parent(adapt(JLArray, Bvec)) isa JLArray
        end
    end

    Aqua.test_all(StridedViews)
end

if CUDACore.functional()
    @testset "CuArrays with StridedView" begin
        @testset for T in (Float64, ComplexF64)
            A = CuArray(randn(T, 10, 10, 10, 10))
            @test isstrided(A)
            B = StridedView(A)
            @test B isa StridedView
            @test parent(B) isa CuArray{T, 1}
            CUDACore.@allowscalar begin
                @test B == A
                @test parent(B) == reshape(A, :)
            end
        end
    end
end

if AMDGPU.functional()
    @testset "ROCArrays with StridedView" begin
        @testset for T in (Float64, ComplexF64)
            A = ROCArray(randn(T, 10, 10, 10, 10))
            @test isstrided(A)
            B = StridedView(A)
            @test B isa StridedView
            @test parent(B) isa ROCArray{T, 1}
            AMDGPU.@allowscalar begin
                @test B == A
                @test parent(B) == reshape(A, :)
            end
        end
    end
end

if Metal.functional()
    @testset "MtlArrays with StridedView" begin
        @testset for T in (Float32, ComplexF32)
            A = MtlArray(randn(T, 10, 10, 10, 10))
            @test isstrided(A)
            B = StridedView(A)
            @test B isa StridedView
            @test parent(B) isa MtlArray{T, 1}
            Metal.@allowscalar begin
                @test B == A
                @test parent(B) == reshape(A, :)
            end
        end
    end
end

if !is_buildkite
    # JET tests: JET ≥ 0.12 (Julia 1.12+) signals through `JET_AVAILABLE` whether it is functional,
    # and loads empty stubs when it is not. Older JET versions, still needed on Julia < 1.12, do not
    # define it, hence the `isdefined` check instead of `using JET: JET_AVAILABLE`.
    using JET: JET
    const jet_new_generation = isdefined(JET, :JET_AVAILABLE)
    const jet_available = !jet_new_generation || JET.JET_AVAILABLE
    # whole-package analysis is pinned to the JET 0.12 generation, so its reports need not be curated for several JET/Julia combinations
    (jet_new_generation && jet_available) && include("jet.jl")
end
