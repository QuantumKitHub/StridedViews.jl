# Precompilation workload
# ------------------------
# Cache the core `StridedView` specializations for the BLAS element types over a small
# range of dimensionalities and the op-wrappers a `StridedView` can carry. These are the
# specializations that downstream packages (e.g. TensorOperations / Strided) hit on their
# first call, so warming them here removes that first-call latency.
#
# The workload is deliberately kept small (BLAS floats, ndims 1:4, identity/conj plus the
# 2D transpose/adjoint cases) so that it adds only a bounded amount to StridedViews' own
# precompile time.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            # construction + property queries + core ops for ndims 1:4
            for N in 1:4
                A = Array{T, N}(undef, ntuple(_ -> 2, N))
                sv = StridedView(A)
                size(sv)
                strides(sv)
                offset(sv)
                conj(sv)
                # permute through the identity permutation (exercises the per-N path)
                permutedims(sv, ntuple(identity, N))
                # reshape to a flat vector and back (also exercises sview on the flat view)
                flat = sreshape(sv, (length(sv),))
                sview(flat, 1:length(sv))
                getindex(sv, ntuple(_ -> 1, N)...)
            end
            # 2D matrix wrappers: transpose / adjoint
            M = Array{T, 2}(undef, 2, 2)
            svM = StridedView(M)
            transpose(svM)
            adjoint(svM)
            # a representative 4D slice view (the SliceIndex `getindex` construction path)
            A4 = Array{T, 4}(undef, 2, 2, 2, 2)
            sv4 = StridedView(A4)
            getindex(sv4, :, 1:2, 1, 1:2)
        end
    end
end
