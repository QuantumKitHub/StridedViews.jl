@testset "JET" begin
    import Pkg
    try
        Pkg.activate(joinpath(@__DIR__); io = devnull)
        Pkg.instantiate(; io = devnull)
        @eval import JET
        JET.test_package(StridedViews; target_modules = (StridedViews,))
    catch e
        e isa Pkg.Types.PkgError || rethrow()
        @test_broken false
    finally
        Pkg.activate(@__DIR__; io = devnull)
    end
end
