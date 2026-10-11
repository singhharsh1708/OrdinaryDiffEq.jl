using StochasticDiffEq, DiffEqNoiseProcess, Random, Statistics, Test

const TEND = 1.0
const FINE_POINTS = 2^19
const FINE_DT = TEND / FINE_POINTS
const FINE_GRID = collect(range(0.0, TEND; length = FINE_POINTS + 1))
const STEP_COUNTS = [FINE_POINTS ÷ 2^k for k in 15:-1:9]
const PATHS = 32
const ALGORITHMS = (RandomEM(), RandomHeun(), RandomTamedEM())

function wiener_path(rng)
    W = zeros(FINE_POINTS + 1)
    W[2:end] .= cumsum(sqrt(FINE_DT) .* randn(rng, FINE_POINTS))
    return W
end

function cumulative_trapezoid(g, dt)
    I = zeros(length(g))
    acc = zero(eltype(g))
    @inbounds for k in 1:(length(g) - 1)
        acc += dt * (g[k] + g[k + 1]) / 2
        I[k + 1] = acc
    end
    return I
end

function fitted_slope(step_counts, errors)
    x = log2.(1 ./ step_counts)
    y = log2.(errors)
    n = length(x)
    return (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

decay_oop(u, p, t, W) = -u .* cos(5W)
decay_iip(du, u, p, t, W) = (du .= -u .* cos(5W))

function decay_solution(alg, noise, n, u0; inplace = false)
    prob = if inplace
        RODEProblem(decay_iip, u0, (0.0, TEND), noise = noise)
    else
        RODEProblem{false}(decay_oop, u0, (0.0, TEND), noise = noise)
    end
    return solve(prob, alg, dt = TEND / n, save_everystep = true, adaptive = false)
end

function decay_study(alg; seed = 20260918)
    rng = MersenneTwister(seed)
    errors = zeros(PATHS, length(STEP_COUNTS))
    floors = zeros(PATHS)
    for m in 1:PATHS
        path = wiener_path(rng)
        g = cos.(5 .* path)
        exact = exp.(-cumulative_trapezoid(g, FINE_DT))
        floors[m] = maximum(abs, exp.(-cumulative_trapezoid(g[1:4:end], 4FINE_DT)) .- exact[1:4:end])
        noise = NoiseGrid(FINE_GRID, path)
        for (j, n) in enumerate(STEP_COUNTS)
            sol = decay_solution(alg, noise, n, 1.0)
            stride = FINE_POINTS ÷ n
            errors[m, j] = maximum(abs(sol.u[k + 1] - exact[k * stride + 1]) for k in 0:n)
        end
    end
    strong = [sqrt(mean(errors[:, j] .^ 2)) for j in eachindex(STEP_COUNTS)]
    pathwise = [fitted_slope(STEP_COUNTS, errors[m, :]) for m in 1:PATHS]
    return (
        strong = strong, strong_order = fitted_slope(STEP_COUNTS, strong),
        pathwise_order = median(pathwise), reference_floor = maximum(floors),
    )
end

@testset "classical orders are visible when the noise leaves the right-hand side" begin
    smooth(u, p, t, W) = -u
    exact = exp.(-FINE_GRID)
    noise = NoiseGrid(FINE_GRID, zeros(FINE_POINTS + 1))
    function smooth_error(alg, n)
        prob = RODEProblem{false}(smooth, 1.0, (0.0, TEND), noise = noise)
        sol = solve(prob, alg, dt = TEND / n, save_everystep = true, adaptive = false)
        stride = FINE_POINTS ÷ n
        return maximum(abs(sol.u[k + 1] - exact[k * stride + 1]) for k in 0:n)
    end
    @test abs(fitted_slope(STEP_COUNTS, [smooth_error(RandomEM(), n) for n in STEP_COUNTS]) - 1) < 0.2
    @test abs(fitted_slope(STEP_COUNTS, [smooth_error(RandomHeun(), n) for n in STEP_COUNTS]) - 2) < 0.2
end

@testset "Wiener-driven RODE order: $(nameof(typeof(alg)))" for alg in ALGORITHMS
    study = decay_study(alg)
    @test minimum(study.strong) > 20 * study.reference_floor
    @test abs(study.strong_order - 1) < 0.2
    @test abs(study.pathwise_order - 1) < 0.2
end

@testset "in-place matches out-of-place: $(nameof(typeof(alg)))" for alg in ALGORITHMS
    noise = NoiseGrid(FINE_GRID, wiener_path(MersenneTwister(20260918)))
    u0 = [1.0, 1.0]
    for n in (STEP_COUNTS[1], STEP_COUNTS[4], STEP_COUNTS[end])
        @test decay_solution(alg, noise, n, u0).u ==
            decay_solution(alg, noise, n, copy(u0); inplace = true).u
    end
end

const TAYLOR_STEP_COUNTS = STEP_COUNTS[1:5]

function taylor15_study(alg; seed = 20260921, sigma = 1.0)
    rng = MersenneTwister(seed)
    errors = zeros(PATHS, length(TAYLOR_STEP_COUNTS))
    floors = zeros(PATHS)
    for m in 1:PATHS
        path = sigma .* wiener_path(rng)
        g = cos.(5 .* path)
        exact = exp.(-cumulative_trapezoid(g, FINE_DT))
        floors[m] = maximum(abs, exp.(-cumulative_trapezoid(g[1:4:end], 4FINE_DT)) .- exact[1:4:end])
        noise = NoiseGrid(FINE_GRID, path)
        for (j, n) in enumerate(TAYLOR_STEP_COUNTS)
            sol = decay_solution(alg, noise, n, 1.0)
            stride = FINE_POINTS ÷ n
            errors[m, j] = maximum(abs(sol.u[k + 1] - exact[k * stride + 1]) for k in 0:n)
        end
    end
    strong = [sqrt(mean(errors[:, j] .^ 2)) for j in eachindex(TAYLOR_STEP_COUNTS)]
    return (
        strong = strong, strong_order = fitted_slope(TAYLOR_STEP_COUNTS, strong),
        reference_floor = maximum(floors),
    )
end

@testset "RandomTaylor15 reaches order 1.5 on a resolved path" begin
    taylor = taylor15_study(RandomTaylor15())
    euler = taylor15_study(RandomEM())
    @test minimum(taylor.strong) > 5 * taylor.reference_floor
    @test 1.5 < taylor.strong_order < 2.2
    @test taylor.strong_order > euler.strong_order + 0.5
    @test taylor.strong[end] < euler.strong[end] / 10
end

@testset "RandomTaylor15 keeps its order on a Brownian path of another amplitude" begin
    taylor = taylor15_study(RandomTaylor15(); sigma = 0.2)
    euler = taylor15_study(RandomEM(); sigma = 0.2)
    @test minimum(taylor.strong) > 3 * taylor.reference_floor
    @test 1.5 < taylor.strong_order < 2.3
    @test taylor.strong[end] < euler.strong[end] / 20
end

@testset "RandomTaylor15 is Heun in the deterministic limit" begin
    smooth(u, p, t, W) = -u
    exact = exp.(-FINE_GRID)
    noise = NoiseGrid(FINE_GRID, zeros(FINE_POINTS + 1))
    function smooth_solution(alg, n)
        prob = RODEProblem{false}(smooth, 1.0, (0.0, TEND), noise = noise)
        return solve(prob, alg, dt = TEND / n, save_everystep = true, adaptive = false)
    end
    function smooth_error(alg, n)
        sol = smooth_solution(alg, n)
        stride = FINE_POINTS ÷ n
        return maximum(abs(sol.u[k + 1] - exact[k * stride + 1]) for k in 0:n)
    end
    errors = [smooth_error(RandomTaylor15(), n) for n in STEP_COUNTS]
    @test abs(fitted_slope(STEP_COUNTS, errors) - 2) < 0.2
    taylor = smooth_solution(RandomTaylor15(), STEP_COUNTS[end]).u
    heun = smooth_solution(RandomHeun(), STEP_COUNTS[end]).u
    @test all(isapprox(a, b, rtol = 1.0e-9) for (a, b) in zip(taylor, heun))
end

@testset "RandomTaylor15 in-place matches out-of-place" begin
    noise = NoiseGrid(FINE_GRID, wiener_path(MersenneTwister(20260921)))
    u0 = [1.0, 2.0]
    for n in (TAYLOR_STEP_COUNTS[1], TAYLOR_STEP_COUNTS[end])
        outofplace = decay_solution(RandomTaylor15(), noise, n, u0).u
        inplace = decay_solution(RandomTaylor15(), noise, n, copy(u0); inplace = true).u
        @test all(isapprox(a, b, rtol = 1.0e-9) for (a, b) in zip(outofplace, inplace))
    end
end

@testset "RandomTaylor15 step integrals are exact on a misaligned grid" begin
    grid = collect(range(0.0, 1.0; length = 13))
    integrals = StochasticDiffEq.StochasticDiffEqRODE.path_integrals
    for (t, dt) in ((0.0, 0.25), (0.05, 0.25), (0.03, 0.04), (1 / 12, 1 / 12), (0.5, 0.5))
        W = (t = grid, W = copy(grid), dW = dt, curW = t)
        I1, I2 = integrals(W, t, dt, t)
        @test I1 ≈ dt^2 / 2
        @test I2 ≈ dt^3 / 3
    end
end

@testset "RandomTaylor15 integrates backwards in time" begin
    n = 64
    back_grid = collect(range(TEND, 0.0; length = FINE_POINTS + 1))
    path = reverse(wiener_path(MersenneTwister(20260921)))
    g = cos.(5 .* path)
    exact = exp.(-cumulative_trapezoid(g, -FINE_DT))
    prob = RODEProblem{false}(decay_oop, 1.0, (TEND, 0.0), noise = NoiseGrid(back_grid, path))
    sol = solve(prob, RandomTaylor15(), dt = -TEND / n, save_everystep = true, adaptive = false)
    stride = FINE_POINTS ÷ n
    @test maximum(abs(sol.u[k + 1] - exact[k * stride + 1]) for k in 0:n) < 1.0e-2
end

@testset "RandomTaylor15 rejects noise it cannot resolve" begin
    prob = RODEProblem{false}(decay_oop, 1.0, (0.0, TEND))
    @test_throws "not compatible with the chosen noise type" solve(
        prob, RandomTaylor15(), dt = TEND / 16, adaptive = false
    )
end

using LinearAlgebra

const RODE = StochasticDiffEq.StochasticDiffEqRODE
const BOOK_INDICES_30 = [
    [0], [1], [2], [3], [4], [5], [0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2],
    [2, 1], [1, 2], [3, 0], [0, 3], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
]
const COARSE_POINTS = 2^12
const COARSE_GRID = collect(range(0.0, TEND; length = COARSE_POINTS + 1))
const COARSE_STEP_COUNTS = [2^k for k in 4:8]
const FBM_POINTS = 2^11
const FBM_GRID = collect(range(0.0, TEND; length = FBM_POINTS + 1))
const FBM_STEP_COUNTS = [2^k for k in 1:6]
const UNEVEN_GRID = [0.0; sort([mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:40]); 1.0]
const UNEVEN_PATH = [sin(7s) + cos(23s) / 2 for s in UNEVEN_GRID]
const UNEVEN_GRIDS = ((UNEVEN_GRID, UNEVEN_PATH), (reverse(UNEVEN_GRID), reverse(UNEVEN_PATH)))
const DETERMINISTIC_PATH = [
    sum(2.0^(-k / 2) * sin(2pi * 2.0^k * s + k) for k in 0:10) / 2 for s in COARSE_GRID
]
const DETERMINISTIC_GRIDS = (
    (COARSE_GRID, DETERMINISTIC_PATH), (reverse(COARSE_GRID), reverse(DETERMINISTIC_PATH)),
)

sine_oop(u, p, t, W) = sin(u + W) - u
timed_oop(u, p, t, W) = sin(u + W) - u + cos(3t) * u^2 / 2
timed_iip(du, u, p, t, W) = (du[1] = timed_oop(u[1], p, t, W); nothing)
quadratic_oop(u, p, t, W) = -2t * u^2
polynomial_oop(u, p, t, W) = evalpoly(W, p)
function coupled_oop(u, p, t, W)
    return [
        sin(u[2] + W) - u[1] + cos(3t) * u[1] * u[2] / 2,
        cos(u[1] - W) - u[2] / 2 + t * u[1]^2 / 3,
    ]
end
function coupled_iip(du, u, p, t, W)
    du[1] = sin(u[2] + W) - u[1] + cos(3t) * u[1] * u[2] / 2
    du[2] = cos(u[1] - W) - u[2] / 2 + t * u[1]^2 / 3
    return nothing
end

function gauss_rule(n)
    jacobi = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]))
    return (jacobi.values .+ 1) ./ 2, jacobi.vectors[1, :] .^ 2
end

const GAUSS_NODES, GAUSS_WEIGHTS = gauss_rule(6)

function decay_exact(grid, path)
    logu = zeros(length(path))
    for k in 1:(length(path) - 1)
        x = 5 * (path[k + 1] - path[k]) / 2
        mean_cos = cos(5 * (path[k] + path[k + 1]) / 2) * (iszero(x) ? one(x) : sin(x) / x)
        logu[k + 1] = logu[k] - (grid[k + 1] - grid[k]) * mean_cos
    end
    return exp.(logu)
end

function rk4_reference(rhs, u0, grid, path; substeps = 8)
    u = zeros(length(grid))
    u[1] = u0
    for k in 1:(length(grid) - 1)
        h = (grid[k + 1] - grid[k]) / substeps
        dw = (path[k + 1] - path[k]) / substeps
        x = u[k]
        for s in 0:(substeps - 1)
            t = grid[k] + s * h
            w = path[k] + s * dw
            r1 = rhs(x, nothing, t, w)
            r2 = rhs(x + h / 2 * r1, nothing, t + h / 2, w + dw / 2)
            r3 = rhs(x + h / 2 * r2, nothing, t + h / 2, w + dw / 2)
            r4 = rhs(x + h * r3, nothing, t + h, w + dw)
            x += h / 6 * (r1 + 2r2 + 2r3 + r4)
        end
        u[k + 1] = x
    end
    return u
end

function grid_solution(rhs, u0, alg, grid, path, n; inplace = false, p = nothing)
    tspan = (grid[1], grid[end])
    prob = if inplace
        RODEProblem(rhs, copy(u0), tspan, p, noise = NoiseGrid(grid, path))
    else
        RODEProblem{false}(rhs, u0, tspan, p, noise = NoiseGrid(grid, path))
    end
    return solve(prob, alg, dt = (grid[end] - grid[1]) / n, adaptive = false)
end

function grid_error(rhs, u0, alg, grid, path, reference, n)
    sol = grid_solution(rhs, u0, alg, grid, path, n)
    stride = (length(grid) - 1) ÷ n
    return maximum(abs(sol.u[k + 1] - reference[k * stride + 1]) for k in 0:n)
end

function strong_errors(algs, rhs, grid, paths, references, step_counts)
    errors = [
        grid_error(rhs, 1.0, alg, grid, paths[m], references[m], n)
            for alg in algs, m in eachindex(paths), n in step_counts
    ]
    return dropdims(sqrt.(mean(errors .^ 2, dims = 2)), dims = 2)
end

function coarse_wiener_path(rng)
    return [0.0; cumsum(sqrt(TEND / COARSE_POINTS) .* randn(rng, COARSE_POINTS))]
end

function wiener_study(algs, rhs, step_counts; seed = 20261011, paths = 64)
    rng = MersenneTwister(seed)
    samples = [coarse_wiener_path(rng) for _ in 1:paths]
    references = if rhs === decay_oop
        [decay_exact(COARSE_GRID, path) for path in samples]
    else
        [rk4_reference(rhs, 1.0, COARSE_GRID, path) for path in samples]
    end
    strong = strong_errors(algs, rhs, COARSE_GRID, samples, references, step_counts)
    return strong, [fitted_slope(step_counts, strong[a, :]) for a in eachindex(algs)]
end

function fbm_factor(hurst)
    t = (1:FBM_POINTS) ./ FBM_POINTS
    covariance = [
        (t[i]^(2hurst) + t[j]^(2hurst) - abs(t[i] - t[j])^(2hurst)) / 2
            for i in 1:FBM_POINTS, j in 1:FBM_POINTS
    ]
    return cholesky(Symmetric(covariance)).L
end

function fbm_study(algs, factor; seed = 20261011, paths = 64)
    rng = MersenneTwister(seed)
    samples = [[0.0; factor * randn(rng, FBM_POINTS)] for _ in 1:paths]
    references = [rk4_reference(sine_oop, 1.0, FBM_GRID, path) for path in samples]
    strong = strong_errors(algs, sine_oop, FBM_GRID, samples, references, FBM_STEP_COUNTS)
    return strong, [fitted_slope(FBM_STEP_COUNTS, strong[a, :]) for a in eachindex(algs)]
end

function path_value(grid, path, s)
    k = clamp(searchsortedlast(grid, s, rev = grid[2] < grid[1]), 1, length(grid) - 1)
    return path[k] + (path[k + 1] - path[k]) * (s - grid[k]) / (grid[k + 1] - grid[k])
end

function step_knots(grid, path, t, dt)
    lo, hi = minmax(t, t + dt)
    ts = [t; sort([s for s in grid if lo < s < hi], rev = dt < 0); t + dt]
    return ts, [path_value(grid, path, s) - path_value(grid, path, t) for s in ts]
end

function quadrature_integral(index, ts, vs, s = ts[end], depth = 1)
    depth > length(index) && return one(s)
    acc = zero(s)
    for k in 1:(length(ts) - 1)
        abs(s - ts[1]) <= abs(ts[k] - ts[1]) && break
        upper = abs(s - ts[1]) < abs(ts[k + 1] - ts[1]) ? s : ts[k + 1]
        for (x, weight) in zip(GAUSS_NODES, GAUSS_WEIGHTS)
            r = ts[k] + (upper - ts[k]) * x
            v = vs[k] + (vs[k + 1] - vs[k]) * (r - ts[k]) / (ts[k + 1] - ts[k])
            acc += (upper - ts[k]) * weight * v^index[depth] *
                quadrature_integral(index, ts, vs, r, depth + 1)
        end
    end
    return acc
end

function affine_coefficient(index, x, t, w0, A, r)
    taylor(q, k) = sum(
        binomial(i, k) * q[i + 1] * w0^(i - k) for i in k:(length(q) - 1); init = zero(q[1])
    )
    forcing(k, d) = sum(
        prod((j - d + 1):j; init = 1) * t^(j - d) * taylor(r[j + 1], k)
            for j in d:(length(r) - 1); init = zero(r[1][1])
    )
    c(k) = taylor(A, k) * x + forcing(k, 0)
    F(k) = c(k)
    F(k, b) = taylor(A, k) * c(b) + (b == 0) * forcing(k, 1)
    function F(k, a, b)
        return taylor(A, k) * taylor(A, a) * c(b) +
            (b == 0) * (taylor(A, k) * forcing(a, 1) + (a == 0) * forcing(k, 2))
    end
    return F(index...)
end

@testset "RandomTaylor index sets are those of the book" begin
    indices(order, holder) = sort(RODE.random_taylor_indices(order, holder))
    @test indices(3 // 1, 1 // 2) == sort(BOOK_INDICES_30)
    @test indices(3 // 2, 1 // 2) == [[0], [0, 0], [1], [2]]
    @test indices(1 // 2, 1 // 2) == [[0]]
    @test indices(3 // 5, 3 // 10) == [[0], [1]]
    @test indices(8 // 5, 3 // 10) ==
        sort([[0], [1], [2], [3], [4], [5], [0, 0], [1, 0], [0, 1]])
    @test indices(21 // 10, 3 // 10) == sort(
        [
            [0], [1], [2], [3], [4], [5], [6], [0, 0], [1, 0], [0, 1], [1, 1], [2, 0],
            [0, 2], [3, 0], [0, 3], [2, 1], [1, 2], [0, 0, 0],
        ]
    )
    for (order, holder) in ((1 // 2, 1 // 2), (5 // 2, 1 // 2), (3, 1), (3, 3 // 10), (2, 3 // 4))
        expected = [
            collect(index) for i in 1:4 for index in Iterators.product(ntuple(_ -> 0:10, i)...)
                if i + holder * sum(index) < order + 1
        ]
        @test indices(order, holder) == sort(expected)
    end
    alg = RandomTaylor(order = 1.6, holder = 0.3)
    @test StochasticDiffEq.alg_order(alg) == 8 // 5
    @test alg.holder == 3 // 10
    @test StochasticDiffEq.alg_order(RandomTaylor()) == 3 // 2
    @test RandomTaylor().holder == 1 // 2
end

@testset "RandomTaylor rejects what it does not support" begin
    @test_throws ArgumentError RandomTaylor(order = 0)
    @test_throws ArgumentError RandomTaylor(order = -1 // 2)
    @test_throws ArgumentError RandomTaylor(order = 7 // 2)
    @test_throws ArgumentError RandomTaylor(order = 3.01)
    @test_throws ArgumentError RandomTaylor(holder = 0)
    @test_throws ArgumentError RandomTaylor(holder = 1.5)
    @test_throws ArgumentError RandomTaylor(order = 2, holder = -0.3)
    @test RandomTaylor(order = 3, holder = 1) isa RandomTaylor
    prob = RODEProblem{false}(decay_oop, 1.0, (0.0, TEND))
    @test_throws "not compatible with the chosen noise type" solve(
        prob, RandomTaylor(), dt = TEND / 16, adaptive = false
    )
    vector_noise = NoiseGrid(COARSE_GRID, [[w, w] for w in COARSE_GRID])
    prob = RODEProblem{false}(decay_oop, 1.0, (0.0, TEND), noise = vector_noise)
    @test_throws "not compatible with the chosen noise type" solve(
        prob, RandomTaylor(), dt = TEND / 16, adaptive = false
    )
end

@testset "RandomTaylor right-hand side evaluations per step" begin
    calls = Ref(0)
    counted(u, p, t, W) = (calls[] += 1; -u * cos(W))
    for (order, holder, expected) in (
            (1 // 2, 1 // 2, 1), (3 // 2, 1 // 2, 5), (2, 1 // 2, 13), (5 // 2, 1 // 2, 23),
            (3, 1 // 2, 51), (8 // 5, 3 // 10, 15), (3 // 2, 3 // 4, 5), (3, 1, 15),
            (3, 3 // 10, 173),
        )
        calls[] = 0
        alg = RandomTaylor(order = order, holder = holder)
        grid_solution(counted, 1.0, alg, UNEVEN_GRID, UNEVEN_PATH, 8)
        @test calls[] == 8 * expected
    end
end

@testset "RandomTaylor step integrals match quadrature on a misaligned grid" begin
    indices = RODE.random_taylor_indices(3 // 1, 1 // 2)
    tab = RODE.RandomTaylorTableau(Float64, 3 // 1, 1 // 2)
    steps = ((0.0, 0.25), (0.05, 0.31), (0.31, 0.004), (0.5, 0.5), (0.13, 0.11))
    for (grid, path) in UNEVEN_GRIDS
        for (start, len) in steps
            t, dt = grid[1] < grid[2] ? (start, len) : (start + len, -len)
            ts, vs = step_knots(grid, path, t, dt)
            w0 = path_value(grid, path, t)
            W = (t = grid, W = path, dW = vs[end], curW = w0)
            integrals = RODE.iterated_integrals!(tab, W, t, dt, w0)
            for (k, index) in enumerate(indices)
                reference = quadrature_integral(index, ts, vs)
                atol = 1.0e-13 * abs(dt)^length(index)
                @test integrals[k] ≈ reference rtol = 1.0e-10 atol = atol
            end
        end
    end
end

@testset "RandomTaylor integrates a polynomial in the noise exactly" begin
    for (order, holder, degree) in (
            (3 // 2, 1 // 2, 2), (5 // 2, 1 // 2, 4), (3, 1 // 2, 5), (8 // 5, 3 // 10, 5),
        )
        coefficients = [(-1)^k * (k + 2) / (k + 1) for k in 0:degree]
        alg = RandomTaylor(order = order, holder = holder)
        for (grid, path) in UNEVEN_GRIDS
            exact = sum(
                (grid[k + 1] - grid[k]) * GAUSS_WEIGHTS[j] *
                    evalpoly(path[k] + (path[k + 1] - path[k]) * GAUSS_NODES[j], coefficients)
                    for k in 1:(length(grid) - 1), j in eachindex(GAUSS_NODES)
            )
            sol = grid_solution(polynomial_oop, 0.0, alg, grid, path, 7; p = coefficients)
            @test sol.u[end] ≈ exact rtol = 1.0e-11
        end
    end
end

@testset "RandomTaylor reproduces the Taylor step for an affine polynomial problem" begin
    A = [[-0.7 0.4; 0.3 -0.5], [0.2 -0.6; 0.5 0.1], [-0.3 0.2; 0.1 0.4]]
    r = [
        [[0.5, -0.2], [0.3, 0.7], [-0.4, 0.2], [0.6, -0.5]],
        [[-0.3, 0.4], [0.8, -0.1], [0.2, 0.5]],
        [[0.4, 0.3], [-0.6, 0.2], [0.1, -0.7]],
    ]
    forcing(t, W) = evalpoly(t, [evalpoly(W, q) for q in r])
    affine_oop(u, p, t, W) = evalpoly(W, A) * u + forcing(t, W)
    affine_iip(du, u, p, t, W) = (du .= evalpoly(W, A) * u .+ forcing(t, W); nothing)
    u0 = [0.8, -0.6]
    for (holder, indices) in (
            (1 // 2, BOOK_INDICES_30), (3 // 10, RODE.random_taylor_indices(3 // 1, 3 // 10)),
        )
        alg = RandomTaylor(order = 3, holder = holder)
        for (grid, path, dt) in (
                (UNEVEN_GRID, UNEVEN_PATH, 0.21), (UNEVEN_GRID, UNEVEN_PATH, 0.01),
                (reverse(UNEVEN_GRID), reverse(UNEVEN_PATH), -0.21),
            )
            t = grid[1]
            ts, vs = step_knots(grid, path, t, dt)
            expected = u0 + sum(
                affine_coefficient(index, u0, t, path[1], A, r) *
                    quadrature_integral(index, ts, vs) for index in indices
            )
            noise = NoiseGrid(grid, path)
            oop = solve(
                RODEProblem{false}(affine_oop, u0, (t, t + dt), noise = noise), alg,
                dt = dt, adaptive = false
            )
            iip = solve(
                RODEProblem(affine_iip, copy(u0), (t, t + dt), noise = noise), alg,
                dt = dt, adaptive = false
            )
            @test only(diff(oop.t)) ≈ dt
            @test oop.u[end] ≈ expected rtol = 1.0e-11
            @test iip.u[end] ≈ expected rtol = 1.0e-11
        end
    end
end

@testset "RandomTaylor strong order on Wiener paths" begin
    algs = (
        RandomTaylor(order = 3 // 2), RandomTaylor(order = 2), RandomTaylor(order = 5 // 2),
        RandomTaylor15(),
    )
    strong, orders = wiener_study(algs, decay_oop, COARSE_STEP_COUNTS; paths = 256)
    @test 1.6 < orders[1] < 2.2
    @test 1.8 < orders[2] < 2.4
    @test 2.4 < orders[3] < 3.4
    @test strong[2, end] < 0.8 * strong[1, end]
    @test strong[3, end] < strong[2, end] / 8
    @test abs(orders[1] - orders[4]) < 0.2
    @test all(1 / 3 .< strong[1, :] ./ strong[4, :] .< 3)
end

@testset "RandomTaylor with a nonlinear time-dependent right-hand side" begin
    algs = (RandomTaylor(order = 5 // 2), RandomTaylor(order = 3 // 2))
    strong, orders = wiener_study(algs, timed_oop, [2^k for k in 2:7]; paths = 16)
    @test 2.7 < orders[1] < 3.3
    @test 1.8 < orders[2] < 2.3
    @test strong[1, end] < strong[2, end] / 20
end

@testset "RandomTaylor on fractional Brownian paths" begin
    algs = (
        RandomTaylor(order = 1, holder = 0.3), RandomTaylor(order = 1.6, holder = 0.3),
        RandomEM(), RandomTaylor(order = 3, holder = 0.3),
    )
    strong, orders = fbm_study(algs, fbm_factor(0.3))
    @test 0.95 < orders[1] < 1.35
    @test 1.4 < orders[2] < 1.95
    @test orders[3] < orders[2] - 0.5
    @test strong[2, end] < strong[1, end] / 5
    @test strong[2, end] < strong[3, end] / 50
    @test 2.8 < orders[4] < 4.1
    @test strong[4, end] < strong[2, end] / 100
    algs = (RandomTaylor(order = 1.5, holder = 0.75), RandomEM())
    strong, orders = fbm_study(algs, fbm_factor(0.75))
    @test 1.4 < orders[1] < 1.9
    @test strong[1, end] < strong[2, end] / 8
end

@testset "RandomTaylor without noise in the right-hand side" begin
    path = zeros(COARSE_POINTS + 1)
    exact = 1 ./ (1 .+ COARSE_GRID .^ 2)
    step_counts = [2^k for k in 3:8]
    for (order, holder, classical) in (
            (1 // 2, 1 // 2, 1), (3 // 2, 1 // 2, 2), (5 // 2, 1 // 2, 3), (3, 1 // 2, 3),
            (8 // 5, 3 // 10, 2),
        )
        alg = RandomTaylor(order = order, holder = holder)
        errors = [
            grid_error(quadratic_oop, 1.0, alg, COARSE_GRID, path, exact, n) for n in step_counts
        ]
        @test abs(fitted_slope(step_counts, errors) - classical) < 0.2
    end
end

@testset "RandomTaylor on a smooth path with holder = 1" begin
    path = [cos(3s) / 2 for s in COARSE_GRID]
    exact = decay_exact(COARSE_GRID, path)
    for order in 1:3
        alg = RandomTaylor(order = order, holder = 1)
        errors = [
            grid_error(decay_oop, 1.0, alg, COARSE_GRID, path, exact, n)
                for n in COARSE_STEP_COUNTS
        ]
        @test abs(fitted_slope(COARSE_STEP_COUNTS, errors) - order) < 0.15
    end
end

@testset "RandomTaylor in-place matches out-of-place" begin
    pairs = ((3 // 2, 1 // 2), (5 // 2, 1 // 2), (3, 1 // 2), (8 // 5, 3 // 10))
    for (grid, path) in DETERMINISTIC_GRIDS, (order, holder) in pairs, n in (7, 64)
        alg = RandomTaylor(order = order, holder = holder)
        scalar = grid_solution(timed_oop, 0.8, alg, grid, path, n).u
        inplace = grid_solution(timed_iip, [0.8], alg, grid, path, n; inplace = true).u
        @test first.(inplace) ≈ scalar rtol = 1.0e-9
        u0 = [0.8, -0.4]
        outofplace = grid_solution(coupled_oop, u0, alg, grid, path, n).u
        inplace = grid_solution(coupled_iip, u0, alg, grid, path, n; inplace = true).u
        @test all(isapprox(a, b, rtol = 1.0e-9) for (a, b) in zip(outofplace, inplace))
    end
end

@testset "RandomTaylor integrates backwards in time" begin
    grid, path = DETERMINISTIC_GRIDS[2]
    exact = decay_exact(grid, path)
    for (order, bound) in ((3 // 2, 3.6e-3), (5 // 2, 5.3e-5), (3, 5.2e-5))
        alg = RandomTaylor(order = order)
        @test grid_error(decay_oop, 1.0, alg, grid, path, exact, 256) < bound
    end
end

@testset "RandomTaylor of order 0.5 is the Euler step" begin
    path = coarse_wiener_path(MersenneTwister(20261011))
    u0 = [0.8, -0.4]
    for n in (16, 100, COARSE_POINTS)
        euler, taylor = (
            grid_solution(timed_oop, 0.8, alg, COARSE_GRID, path, n).u
                for alg in (RandomEM(), RandomTaylor(order = 1 // 2))
        )
        @test taylor ≈ euler rtol = 1.0e-12
        euler, taylor = (
            grid_solution(coupled_iip, u0, alg, COARSE_GRID, path, n; inplace = true).u
                for alg in (RandomEM(), RandomTaylor(order = 1 // 2))
        )
        @test taylor ≈ euler rtol = 1.0e-12
    end
end
