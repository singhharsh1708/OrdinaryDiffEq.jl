# One SDC step.
#
# The collocation problem on the step is
#
#     u_m = u_n + Δt Σ_j Q[m,j] f(u_j, t_n + τ_j Δt),      m = 1 … M
#
# and SDC solves it by the preconditioned iteration
#
#     u^{k+1} - Δt QΔ f^{k+1} = u_n + Δt (Q - QΔ) f^k,
#
# which, because `QΔ` is lower triangular, splits into `M` successive solves
#
#     u_m^{k+1} = rhs_m + Δt QΔ[m,m] f(u_m^{k+1}, t_n + τ_m Δt),
#     rhs_m = u_n + Δt Σ_{j=1}^{M} (Q - QΔ)[m,j] f_j^k
#                 + Δt Σ_{j<m}     QΔ[m,j]       f_j^{k+1}.
#
# The cache stores `z_m = Δt f_m` rather than `f_m` or `u_m`: that is exactly the
# variable `nlsolve!` solves for, so no node value is ever materialised and no
# division by `dt` is needed.
#
# The `Σ_{j<m}` term is the only node-to-node coupling in the sweep. It is empty
# for a diagonal `QΔ`, at which point the `m` loop becomes independent across
# nodes — this is what parallel-across-the-nodes SDC (Speck 2018) exploits, and
# why the loop body below is written to touch only `z` (sweep `k`), `znew[1:m-1]`
# and node-local scratch.

function initialize!(integrator, cache::Union{SDCCache, SDCConstantCache}) end

@muladd function perform_step!(integrator, cache::SDCCache, repeat_step = false)
    (; t, dt, uprev, u, f, p) = integrator
    (; tmp, ubuf, k, nlsolver, tab, solver_index) = cache
    (; nodes, weights, Q, QΔ) = tab
    alg = unwrap_alg(integrator, true)
    M = length(nodes)
    stats = integrator.stats

    # k = 0: copy initialisation, u⁰_m = u_n at every node.
    for m in 1:M
        f(k, uprev, p, t + nodes[m] * dt)
        @.. broadcast = false cache.z[m] = dt * k
    end
    OrdinaryDiffEqCore.increment_nf!(stats, M)
    @.. broadcast = false ubuf = uprev

    zk, zk1 = cache.z, cache.znew
    for _ in 1:(alg.num_sweeps)
        for m in 1:M
            @.. broadcast = false tmp = uprev
            for j in 1:M
                coeff = Q[m, j] - QΔ[m, j]
                iszero(coeff) && continue
                @.. broadcast = false tmp = tmp + coeff * zk[j]
            end
            for j in 1:(m - 1)
                coeff = QΔ[m, j]
                iszero(coeff) && continue
                @.. broadcast = false tmp = tmp + coeff * zk1[j]
            end
            index = solver_index[m]
            if iszero(index)
                # Explicit node: QΔ[m,m] = 0, so u_m is the right-hand side.
                @.. broadcast = false ubuf = tmp
                f(k, ubuf, p, t + nodes[m] * dt)
                OrdinaryDiffEqCore.increment_nf!(stats, 1)
                @.. broadcast = false zk1[m] = dt * k
            else
                nls = nlsolver[index]
                @.. broadcast = false nls.tmp = tmp
                @.. broadcast = false nls.z = zk[m]
                nls.γ = QΔ[m, m]
                nls.c = nodes[m]
                markfirststage!(nls)
                znode = nlsolve!(nls, integrator, cache, repeat_step)
                nlsolvefail(nls) && return
                @.. broadcast = false zk1[m] = znode
                @.. broadcast = false ubuf = tmp + QΔ[m, m] * znode
            end
        end
        zk, zk1 = zk1, zk
    end

    if alg.step_update === :quadrature
        @.. broadcast = false u = uprev
        for m in 1:M
            iszero(weights[m]) && continue
            @.. broadcast = false u = u + weights[m] * zk[m]
        end
    else
        @.. broadcast = false u = ubuf
    end
    return nothing
end

@muladd function perform_step!(integrator, cache::SDCConstantCache, repeat_step = false)
    (; t, dt, uprev, f, p) = integrator
    (; nlsolver, tab, solver_index) = cache
    (; nodes, weights, Q, QΔ) = tab
    alg = unwrap_alg(integrator, true)
    M = length(nodes)
    stats = integrator.stats

    zk = [dt * f(uprev, p, t + nodes[m] * dt) for m in 1:M]
    zk1 = similar(zk)
    OrdinaryDiffEqCore.increment_nf!(stats, M)
    ulast = uprev

    for _ in 1:(alg.num_sweeps)
        for m in 1:M
            tmp = uprev
            for j in 1:M
                coeff = Q[m, j] - QΔ[m, j]
                iszero(coeff) && continue
                tmp = @.. broadcast = false tmp + coeff * zk[j]
            end
            for j in 1:(m - 1)
                coeff = QΔ[m, j]
                iszero(coeff) && continue
                tmp = @.. broadcast = false tmp + coeff * zk1[j]
            end
            index = solver_index[m]
            if iszero(index)
                ulast = tmp
                zk1[m] = dt * f(ulast, p, t + nodes[m] * dt)
                OrdinaryDiffEqCore.increment_nf!(stats, 1)
            else
                nls = nlsolver[index]
                nls.tmp = tmp
                nls.z = zk[m]
                nls.γ = QΔ[m, m]
                nls.c = nodes[m]
                markfirststage!(nls)
                znode = nlsolve!(nls, integrator, cache, repeat_step)
                nlsolvefail(nls) && return
                zk1[m] = znode
                ulast = @.. broadcast = false tmp + QΔ[m, m] * znode
            end
        end
        zk, zk1 = zk1, zk
    end

    integrator.u = if alg.step_update === :quadrature
        unew = uprev
        for m in 1:M
            iszero(weights[m]) && continue
            unew = @.. broadcast = false unew + weights[m] * zk[m]
        end
        unew
    else
        ulast
    end
    return nothing
end
