@inline function add_segment!(tab::RandomTaylorTableau, h, v0, v1)
    (; depth, prefix, suffix, singles, recip, coeffs, moments, integrals) = tab
    dv = v1 - v0
    pw = one(v0)
    acc = one(v0)
    @inbounds for k in 2:singles
        pw *= v0
        acc = v1 * acc + pw
        moments[k] = recip[k] * acc
    end
    @inbounds for i in (singles + 1):length(moments)
        c = coeffs[i]
        pw = one(v0)
        acc = c[1]
        for j in 2:length(c)
            pw *= dv
            acc = v0 * acc + c[j] * pw
        end
        moments[i] = acc
    end
    @inbounds for i in length(integrals):-1:(singles + 1)
        p = prefix[i]
        s = suffix[i]
        if depth[i] == 2
            integrals[i] += h * (moments[p] * integrals[s] + h * moments[i])
        else
            integrals[i] += h * (
                moments[prefix[p]] * integrals[s] +
                    h * (moments[p] * integrals[suffix[s]] + h * moments[i])
            )
        end
    end
    @inbounds for k in 1:singles
        integrals[k] += h * moments[k]
    end
    return nothing
end

function iterated_integrals!(tab::RandomTaylorTableau, W, t, dt, w0)
    tend = t + dt
    tdir = dt < zero(dt) ? -one(dt) : one(dt)
    rev = length(W.t) > 1 && W.t[2] < W.t[1]
    ilo = searchsortedfirst(W.t, t, rev = rev)
    ihi = searchsortedlast(W.t, tend, rev = rev)
    fill!(tab.integrals, false)
    tprev = t
    vprev = zero(w0)
    @inbounds for i in ilo:ihi
        ti = W.t[i]
        tdir * (ti - tprev) <= zero(dt) && continue
        vi = W.W[i] - w0
        add_segment!(tab, ti - tprev, vprev, vi)
        tprev = ti
        vprev = vi
    end
    add_segment!(tab, tend - tprev, vprev, W.dW)
    return tab.integrals
end

function stage_weights!(tab::RandomTaylorTableau, W, t, dt, w0, δ)
    (; lead, depth, integrals, root, children, grandchildren) = tab
    iterated_integrals!(tab, W, t, dt, w0)
    for i in eachindex(integrals)
        integrals[i] /= δ^lead[i] * dt^(depth[i] - 1)
    end
    mul!(root.weights, root.map, integrals)
    for child in children
        mul!(child.weights, child.map, integrals)
    end
    for row in grandchildren, grand in row
        mul!(grand.weights, grand.map, integrals)
    end
    return nothing
end

forward_weight(m) = m == 1 ? 4 : -1

function shifted(uprev, rtmp, center, weights, scale)
    n = (length(weights) - 1) ÷ 2
    tmp = uprev
    for j in (-n):n
        tmp = tmp .+ (scale * weights[j + n + 1]) .* rtmp[center + j]
    end
    return tmp
end

function shifted!(tmp, uprev, rtmp, center, weights, scale)
    n = (length(weights) - 1) ÷ 2
    @.. tmp = uprev
    for j in (-n):n
        c = scale * weights[j + n + 1]
        r = rtmp[center + j]
        @.. tmp = tmp + c * r
    end
    return nothing
end

@muladd function perform_step!(integrator, cache::RandomTaylorConstantCache)
    (; t, dt, uprev, W, p) = integrator
    (; tab, rtmp1, rtmp2) = cache
    (; directions, root, children, grandchildren) = tab
    w0 = W.curW
    δ = abs(dt)^tab.holder
    η = dt / 2
    stage_weights!(tab, W, t, dt, w0, δ)
    n = root.halfwidth
    u = uprev
    for j in (-n):n
        ftmp = integrator.f(uprev, p, t, w0 + j * δ)
        rtmp1[j + n + 1] = ftmp
        u = u .+ root.weights[j + n + 1] .* ftmp
    end
    for b in eachindex(children)
        child = children[b]
        nb = child.halfwidth
        for m in 1:2
            tmp = shifted(uprev, rtmp1, n + 1, directions[b], m * η / δ^(b - 1))
            t1 = b == 1 ? t + m * η : t
            for j in (-nb):nb
                ftmp = integrator.f(tmp, p, t1, w0 + j * δ)
                rtmp2[j + nb + 1] = ftmp
                u = u .+ (forward_weight(m) * child.weights[j + nb + 1]) .* ftmp
            end
            for a in eachindex(grandchildren[b])
                grand = grandchildren[b][a]
                ng = grand.halfwidth
                for l in 1:2
                    tmp1 = shifted(tmp, rtmp2, nb + 1, directions[a], l * η / δ^(a - 1))
                    t2 = a == 1 ? t1 + l * η : t1
                    c = forward_weight(m) * forward_weight(l)
                    for j in (-ng):ng
                        ftmp = integrator.f(tmp1, p, t2, w0 + j * δ)
                        u = u .+ (c * grand.weights[j + ng + 1]) .* ftmp
                    end
                end
            end
        end
    end
    integrator.u = u
end

@muladd function perform_step!(integrator, cache::RandomTaylorCache)
    (; tab, tmp, tmp1, rtmp, rtmp1, rtmp2) = cache
    (; t, dt, uprev, u, W, p) = integrator
    (; directions, root, children, grandchildren) = tab
    w0 = W.curW
    δ = abs(dt)^tab.holder
    η = dt / 2
    stage_weights!(tab, W, t, dt, w0, δ)
    n = root.halfwidth
    @.. u = uprev
    for j in (-n):n
        r = rtmp1[j + n + 1]
        integrator.f(r, uprev, p, t, w0 + j * δ)
        c = root.weights[j + n + 1]
        @.. u = u + c * r
    end
    for b in eachindex(children)
        child = children[b]
        nb = child.halfwidth
        for m in 1:2
            shifted!(tmp, uprev, rtmp1, n + 1, directions[b], m * η / δ^(b - 1))
            t1 = b == 1 ? t + m * η : t
            for j in (-nb):nb
                r = rtmp2[j + nb + 1]
                integrator.f(r, tmp, p, t1, w0 + j * δ)
                c = forward_weight(m) * child.weights[j + nb + 1]
                @.. u = u + c * r
            end
            for a in eachindex(grandchildren[b])
                grand = grandchildren[b][a]
                ng = grand.halfwidth
                for l in 1:2
                    shifted!(tmp1, tmp, rtmp2, nb + 1, directions[a], l * η / δ^(a - 1))
                    t2 = a == 1 ? t1 + l * η : t1
                    for j in (-ng):ng
                        integrator.f(rtmp, tmp1, p, t2, w0 + j * δ)
                        c = forward_weight(m) * forward_weight(l) * grand.weights[j + ng + 1]
                        @.. u = u + c * rtmp
                    end
                end
            end
        end
    end
end
