module metrics

using ..operators

export tracedistance


function tracedistance(rho::Operator, sigma::Operator)
    delta = (rho - sigma).data
    @assert size(delta, 1) == size(delta, 2)
    for i=1:size(delta,1)
        delta[i,i] = real(delta[i,i])
    end
    s = eigvals(Hermitian(delta))
    return 0.5*sum(abs(s))
end

end # module