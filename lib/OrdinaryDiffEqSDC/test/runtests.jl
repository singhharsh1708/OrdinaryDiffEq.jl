using SafeTestsets

const TEST_GROUP = get(ENV, "ODEDIFFEQ_TEST_GROUP", "ALL")

if TEST_GROUP == "Core" || TEST_GROUP == "ALL"
    @time @safetestset "SDC Tableau Tests" include("sdc_tableau_tests.jl")
    @time @safetestset "SDC Convergence Tests" include("sdc_convergence_tests.jl")
end
