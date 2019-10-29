using PowerModels
using .PP2PM
import JuMP

function increase_rate_a(pm)
    for (i, branch) in pm["branch"]
            branch["rate_a"] = 1.01 * branch["rate_a"]
    end
    return pm
end

function run_powermodels(json_path)
    # function to run optimal transmission switching (OTS) optimization from powermodels.jl
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])

    solver = PP2PM.get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    result = run_ots(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    i = 0
    max_iteration = 6
    while i < max_iteration && !(result["termination_status"] == JuMP.MOI.LOCALLY_SOLVED || result["termination_status"] == JuMP.MOI.OPTIMAL)
        print("\n\ntrying to increase rate A by 1%: ", i, "/", max_iteration, "\n\n")
        pm = increase_rate_a(pm)
        result = run_ots(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
        i = i + 1
        print_summary(result)
    end

    return result
end


