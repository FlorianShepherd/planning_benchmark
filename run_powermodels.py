import copy
import getopt
import os
import sys
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import pandapower as pp
from pandapower.converter import init_ne_line
from pandapower.converter.powermodels.from_pm import read_ots_results

pm_time_limits = {
    "pm_time_limit": 300.,
    "pm_nl_time_limit": 150.,
    "pm_mip_time_limit": 150.,
}


def check_opf_converged(net):
    return True if net["OPF_converged"] else False


def init_repl(net):
    max_idx = max(net["line"].index)
    net["line"] = pd.concat([net["line"]] * 2, ignore_index=True)
    net["line"].loc[max_idx + 1:, "in_service"] = False
    new_lines = net["line"].loc[max_idx + 1:]
    construction_cost = new_lines.length_km.values
    init_ne_line(net, new_lines.index, construction_costs=construction_cost)
    return net


def set_p_va(net, pm_model):
    results_available = False
    init_p_mw = copy.deepcopy(net.gen.loc[:, "p_mw"])
    init_va = copy.deepcopy(net.ext_grid.loc[:, "va_degree"].values)
    if len(net.res_bus):
        net.gen.loc[:, "p_mw"] = net.res_gen.loc[:, "p_mw"]
        if net.name != "simbench_mixed" and pm_model != "SOCWRPowerModel":
            net.ext_grid.loc[:, "va_degree"] = net.res_bus.loc[net.ext_grid.bus, "va_degree"].values
        results_available = True
    return results_available, init_p_mw, init_va


def reset_va_p(net, init_p_mw, init_va):
    net.gen.loc[:, "p_mw"] = init_p_mw
    net.ext_grid.loc[:, "va_degree"] = init_va


def assert_tnep_result(net, pm_model):
    n_built = sum(net["res_ne_line"].values)
    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True
    cost = net["line"].loc[lines_to_built, "length_km"].sum()
    # set changed generator setpoint
    results_available, init_p_mw, init_va = set_p_va(net, pm_model)

    # run a power flow calculation again and check if max_loading percent is still violated
    pp.rundcpp(net)
    max_line_loading_dc = max(net["res_line"].loc[:, "loading_percent"])
    pp.runpp(net)
    # check max line loading results
    worked = not np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])
    max_line_loading = max(net["res_line"].loc[:, "loading_percent"])
    v_min = net["res_bus"].loc[:, "vm_pu"].min()
    v_max = net["res_bus"].loc[:, "vm_pu"].max()

    # reset
    net["line"].loc[lines_to_built, "in_service"] = False
    if results_available:
        reset_va_p(net, init_p_mw, init_va)
    return worked, max_line_loading, max_line_loading_dc, v_min, v_max, n_built, list(lines_to_built), cost


def assert_ots_result(net, pm_model):
    backup_in_service_l = copy.deepcopy(net.line.loc[:, "in_service"].values)
    backup_in_service_t = copy.deepcopy(net.trafo.loc[:, "in_service"].values)

    closed_l = net["res_line"].loc[:, "in_service"].values.astype(bool)
    closed_t = net["res_trafo"].loc[:, "in_service"].values.astype(bool)
    lines_is = list(net["res_line"].loc[closed_l, "in_service"].index) + list(
        net["res_trafo"].loc[closed_t, "in_service"].index)
    n_closed = closed_l.sum() + closed_t.sum()

    # set closed switches
    net["line"].loc[closed_l, "in_service"] = True
    net["line"].loc[~closed_l, "in_service"] = False
    net["trafo"].loc[closed_t, "in_service"] = True
    net["trafo"].loc[~closed_t, "in_service"] = False
    # set changed generator setpoint
    results_available, init_p_mw, init_va = set_p_va(net, pm_model)

    pp.rundcpp(net)
    max_line_loading_dc = max(net["res_line"].loc[:, "loading_percent"])
    pp.runpp(net)
    # check max line loading results and if all buses are still in service
    worked = not np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])
    worked = worked and (len(net.bus) == len(net.res_bus.dropna()))
    max_line_loading = max(net["res_line"].loc[:, "loading_percent"])
    v_min = net["res_bus"].loc[:, "vm_pu"].min()
    v_max = net["res_bus"].loc[:, "vm_pu"].max()

    # reset
    if results_available:
        reset_va_p(net, init_p_mw, init_va)
    net["line"].loc[:, "in_service"] = backup_in_service_l
    net["trafo"].loc[:, "in_service"] = backup_in_service_t

    return worked, max_line_loading, max_line_loading_dc, v_min, v_max, n_closed, lines_is


def drop_als(net):
    if "additional_line" in net.line.columns:
        # drop additional lines
        net["line"] = net["line"].drop(index=net.line.loc[net.line.additional_line.values].index)
    return net


def init_tnep(net):
    new_lines = net.line.loc[net.line.loc[:, "additional_line"]]
    construction_cost = new_lines.length_km.values
    init_ne_line(net, new_lines.index, construction_costs=construction_cost)
    return net


def custom_ots(net, pm_model, pm_solver, with_ratea=True):
    julia_file = "ots_with_ratea.jl"
    with_ratea = False if pm_model == "ACPPowerModel" else True

    if pm_solver == "knitro":
        if with_ratea:
            pp.runpm(net, julia_file=julia_file, pm_model=pm_model, pm_solver="knitro", pm_time_limits=pm_time_limits,
                     pm_log_level=1)
            read_ots_results(net)
        else:
            pp.runpm_ots(net, pm_model=pm_model, pm_solver="knitro", pm_time_limits=pm_time_limits, pm_log_level=1)

    elif pm_solver == "cbc":
        if with_ratea:
            pp.runpm(net, julia_file=julia_file, pm_model=pm_model, pm_solver="cbc", pm_time_limits=pm_time_limits,
                     pm_log_level=1)
            read_ots_results(net)
        else:
            pp.runpm_ots(net, pm_model=pm_model, pm_solver="cbc", pm_time_limits=pm_time_limits, pm_log_level=1)
    elif pm_solver == "gurobi":
        if with_ratea:
            pp.runpm(net, julia_file=julia_file, pm_model=pm_model, pm_solver="juniper", pm_nl_solver="gurobi",
                     pm_time_limits=pm_time_limits, pm_log_level=1)
            read_ots_results(net)
        else:
            pp.runpm_ots(net, pm_model=pm_model, pm_solver="juniper", pm_nl_solver="gurobi",
                         pm_time_limits=pm_time_limits, pm_log_level=1)
    else:
        if with_ratea:
            pp.runpm(net, julia_file=julia_file, pm_model=pm_model, pm_solver="juniper", pm_nl_solver="ipopt",
                     pm_time_limits=pm_time_limits, pm_log_level=1)
            read_ots_results(net)
        else:
            pp.runpm_ots(net, pm_model=pm_model, pm_solver="juniper", pm_nl_solver="ipopt",
                         pm_time_limits=pm_time_limits, pm_log_level=1)


def load_from_disk(net_path, pm_model):
    ots = pp.from_json(net_path)
    repl = pp.from_json(net_path)
    tnep = pp.from_json(net_path)
    repl = drop_als(repl)
    ots = drop_als(ots)

    tnep = init_tnep(tnep)
    repl = init_repl(repl)

    return repl, tnep, ots


def restore_df(lc_path, net):
    # restores dataframe from files on disk
    df_names = os.listdir(lc_path)
    for file in df_names:
        element = file.replace(".json", "")
        file_path = os.path.join(lc_path, file)
        net[element] = pd.read_json(file_path)
    return net


def test_power_models(loadcases_path, net_path, grid, pm_model="DCPPowerModel", pm_solver="gurobi", kind="tnep,repl",
                      lc="scaled"):
    repl, tnep, ots = load_from_disk(net_path, pm_model)
    res_file = os.path.join(get_res_dir(pm_model, pm_solver, init=lc), grid + ".xlsx")
    # get loadcases
    loadcases = os.listdir(loadcases_path)
    df = pd.DataFrame(index=np.arange(len(loadcases)), columns=["lc", "ll_init", "vmin_init", "vmax_init",
                                                                "tnep_converged", "tnep_worked", "tnep_max_ll",
                                                                "tnep_max_ll_dc", "tnep_vmax", "tnep_vmin",
                                                                "tnep_built", "tnep_solution", "tnep_cost", "tnep_t",
                                                                "repl_converged", "repl_worked", "repl_max_ll",
                                                                "repl_max_ll_dc", "repl_vmax", "repl_vmin",
                                                                "repl_built", "repl_solution", "repl_cost", "repl_t",
                                                                "ots_converged", "ots_worked", "ots_max_ll",
                                                                "ots_max_ll_dc", "ots_vmax",
                                                                "ots_vmin", "ots_closed", "ots_solution", "ots_t"])

    for i, lc in tqdm(enumerate(loadcases), desc="getting limits for loadcases"):
        # get initial values prior to any optimization
        df.at[i, "lc"] = lc
        lc_path = os.path.join(loadcases_path, lc)
        repl = restore_df(lc_path, repl)
        try:
            pp.runpp(repl)
            df.at[i, "ll_init"] = repl.res_line.loading_percent.max()
            df.at[i, "vmin_init"] = repl.res_bus.vm_pu.min()
            df.at[i, "vmax_init"] = repl.res_bus.vm_pu.max()
        except pp.LoadflowNotConverged:
            pass

    df.to_excel(res_file)

    repl, tnep, ots = load_from_disk(net_path, pm_model)
    # backup init nets
    repl_c, tnep_c, ots_c = copy.deepcopy(repl), copy.deepcopy(tnep), copy.deepcopy(ots)

    for i, lc in tqdm(enumerate(loadcases), desc="testing powermodels"):
        # run the optimization
        lc_path = os.path.join(loadcases_path, lc)
        repl = restore_df(lc_path, repl)
        tnep = restore_df(lc_path, tnep)
        ots = restore_df(lc_path, ots)

        if "repl" in kind:
            try:
                t0 = time()
                # change this part if you like to run your own optimization
                pp.runpm_tnep(repl, pm_model=pm_model, pm_solver=pm_solver, pm_time_limits=pm_time_limits,
                              pm_log_level=1)
                df.at[i, "repl_t"] = time() - t0
            except Exception as e:
                repl = copy.deepcopy(repl_c)
                print(e)

            df.at[i, "repl_converged"] = check_opf_converged(repl)
            try:
                df.loc[i, ["repl_worked", "repl_max_ll", "repl_max_ll_dc", "repl_vmin", "repl_vmax",
                           "repl_built", "repl_solution", "repl_cost"]] = assert_tnep_result(
                    repl, pm_model)
            except Exception as e:
                print(e)
                repl = copy.deepcopy(repl_c)
            df.to_excel(res_file)

        if "tnep" in kind:
            try:
                t0 = time()
                # change this part if you like to run your own optimization
                pp.runpm_tnep(tnep, pm_model=pm_model, pm_solver=pm_solver, pm_time_limits=pm_time_limits,
                              pm_log_level=1)
                df.at[i, "tnep_t"] = time() - t0
            except Exception as e:
                print(e)
                tnep = copy.deepcopy(tnep_c)
            df.at[i, "tnep_converged"] = check_opf_converged(tnep)

            try:
                df.loc[i, ["tnep_worked", "tnep_max_ll", "tnep_max_ll_dc", "tnep_vmin", "tnep_vmax",
                           "tnep_built", "tnep_solution", "tnep_cost"]] = assert_tnep_result(
                    tnep, pm_model)
            except Exception as e:
                tnep = copy.deepcopy(tnep_c)
                print(e)
            df.to_excel(res_file)

        if "ots" in kind:
            try:
                t0 = time()
                # change this part if you like to run your own optimization
                custom_ots(ots, pm_model=pm_model, pm_solver=pm_solver)
                df.at[i, "ots_t"] = time() - t0
            except Exception as e:
                ots = copy.deepcopy(ots_c)
                print(e)

            try:
                df.at[i, "ots_converged"] = check_opf_converged(ots)
                df.loc[i, ["ots_worked", "ots_max_ll", "ots_max_ll_dc", "ots_vmin", "ots_vmax",
                           "ots_closed", "ots_solution"]] = assert_ots_result(ots, pm_model)
            except Exception as e:
                print(e)
                ots = copy.deepcopy(ots_c)

            df.to_excel(res_file)
    print(f"Optimization finished. Results are stored to {res_file}")
    return df


def get_res_dir(pm_model, pm_solver, init):
    return os.path.join(os.getcwd(), "results", init, pm_model + "_" + pm_solver)


def get_grids(grid_name=None):
    if grid_name is None:
        grids = ["brigande", "simbench_mixed", "simbench_urban", "rts"]
    else:
        grids = [grid_name]
    return grids


def start_power_models(pm_model, pm_solver, grid_name=None, kind="tnep,repl", lc="scaled"):
    # runs the optimization and stores results to excel files
    # see details for options in get_command_line_args
    lc_type = "scaled_loadcases" if lc == "scaled" else "time_series_loadcases"
    start_path = os.path.join(os.getcwd(), lc_type)
    os.makedirs(get_res_dir(pm_model, pm_solver, init=lc), exist_ok=True)
    grids = get_grids(grid_name)
    for grid in grids:
        net_path = os.path.join(start_path, grid + ".json")
        loadcases_path = os.path.join(start_path, grid)
        print(f"Calculating optimization results for '{grid}' with optimizations '{kind}'")
        print(f"Model: '{pm_model}', Solver: '{pm_solver}'")
        df = test_power_models(loadcases_path, net_path, grid, pm_model, pm_solver, kind=kind, lc=lc)


def get_command_line_args(argv):
    """
    Gets the command line arguments which are:

    Returns
    -------
    pm_model (str) - The PowerModels.jl power model e.g. "DCPPowerModel"
    pm_solver (str) - The solver to use, e.g. "juniper", "gurobi"
    grid_name (str) - optional if you only want to calculate one grid, e.g. "rts"
    lc (str) - the loadcases to calculate. Either "ts" or "scaled"
    kind (str) - the optimizations to run, e.g. "tnep,ots,repl". Can only be a part of these like "tnep,repl"

    """
    pm_model, pm_solver, grid_name = None, None, None
    kind = "tnep,repl,ots"
    lc = "scaled"
    try:
        opts, args = getopt.getopt(argv, ":m:s:g:k:",
                                   ["model=", "solver=", "grid=", "kind="])
    except getopt.GetoptError:
        raise getopt.GetoptError("error reading input")
    for opt, arg in opts:
        if opt in ("-m", "--model"):
            pm_model = arg
        if opt in ("-s", "--solver"):
            pm_solver = arg
        if opt in ("-g", "--grid"):
            grid_name = arg
        if opt in ("-k", "--kind"):
            kind = arg
    if pm_solver is None:
        UserWarning("pm_solver is None. You must specify a solver with '--solver='")
    if pm_model is None:
        UserWarning("pm_model is None. You must specify a model with '--model='")
    return pm_model, pm_solver, grid_name, lc, kind


if __name__ == "__main__":
    # read command line args
    pm_model, pm_solver, grid_name, lc, kind = get_command_line_args(sys.argv[1:])
    # run opt
    start_power_models(pm_model, pm_solver, grid_name, kind=kind, lc=lc)
