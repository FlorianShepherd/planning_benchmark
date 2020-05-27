import argparse
import copy
import getpass
import logging
import pathlib
import random
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

import pandapower as pp
from planning_benchmark.seeds import seeds as seed_dict

user = getpass.getuser()
logger = logging.getLogger(__name__)


def init_logger(**settings):
    # init the logger
    logger.addHandler(logging.StreamHandler())
    if settings["log_file"] != "":
        logfile_path = pathlib.Path(settings["res_dir"], settings["grid"] + "_" + settings["log_file"])
        logfile_path.parents[0].mkdir(exist_ok=True, parents=True)
        logger.addHandler(logging.FileHandler(logfile_path))
    loglevel = settings["loglevel"]
    if loglevel == "error":
        logger.setLevel(logging.ERROR)
    elif loglevel == "debug":
        logger.setLevel(logging.DEBUG)
    elif loglevel == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    return logger


def get_random_scaling(seed, net, low=50., high=100.):
    # scales the generation and load values according to a random values so that we get different load cases
    if net.name == "rts":
        low, high = 90., 100.
    n_sgens = len(net.sgen)
    n_loads = len(net.load)

    random.seed(seed)
    np.random.seed(int(seed))

    load_scaling = np.random.randint(low, high, n_loads)
    sgen_scaling = np.random.randint(low, high, n_sgens)
    net.sgen.loc[:, "scaling"] = sgen_scaling / 100.
    net.load.loc[:, "scaling"] = load_scaling / 100.
    return net


def get_file(grid):
    # gets file path
    net_file = pathlib.Path(pathlib.Path.cwd(), "power_system_data", grid + "_measures.json")
    return net_file


def get_grid(seed, **settings):
    # open the current test systems
    net = pp.from_json(get_file(settings["grid"]))
    # deactivate additional and replacement lines first
    net.line.loc[net.line.loc[:, "repl"], "in_service"] = False
    net.line.loc[net.line.loc[:, "additional_line"], "in_service"] = False
    # remove additional/repl lines depending on the optimization
    kind = settings["kind"]
    als = net.line.loc[net.line.loc[:, "additional_line"]].index
    repls = net.line.loc[net.line.loc[:, "repl"]].index
    if kind == "repl":
        # keep only repl measures
        net.line.drop(index=als, inplace=True)
        net.ne_line.drop(index=als, inplace=True)
        net.res_ne_line.drop(index=als, inplace=True)
    if kind == "ial":
        # keep only additional lines
        net.line.drop(index=repls, inplace=True)
        net.ne_line.drop(index=repls, inplace=True)
        net.res_ne_line.drop(index=repls, inplace=True)
    if kind == "sso":
        # drop all line measures
        net.line.drop(index=repls, inplace=True)
        net.ne_line.drop(index=repls, inplace=True)
        net.res_ne_line.drop(index=repls, inplace=True)
        net.line.drop(index=als, inplace=True)
        net.ne_line.drop(index=als, inplace=True)
        net.res_ne_line.drop(index=als, inplace=True)

    # drop switches of additional lines / replacement lines
    max_line = net.line.index.max()
    bl_switch = net.switch.loc[net.switch.et == "l"]
    drop_switch = bl_switch.loc[bl_switch.element > max_line].index
    net.switch.drop(index=drop_switch, inplace=True)

    # set limits for I and V
    net.line.loc[:, "max_loading_percent"] = settings["i_lim"]
    net.bus.loc[:, "max_vm_pu"] = settings["v_max_lim"]
    net.bus.loc[:, "min_vm_pu"] = settings["v_min_lim"]
    # initialize load case by scaling the sgen and load values
    net = get_random_scaling(seed, net)
    return net


def init_weights():
    # weights are: max. current, upper voltage limit violations, lower voltage limit violations,
    # number of current violations (line length), and number of voltage violations (number of buses)
    return {"i_max": 1., "v_max": 1., "v_min": 1., "n_i_viol": 1., "n_v_viol": 1.}


def eval_pf(net):
    pf_results = {
        "i_max": 9999.,
        "v_max": 9999.,
        "v_min": 0.,
        "converged": False,
        "n_i_viol": len(net.line),
        "n_v_viol": len(net.bus)
    }
    try:
        pp.runpp(net, calculate_voltage_angles=True)
        pf_results["converged"] = True
        pf_results["i_max"] = net.res_line.loading_percent.max()
        pf_results["v_max"] = net.res_bus.vm_pu.max()
        pf_results["v_min"] = net.res_bus.vm_pu.min()
        pf_results["n_i_viol"] = ((np.sum(net.line.loc[net.res_line.loading_percent.values > settings["i_lim"],
                                                       "length_km"].values)) / net.line.length_km.sum())
        pf_results["n_v_viol"] = (np.sum(net.res_bus.vm_pu.values > settings["v_max_lim"])
                                  + np.sum(net.res_bus.vm_pu.values < settings["v_min_lim"]) / len(net.bus))
    except pp.LoadflowNotConverged:
        logger.debug("loadflow not converged. cost max.")
    return pf_results


def cost_function(net, weights=None, opt_type="switch", **settings):
    weights = init_weights() if weights is None else weights

    pf_results = eval_pf(net)
    if not pf_results["converged"]:
        return 1e12

    n_remain = len(net.res_bus.dropna())
    n_bus = len(net.bus)
    disco = n_bus != n_remain
    if disco:
        cost = 1e10 + (n_bus - n_remain)
    else:
        i_max, v_max, v_min = pf_results["i_max"], pf_results["v_max"], pf_results["v_min"]
        # cost = 0 if nothing is violated. Changing the weights changes the importance of the cost term.
        # every cost term is in percent (100.)
        cost = max(0, i_max - settings["i_lim"]) * weights["i_max"] + \
               max(0, v_max - settings["v_max_lim"]) * weights["v_max"] * 100. + \
               max(0, settings["v_min_lim"] - v_min) * weights["v_min"] * 100. + \
               pf_results["n_v_viol"] * weights["n_v_viol"] * 100. + \
               pf_results["n_i_viol"] * weights["n_i_viol"] * 100.
        if cost <= 0.:
            if opt_type == "line":
                cost -= (1 - net.line.loc[net.line.loc[:, "in_service"], "length_km"].sum() /
                         net.line.loc[:, "length_km"].sum())
            else:
                # minimize losses
                cost -= sum(abs(net.res_line.p_from_mw + net.res_line.p_to_mw))
    return cost


def apply(net, m, opt_type):
    # apply = open switch or set new power line in service
    if opt_type == "switch":
        net["switch"].at[m, "closed"] = False
    else:
        net["line"].at[m, "in_service"] = True
    return net


def unapply(net, m, opt_type):
    # unapply = close switch or set new power line out of service
    if opt_type == "switch":
        net["switch"].at[m, "closed"] = True
    else:
        net["line"].at[m, "in_service"] = False
    return net


def greedy_alg(net, weights=None, opt_type="switch", measures=None, **settings):
    weight_string = [str(key) + ': ' + f'{val:.2f}' for key, val in weights.items()]
    logger.info(f"{opt_type} optimization with weights {weight_string}")

    # start opt
    init_switch_pos = copy.deepcopy(net.switch.loc[:, "closed"].values)

    if opt_type == "switch":
        measures = net.switch.index if measures is None else measures
        measures = set(measures)
    else:
        measures = net.ne_line.index if measures is None else measures
        measures = set(measures)

    cost = cost_function(net, weights, opt_type=opt_type, **settings)
    cost_best = cost
    bests = list()
    while len(measures):
        best = None
        for m in measures:
            net = apply(net, m, opt_type)
            cost = cost_function(net, weights, opt_type=opt_type, **settings)
            if cost > 1e11:
                net = unapply(net, m, opt_type)
                continue

            if cost < cost_best:
                best = m
                cost_best = cost
            net = unapply(net, m, opt_type)
        if best is not None:
            # best found -> keep
            net = apply(net, best, opt_type)
            bests.append(best)
            # remove best from indices
            measures = measures - {best}
        if best is None:  # and worst is None:
            # no improvement / worsening possible
            break

    logger.debug(f"{opt_type} solution:")
    logger.debug(bests)
    if opt_type == "switch":
        net.switch.loc[:, "closed"] = init_switch_pos
        net.switch.loc[bests, "closed"] = False

    cost = cost_function(net, weights, opt_type=opt_type, **settings)
    pf_results = eval_pf(net)
    i_max, v_max, v_min = pf_results["i_max"], pf_results["v_max"], pf_results["v_min"]
    # get km of built new lines
    new_lines = net.line.loc[net.ne_line.index]
    lines = new_lines[new_lines.in_service.values]
    km = lines.loc[:, "length_km"].sum()

    logger.debug(f"cost {cost}")
    logger.debug("power flow results result:")
    logger.debug(f"i_max: {i_max:.2f}, v_max {v_max:.3f}, v_min {v_min:.3f}")
    logger.debug("\n")

    result = {"sol": net.switch, "cost": cost, "i_max": i_max, "t": perf_counter() - settings["t0"],
              "v_max": v_max, "v_min": v_min, "lines": list(lines.index), "km": km}

    return result


def run_pm_tnep(net, **settings):
    logger.debug("Starting PowerModels.jl TNEP:")
    lines = set()
    # these steps are the decreasing line loading limits so that we get different measures
    # we need this becaus: DCP 80% I_max != DCP 90% I_max
    steps = [.9, .95, 1.]
    step1_sol = []

    # time limits for PowerModels.jl solver
    pm_time_limits = {"pm_time_limit": settings["pm_time_limit"],
                      "pm_nl_time_limit": settings["pm_nl_time_limit"],
                      "pm_mip_time_limit": settings["pm_mip_time_limit"]}

    for step in steps:
        logger.debug(f"starting PowerModels.jl TNEP with with max. line loading {step}")
        # reduce the max. line loading
        net.line.loc[:, "max_loading_percent"] = settings["i_lim"] * step
        net.ne_line.loc[:, "max_loading_percent"] = settings["i_lim"] * step
        try:
            # runs PowerModels.jl TNEP
            pp.runpm_tnep(net, pm_model=settings["model"], pm_solver=settings["solver"], pm_log_level=1,
                          pm_time_limits=pm_time_limits)
        except:
            logger.error("power models failed.")
            continue
        # the PowerModels.jl solution
        sol = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
        if step == 1.:
            # 90% solution of PowerModels (probably not AC feasible)
            step1_sol = sol
        # all lines 'recommended' by the TNEP opt
        lines |= set(list(sol))
    # reset the limits to the original value
    net.ne_line.loc[:, "max_loading_percent"] = settings["i_lim"]
    net.line.loc[:, "max_loading_percent"] = settings["i_lim"]

    if len(step1_sol):
        # check if the best PowerModels.jl solution is good
        check_pm_solution(net, step1_sol)

    return lines, step1_sol


def check_pm_solution(net, step1_sol):
    logger.debug(f"applying step1 sol measures: {step1_sol}")
    net.line.loc[step1_sol, "in_service"] = True
    pf_results = eval_pf(net)
    i_max, v_max, v_min = pf_results["i_max"], pf_results["v_max"], pf_results["v_min"]
    logger.debug(f"i_max: {i_max:.2f}, v_max {v_max:.3f}, v_min {v_min:.3f}")
    net.line.loc[step1_sol, "in_service"] = False


def line_opt(net, weights=None, **settings):
    # set of ne lines are the measures
    measures = set(net.ne_line.index)

    # info about the initial solution
    logger.debug("line opt power flow result init:")
    pf_results = eval_pf(net)
    i_max, v_max, v_min = pf_results["i_max"], pf_results["v_max"], pf_results["v_min"]
    logger.debug(f"i_max: {i_max:.2f}, v_max {v_max:.3f}, v_min {v_min:.3f}")

    if "tnep" in settings["pm"]:
        # run PowerModels.jl TNEP
        res, step1_sol = run_pm_tnep(net, **settings)
        # define the TNEP result as the measure set
        measures = set(res) if len(res) else measures

    # start greedy algorithm with measures from PowerModels
    result = greedy_alg(net, weights=weights, opt_type="line", measures=measures, **settings)

    # if that didn't work -> start greedy again with all measures
    if settings["pm"] != "" and result["cost"] > 0.:
        logger.debug("PowerModels.jl OPF could not find an AC solution. Starting with all measures")
        measures = set(net.ne_line.index)
        net.line.loc[measures, "in_service"] = False
        result = greedy_alg(net, weights=weights, opt_type="line", measures=measures, **settings)

    return result


def change_weights(iteration):
    weights = init_weights()
    if iteration == 0:
        # identical weights
        pass
    # elif iteration == 1:
    #     # reduce v_max violation is most important
    #     weights["v_max"] = 1000.
    # elif iteration == 2:
    #     # reduce i_max violation is most important
    #     weights["i_max"] = 1000.
    else:
        # random weights
        factor = 1.e1
        randy = lambda: np.random.random() * factor
        weights = {"i_max": randy(), "v_max": randy(),
                   "v_min": randy(), "n_i_viol": randy(),
                   "n_v_viol": randy()}
    return weights


def get_best(result):
    # return best solution found
    best_cost = 1e13
    best_solution = dict()
    for key, val in result.items():
        # cost are the shortest replaced length or the minimal losses if nothing was replaced
        cost = val["km"]
        if cost <= 1e-4:
            cost = val["cost"]
        if cost < best_cost:
            best_cost = cost
            best_solution = result[key]
    return best_solution


def write_results(df, i, result, **settings):
    # writes results to an excel file
    best = get_best(result)
    grid = settings["grid"]
    kind = settings["kind"]
    sol, lines, km = best["sol"], best["lines"], best["km"]
    cost = best["cost"]
    i_max, v_max, v_min = best["i_max"], best["v_max"], best["v_min"]

    df.at[i, kind + "_t"] = best["t"]
    lines.sort()
    open_switches = list(sol.loc[~sol.closed.values.astype(bool)].index) if sol is not None else []
    closed_switches = list(sol.loc[sol.closed.values.astype(bool)].index) if sol is not None else []
    df.at[i, kind + "_solution"] = str(open_switches + ["line_" + str(l) for l in lines])
    df.at[i, kind + "_built"] = str(lines)
    df.at[i, kind + "_closed"] = str(closed_switches)
    df.at[i, kind + "_converged"] = 1
    df.at[i, kind + "_worked"] = 1 if cost <= 0. else 0
    df.at[i, kind + "_ap_level"] = 0 if cost <= 0. else (3 if i_max > 100. else 2)
    df.at[i, kind + "_max_ll"] = i_max
    df.at[i, kind + "_vmin"] = v_min
    df.at[i, kind + "_vmax"] = v_max
    if km > 0.:
        # costs are replaced line km
        df.at[i, kind + "_cost"] = km
    else:
        # costs are losses
        df.at[i, kind + "_cost"] = cost
    if settings["pm"] != "":
        file_path = pathlib.Path(settings["res_dir"], "greedy",
                                 grid + "_" + settings["pm"] + "_" + settings["kind"] + ".xlsx")
    else:
        file_path = pathlib.Path(settings["res_dir"], "greedy", grid + "_" + settings["kind"] + ".xlsx")

    file_path.parent.mkdir(exist_ok=True)
    df.to_excel(file_path)


def initial_power_flow(seed, df, i, **settings):
    # run an initial power flow calculation to get V and I violations before any optimization
    logger.info(f"Initial power flow results for seed {seed}:")
    net = get_grid(seed, **settings)
    pf_results = eval_pf(net)
    i_max, v_max, v_min = pf_results["i_max"], pf_results["v_max"], pf_results["v_min"]
    logger.info(f"i_max: {i_max:.2f}, v_max {v_max:.3f}, v_min {v_min:.3f}")
    if df is not None:
        df.at[i, "ll_init"] = i_max
        df.at[i, "vmax_init"] = v_max
        df.at[i, "vmin_init"] = v_min


def get_columns(kind):
    # inits the result columns for the pandas dataframe which are always "kind_xy", e.g., "repl_cost" etc.
    return ["seed", "ll_init", "vmin_init", "vmax_init",
            kind + "_converged", kind + "_worked", kind + "_solution", kind + "_built", kind + "_closed",
            kind + "_max_ll", kind + "_vmax", kind + "_vmin", kind + "_t", kind + "_ap_level",
            kind + "_cost"]


def init_df(kind, seeds):
    # init result DataFrame
    columns = get_columns(kind)
    df = pd.DataFrame(index=list(range(len(seeds))), columns=columns)
    return df


def init_iteration(df, iteration, i, seed, **settings):
    net = get_grid(seed, **settings)
    # some init values for the current iteration
    res = {"cost": np.inf}
    switch_sol = None
    np.random.seed(seed * iteration)
    weights = change_weights(iteration)
    initial_power_flow(seed, df, i, **settings)
    return net, df, res, switch_sol, weights


def print_solution(res):
    sol = res["sol"]
    open_switches = list(sol.loc[~sol.closed.values.astype(bool)].index) if sol is not None else []
    cost = res["cost"]
    i_max = res["i_max"]
    v_max = res["v_max"]
    v_min = res["v_min"]
    t = res["t"]
    if cost < 0:
        km = res["km"]
        if km > 0.:
            lines = res["lines"]
            logger.info(f"Valid AC line solution found after {t:.2f}s. Built line km: {km:.2f}, lines {lines}, "
                        f"Opened switches {open_switches}, i_max: {i_max:.2f}, v_max: {v_max:.2f}, v_min {v_min:.2f}")
        else:
            logger.info(f"Valid AC switch solution found after {t:.2f}s. Losses {cost:.2f} MWh, "
                        f"Opened switches {open_switches}, i_max: {i_max:.2f}, v_max: {v_max:.2f}, v_min {v_min:.2f}")
    else:
        logger.info(f"No valid AC line solution found after {t:.2f}s. "
                    f"Opened switches {open_switches}, i_max: {i_max:.2f}, v_max: {v_max:.2f}, v_min {v_min:.2f}")


def start_greedy(df, i, seed, **settings):
    # iteration limit -> how often the algorithm is restarted
    iteration_limit = settings["max_iterations"]
    # optimization kind, e.g., 'repl', 'ial', 'sso', or 'combo'
    kind = settings["kind"]

    # df contains results
    df.at[i, "seed"] = seed
    result = dict()

    # measure time
    t0 = perf_counter()
    settings["t0"] = t0
    iteration = 0

    while iteration < iteration_limit and perf_counter() - t0 < settings["time_limit"]:
        logger.info(f"\nseed {seed}. iteration {iteration}. time {perf_counter() - t0:.2f}")
        net, df, res, switch_sol, weights = init_iteration(df, iteration, i, seed, **settings)

        if kind in ["sso", "combo"]:
            # optimize switching state
            res = greedy_alg(net, weights=weights, opt_type="switch", **settings)

        if kind in ["ial", "repl", "combo"] and res["cost"] > 0.:
            # optimize line replacements / new line measures or a combination of both line types
            res = line_opt(net, weights=weights, **settings)

        result[iteration] = res
        print_solution(res)

        # store results to an excel file if res_dir is given
        if settings["res_dir"] != "":
            write_results(df, i, result, **settings)
        iteration += 1


def get_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', type=str, default="combo", help="kind of optimization: 'repl', 'sso', 'ial', 'combo'")
    parser.add_argument('-pm', type=str, default="tnep", help="use PowerModels.jl for optimization. "
                                                              "To deactive set -pm ''")
    parser.add_argument('-pm_time_limit', type=float, default=300., help="PowerModels.jl overall time limit in s. "
                                                                         "Default 300")
    parser.add_argument('-pm_nl_time_limit', type=float, default=300., help="PowerModels.jl nonlinear solver time "
                                                                            "limit in s. Default 300")
    parser.add_argument('-pm_mip_time_limit', type=float, default=300., help="PowerModels.jl MIP solver time "
                                                                             "limit in s. Default 300")
    parser.add_argument('-model', type=str, default="DCPPowerModel", help="PowerModels.jl model. Use, e.g., "
                                                                          "'DCPPowerModel' or 'ACPPowerModel'.")
    parser.add_argument('-solver', type=str, default="gurobi", help="PowerModels.jl solver. Default is 'gurobi'")
    parser.add_argument('-grid', type=str, default="brigande", help="The name of the power system you want to optimize."
                                                                    "Choose either 'brigande' 'sb_mixed'"
                                                                    " 'sb_urban' or 'rts'")
    parser.add_argument('-i_lim', type=float, default=100., help="The maximum line loading limit. Default 100 percent")
    parser.add_argument('-v_max_lim', type=float, default=1.1, help="upper voltage limit. Default 1.1 p.u.")
    parser.add_argument('-v_min_lim', type=float, default=0.9, help="lower voltage limit. Default 0.9 p.u.")
    parser.add_argument('-max_iterations', type=int, default=10, help="max. iterations for the greedy algorithm. "
                                                                      "Default 10")
    parser.add_argument('-time_limit', type=int, default=600, help="time limit in seconds for the optimization. Default"
                                                                   " 600")
    parser.add_argument('-res_dir', type=str, default=str(pathlib.Path(pathlib.Path.cwd(), "results")),
                        help="dir to write excel results to")
    parser.add_argument('-loglevel', type=str, default="info", help="logging output. can be 'info' 'debug' 'warning'"
                                                                    "or 'error'")
    parser.add_argument('-log_file', type=str, default="greedy.log", help="log to file")
    args = parser.parse_args()
    # return as dict
    return vars(args)


def main_optimization(**settings):
    # grid is the name of the test case, e.g., "brigande".
    grid = settings["grid"]
    # kind is either "repl", "ial", "sso", or "combo"
    kind = settings["kind"]

    logger.info(f"Starting greedy for {grid}")
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # pre-defined seeds so that we get the 130 benchmark cases
    seeds = seed_dict[grid]
    # results are written to df
    df = init_df(kind, seeds)

    for i, seed in enumerate(seeds):
        # start SSO, REPL, IAL, or combo
        start_greedy(df, i, seed, **settings)


if __name__ == "__main__":
    # settings contains all settings from the command line
    settings = get_command_line_args()
    logger = init_logger(**settings)
    main_optimization(**settings)
