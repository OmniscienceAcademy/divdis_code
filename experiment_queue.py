from subprocess import call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("device")
args = parser.parse_args()

if args.experiment_name == "waterbirds_sweep":
    layers = [18, 34]
    for l in layers:
        call(
            "python run_experiment.py --container CorrelatedBirdsContainer --note {} --user cuda:{} --n_layers".format(
                args.experiment_name,
                args.device,
                l,
            ).split(
                " "
            )
        )

if args.experiment_name == "HF_best_limit_target":
    limit = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
    for n in range(4):
        for l in limit:
            call(
                "python run_experiment.py --mix_rate 1 --limit {} --preset HF_best --note {} --user cuda:{}".format(
                    l, args.experiment_name, args.device
                ).split(
                    " "
                )
            )


if args.experiment_name == "CM_best_test":
    mix_rates = range(0, 11)
    for n in range(8):
        for mr in mix_rates:
            call(
                "python run_experiment.py --mix_rate {} --test --preset CM_best --note {} --user cuda:{}".format(
                    (mr / 10), args.experiment_name, args.device
                ).split(
                    " "
                )
            )

if args.experiment_name == "CM_best_dd_test":
    mix_rates = range(0, 11)
    for n in range(8):
        for mr in mix_rates:
            call(
                "python run_experiment.py --loss_type 1 --mix_rate {} --test --preset CM_best_ccs1 --note {} --user cuda:{}".format(
                    (mr / 10), args.experiment_name, args.device
                ).split(
                    " "
                )
            )

if args.experiment_name == "HF_best_ccs1_test_2":
    mix_rates = range(0, 11)
    for n in range(4):
        for mr in mix_rates:
            call(
                "python run_experiment.py --loss_type 1 --mix_rate {} --test --preset HF_best_ccs1_2 --note {} --user cuda:{}".format(
                    mr / 10, args.experiment_name, args.device
                ).split(
                    " "
                )
            )

if args.experiment_name == "HF_multi_dd_best5_ccs1":
    mix_rates = range(0, 11)
    for mr in mix_rates:
        call(
            "python run_experiment.py --loss_type 1 --mix_rate {} --preset HF_multi_best5_ccs1 --note {} --user cuda:{}".format(
                mr / 10, args.experiment_name, args.device
            ).split(
                " "
            )
        )
