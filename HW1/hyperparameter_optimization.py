import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import time
import sys, os
import argparse
from script import solve_puzzle, Args


PUZZLE='puzzle_affine_7'


def sample_params(args, trial):
    args.max_iterations = trial.suggest_int('max_iterations', 100, 3000)
    args.distance_threshold = trial.suggest_float('distance_threshold', 0.5, 10.0)
    # args.nfeatures = trial.suggest_int('nfeatures', 100, 1000)
    # args.nOctaveLayers = trial.suggest_int('nOctaveLayers', 1, 10)
    # args.contrastThreshold = trial.suggest_uniform('contrastThreshold', 0.01, 0.1)
    # args.edgeThreshold = trial.suggest_int('edgeThreshold', 1, 10)
    # args.sigma = trial.suggest_uniform('sigma', 1.0, 2.0)
    # args.nOctaves = trial.suggest_int('nOctaves', 1, 10)
    args.best_match_threshold = trial.suggest_float('best_match_threshold', 0.5, 1.0)
    args.min_required_matches = trial.suggest_int('min_required_matches', 8, 30)


def objective(trial):
    args = Args()
    args.save_results = False
    args.puzzle_dir = f'puzzles/{PUZZLE}'
    sample_params(args, trial)
    pieces_solved_ratio, _ = solve_puzzle(args.puzzle_dir, args)
    return pieces_solved_ratio

def run_study():
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler()

    study = optuna.create_study(study_name=f'{PUZZLE}_opt_{time.time()}', storage='sqlite:///db.sqlite3', sampler=sampler, direction='maximize')
    try:
        study.optimize(objective, n_trials=100, n_jobs=4)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    if not os.path.exists('study_results'):
        os.makedirs('study_results')
    study.trials_dataframe().to_csv(f"study_results/study_results_{PUZZLE}.csv")

    return study.best_params

def args_parser(argv):
    parser = argparse.ArgumentParser(description='Optimize hyperparameters')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle name')
    args=parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = args_parser(sys.argv[1:])
    if args.puzzle:
        PUZZLE = args.puzzle
        run_study()
    else:
        for puzzle in os.listdir('puzzles'):
            PUZZLE = puzzle
            run_study()
