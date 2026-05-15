import argparse
from ..theory.bound_validator import BoundValidator

def main():
    parser = argparse.ArgumentParser(description='Bound Validation Experiment')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--experiment', type=str, choices=['a', 'b'], required=True)
    args = parser.parse_args()
    validator = BoundValidator()
    if args.experiment == 'a':
        results = validator.run_experiment_a(None, None, None)
        validator.plot_experiment_a(results, 'outputs/bound_validation/experiment_a.pdf')
    else:
        results = validator.run_experiment_b(None, None, None)
        validator.plot_experiment_b(results, 'outputs/bound_validation/experiment_b.pdf')
    print('Experiment complete. Results saved.')

if __name__ == '__main__':
    main()
