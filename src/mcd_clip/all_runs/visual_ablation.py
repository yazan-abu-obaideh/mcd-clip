from mcd_clip.all_runs.ablation_runs import run

if __name__ == '__main__':
    run(
        features_on=True,
        run_id_suffix='-mcd'
    )
    run(
        features_on=False,
        run_id_suffix='classical'
    )
