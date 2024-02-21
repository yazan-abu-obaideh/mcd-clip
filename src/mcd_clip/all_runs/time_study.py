import cProfile

# this is used so the call inside cProfile runs
# noinspection PyUnresolvedReferences
from mcd_clip.all_runs.difficult_combined_run import run

if __name__ == '__main__':
    cProfile.run(
        statement="""run(
            plot=False,
            generations=200,
            batch_size=200
        )
    """, filename='profile.txt'
    )
