from git import Repo, exc
from git.objects import commit

repo = Repo('/scratch/brussel/102/vsc10255/Experimental-Reactivity-Prediction')
assert not repo.bare

def batched_commit():
    untracked = repo.untracked_files
    selection = untracked[:250]
    repo.index.add(selection)
    repo.index.commit('Another chunk')
    origin = repo.remote()
    origin.push()
    batched_commit()

batched_commit()
