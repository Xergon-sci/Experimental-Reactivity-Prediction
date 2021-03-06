from git import Repo, exc
from git.objects import commit

repo = Repo('/scratch/brussel/102/vsc10255/Experimental-Reactivity-Prediction')
assert not repo.bare

def batched_commit():
    commit = 0
    cnt = 0
    for f in repo.untracked_files:
        if cnt < 50:
            cnt = cnt +1
            try:
                repo.index.add([f])
            except:
                print(f, ' gave an error.')
        else:
            print('Git commit and push')
            commit = commit +1
            repo.index.commit('Batch commit nr: {}'.format(commit))
            cnt = 0

batched_commit()
