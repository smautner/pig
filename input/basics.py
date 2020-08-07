lmap = lambda *x: list(map(*x))
lzip = lambda *x: list(zip(*x))


import dill
import time
dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
loadfile = lambda filename: dill.load(open(filename, "rb"))

def mpmap(func, iterable, chunksize=10, poolsize=2):
    import multiprocessing as mp
    """pmap."""
    pool = mp.Pool(poolsize)
    result = pool.map(func, iterable, chunksize=chunksize)
    pool.close()
    pool.join()
    return list(result)

# functools partial can set some arguments...
def mpmap_prog(func, iterable, chunksize=10, poolsize=2):
    import multiprocessing as mp
    import tqdm
    """pmap."""
    pool = mp.Pool(poolsize)
    result = list(tqdm.tqdm( pool.imap(func, iterable, chunksize=chunksize), total=len(iterable)))
    pool.close()
    pool.join()
    return result

def shexec(cmd):
    import subprocess
    '''
    :param cmd:
    :return: (exit-code, stderr, stdout)
    the subprocess module is chogeum.. here is a workaround
    '''
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, stderr = process.communicate()
    retcode = process.poll()
    return (retcode,stderr,output) # .decode('utf-8') something like this might be necessary for py3

def shexec_and_wait(cmd):
    ret,stderr,out = shexec(cmd)
    taskid = out.split()[2][:7]
    print("taskid:", int(taskid))
    while taskid in shexec("qstat")[2]:
        time.sleep(10)

def interact():
    import code
    code.interact(local=dict(globals(), **locals()))
