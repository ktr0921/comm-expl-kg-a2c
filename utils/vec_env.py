import os
import redis
import time
import subprocess
from multiprocessing import Process, Pipe


# redis_path redis_config_path redis_port
def start_redis(redis_path, redis_config_path):
    print(f'Starting Redis')
    redis_command = f'{redis_path} {redis_config_path}'
    print(redis_command)
    subprocess.Popen(redis_command, shell=True)
    # subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])

    time.sleep(1)


def start_openie(install_path, openie_port):
    print('Starting OpenIE from', install_path)
    subprocess.Popen(
        ['java', '-mx8g', '-cp', '*', 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', '-port', str(openie_port),
         '-timeout', '15000', '-quiet'], cwd=install_path)
    time.sleep(1)


def worker(remote, parent_remote, env):
    parent_remote.close()
    env.create()
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if done:
                    ob, info, graph_info = env.reset()
                    rew = 0
                    done = False
                else:
                    ob, rew, done, info, graph_info = env.step(data)
                remote.send((ob, rew, done, info, graph_info))
            elif cmd == 'reset':
                ob, info, graph_info = env.reset()
                remote.send((ob, info, graph_info))
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env, openie_path, openie_port, redis_path, redis_config_path, redis_port):
        start_redis(redis_path, redis_config_path)
        start_openie(openie_path, openie_port)
        self.conn_valid = redis.Redis(host='localhost', port=redis_port, db=0)

        self.closed = False
        self.total_steps = 0
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=self.worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        if self.total_steps % 1024 == 0:
            self.conn_valid.flushdb()
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]  # .recv() means receiving from .send()
        self.waiting = False
        return zip(*results)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]  # .recv() means receiving from .send()
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def worker(self, remote, parent_remote, env):
        try:
            worker(remote, parent_remote, env)
        except:
            print('exception occured')
            self.conn_valid.flushdb()
            self.conn_valid.flushall()
            worker(remote, parent_remote, env)
