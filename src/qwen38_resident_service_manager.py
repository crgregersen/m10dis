"""Host-side layer readiness and request gate for the routed/shared CUDA workers."""
from __future__ import annotations
import json, os, struct, subprocess, threading, time, uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

def _canonical_bdf(value: str) -> str:
    domain, bus, devfn = value.strip().lower().split(":")
    device, function = devfn.split(".")
    return (f"{int(domain, 16):04x}:{int(bus, 16):02x}:"
            f"{int(device, 16):02x}.{int(function, 16):x}")

class ResidentServiceManager:
    def __init__(self, routed_binary: Path, shared_binary: Path, out: Path,
                 num_layers: int = 48):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen38_m10_expert_sim as q
        routed, shared, groups = q.choose_gpu_groups()
        self._branch_bdfs = {
            'routed': [_canonical_bdf(bdf) for group in groups[:5]
                       for bdf in group['bdfs']],
            'shared': [_canonical_bdf(bdf) for bdf in groups[5]['bdfs']],
        }
        snapshot_root = os.environ.get('Q38_MULTICAST_SNAPSHOT')
        if snapshot_root:
            run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            snapshot_root = str(Path(snapshot_root) / f'daemon-{run_id}-{os.getpid()}')
            os.environ['Q38_MULTICAST_SNAPSHOT'] = snapshot_root
        self.multicast_snapshot_root = snapshot_root
        self._scratch = out / 'ple_rpc_scratch'
        self._scratch.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        er = env.copy(); er['CUDA_VISIBLE_DEVICES'] = ','.join(routed)
        es = env.copy(); es['CUDA_VISIBLE_DEVICES'] = ','.join(shared)
        def worker_command(binary: Path, worker_env: dict[str, str]) -> list[str]:
            if env.get('Q38_EXPERIMENTAL_PRIVILEGED_WORKERS') != '1':
                return [str(binary.resolve())]
            # Only the experimental multicast workers need access to the
            # capability-gated BAR mapper and PCI configuration. Keep the
            # daemon, PLE owner and vLLM client under the invoking user.
            names = ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
                     'Q38_MULTICAST_GROUPS', 'Q38_MULTICAST_SNAPSHOT',
                     'Q38_HEARTBEAT_SECONDS', 'Q38_RESIDENT_IDLE_DEADLINE_CYCLES',
                     'Q38_PLE_IPC_MANIFEST')
            values = [f'{name}={worker_env[name]}' for name in names
                      if name in worker_env]
            preload = worker_env.get('Q38_RM_TRACE_PRELOAD')
            if not preload or not all(name in worker_env for name in
                                      ('Q38_MULTICAST_GROUPS', 'Q38_MULTICAST_SNAPSHOT')):
                raise ValueError('experimental multicast worker configuration is incomplete')
            values.append(f'LD_PRELOAD={preload}')
            return ['sudo', '-n', '/usr/bin/env', *values, str(binary.resolve())]
        out.mkdir(parents=True, exist_ok=True)
        self._rlog = (out/'manager_routed.log').open('w')
        self._slog = (out/'manager_shared.log').open('w')
        self._r = subprocess.Popen(worker_command(routed_binary, er), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._rlog, text=True, bufsize=1, env=er)
        self._s = subprocess.Popen(worker_command(shared_binary, es), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._slog, text=True, bufsize=1, env=es)
        self._locks = {'routed': threading.Lock(), 'shared': threading.Lock()}
        self._reload_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._states = ['NOT_LOADED'] * num_layers
        self._worker_activity = 'ACTIVE'
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._futures: list[Future] = []

    def _send(self, branch: str, command: str) -> dict[str, Any]:
        p = self._r if branch == 'routed' else self._s
        with self._locks[branch]:
            if p.poll() is not None:
                raise RuntimeError(f'{branch} service exited with {p.returncode}')
            p.stdin.write(command+'\n'); p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f'{branch} service closed stdout while handling {command}')
            return json.loads(line)

    def _timed_send(self, branch: str, command: str):
        started = time.perf_counter()
        response = self._send(branch, command)
        return response, time.perf_counter() - started

    def reload_kernel_images(self, routed_image: Path, shared_image: Path) -> dict[str, Any]:
        """Swap resident kernel cubins in-place without releasing expert weights.

        Callers must first quiesce external direct-PEX clients. Each worker
        stops and acknowledges its resident grid, loads the new module in its
        existing CUDA contexts, then relaunches the grid against the same
        allocations.
        """
        paths = {'routed': routed_image.resolve(), 'shared': shared_image.resolve()}
        for branch, path in paths.items():
            if not path.is_file():
                return {'status': 'error', 'code': 'KERNEL_IMAGE_NOT_FOUND',
                        'branch': branch, 'path': str(path)}
            if any(ch.isspace() for ch in str(path)):
                return {'status': 'error', 'code': 'INVALID_KERNEL_IMAGE_PATH',
                        'branch': branch, 'path': str(path)}
        with self._reload_lock:
            futures = {
                branch: self._pool.submit(self._send, branch,
                                          f'RELOAD_KERNELS {path}')
                for branch, path in paths.items()
            }
            replies = {branch: future.result()
                       for branch, future in futures.items()}
        ok = all(reply.get('status') == 'ok' and
                 reply.get('code') == 'KERNELS_RELOADED'
                 for reply in replies.values())
        return {'status': 'ok' if ok else 'error',
                'code': 'KERNELS_RELOADED' if ok else 'KERNEL_RELOAD_FAILED',
                'workers': replies}

    def set_workers_active(self, active: bool) -> dict[str, Any]:
        """Exit or relaunch polling kernels while retaining all model allocations."""
        command = 'WAKE_WORKERS' if active else 'PARK_WORKERS'
        expected = 'WORKERS_ACTIVE' if active else 'WORKERS_PARKED'
        with self._reload_lock:
            futures = {branch: self._pool.submit(self._send, branch, command)
                       for branch in ('routed', 'shared')}
            replies = {branch: future.result() for branch, future in futures.items()}
        ok = all(reply.get('status') == 'ok' and reply.get('code') == expected
                 for reply in replies.values())
        if ok:
            self._worker_activity = 'ACTIVE' if active else 'PARKED'
        return {'status': 'ok' if ok else 'error',
                'code': expected if ok else 'WORKER_LIFECYCLE_FAILED',
                'worker_activity': self._worker_activity, 'workers': replies}

    @property
    def worker_activity(self) -> str:
        return self._worker_activity

    def load_cpu_mailbox_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Register one CPU-owned POSIX shm mailbox in both CUDA workers."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from qwen38_cpu_mailbox_manifest import validate_manifest
        path = manifest_path.resolve()
        manifest = validate_manifest(path)
        if any(ch.isspace() for ch in str(path)):
            return {'status': 'error', 'code': 'INVALID_MANIFEST_PATH'}
        with self._reload_lock:
            futures = {branch: self._pool.submit(
                self._send, branch, f'LOAD_CPU_MAILBOX_MANIFEST {path}')
                for branch in ('routed', 'shared')}
            replies = {}
            for branch, future in futures.items():
                try:
                    replies[branch] = future.result()
                except Exception as exc:
                    replies[branch] = {'status': 'error', 'code': 'WORKER_COMMAND_FAILED',
                                       'message': f'{type(exc).__name__}: {exc}'}
            ok = all(reply.get('status') == 'ok' and reply.get('code') in
                     ('CPU_MAILBOX_LOADED', 'CPU_MAILBOX_ALREADY_LOADED')
                     for reply in replies.values()) and len(replies) == 2
            if not ok:
                rollback = {branch: self._pool.submit(
                    self._send, branch, 'UNLOAD_CPU_MAILBOX_MANIFEST')
                    for branch, reply in replies.items()
                    if reply.get('code') == 'CPU_MAILBOX_LOADED'}
                for future in rollback.values():
                    try:
                        future.result()
                    except Exception:
                        pass
            device_slots = []
            validation_error = None
            if ok:
                for branch in ('routed', 'shared'):
                    slots = replies[branch].get('device_slots')
                    expected_count = len(self._branch_bdfs[branch])
                    if not isinstance(slots, list) or len(slots) != expected_count:
                        validation_error = (f"{branch} returned {len(slots) if isinstance(slots, list) else 'non-list'} "
                                            f"device slots; expected {expected_count}")
                        ok = False
                        break
                    for local_index, slot in enumerate(slots):
                        if not isinstance(slot, dict):
                            validation_error = f"{branch} device slot {local_index} is not an object"
                            ok = False
                            break
                        group, worker = ((local_index // 4, local_index % 4)
                                         if branch == 'routed'
                                         else (5, local_index))
                        bdf = self._branch_bdfs[branch][local_index].lower()
                        try:
                            actual_bdf = _canonical_bdf(slot.get('bdf', ''))
                            device_va = int(slot.get('device_va', '0'), 0)
                        except (AttributeError, TypeError, ValueError) as exc:
                            validation_error = (f"{branch} device slot {local_index} has invalid "
                                                f"BDF/VA: {exc}")
                            ok = False
                            break
                        if (slot.get('group') != group or slot.get('worker') != worker or
                                actual_bdf != bdf or device_va <= 0):
                            validation_error = (
                                f"{branch} device slot {local_index} mismatch: "
                                f"expected group={group}, worker={worker}, bdf={bdf}; "
                                f"got group={slot.get('group')}, worker={slot.get('worker')}, "
                                f"bdf={actual_bdf}, va=0x{device_va:x}")
                            ok = False
                            break
                        device_slots.append({
                            'group': group, 'worker': worker, 'bdf': bdf,
                            'device_va': f"0x{device_va:x}",
                        })
                    if not ok:
                        break
            if not ok and all(reply.get('status') == 'ok' for reply in replies.values()):
                # Alias metadata is part of the registration contract. Roll
                # back both branches if it is absent or inconsistent.
                unloads = {branch: self._pool.submit(
                    self._send, branch, 'UNLOAD_CPU_MAILBOX_MANIFEST')
                    for branch in ('routed', 'shared')}
                for future in unloads.values():
                    try:
                        future.result()
                    except Exception:
                        pass
            return {'status': 'ok' if ok else 'error',
                    'code': 'CPU_MAILBOX_LOADED' if ok else 'CPU_MAILBOX_LOAD_FAILED',
                    'run_id': manifest['run_id'], 'manifest': str(path),
                    'workers': replies, 'device_slots': device_slots,
                    **({'validation_error': validation_error} if validation_error else {})}

    def unload_cpu_mailbox_manifest(self) -> dict[str, Any]:
        with self._reload_lock:
            futures = {branch: self._pool.submit(
                self._send, branch, 'UNLOAD_CPU_MAILBOX_MANIFEST')
                for branch in ('routed', 'shared')}
            replies = {branch: future.result() for branch, future in futures.items()}
        ok = all(reply.get('status') == 'ok' and
                 reply.get('code') == 'CPU_MAILBOX_UNLOADED'
                 for reply in replies.values())
        return {'status': 'ok' if ok else 'error',
                'code': 'CPU_MAILBOX_UNLOADED' if ok else 'CPU_MAILBOX_UNLOAD_FAILED',
                'workers': replies}

    def state(self, layer: int) -> dict[str, Any]:
        if not 0 <= layer < len(self._states):
            return {'status':'error','code':'INVALID_LAYER','layer_id':layer,
                    'valid_layer_range':[0,len(self._states)-1]}
        with self._state_lock:
            return {'status':'ok','layer_id':layer,'layer_state':self._states[layer]}

    def load_layer_async(self, layer: int, routed_dir: Path, shared_dir: Path) -> Future:
        if not 0 <= layer < len(self._states):
            f=Future(); f.set_result({'status':'error','code':'INVALID_LAYER','layer_id':layer,
                'valid_layer_range':[0,len(self._states)-1]}); return f
        with self._state_lock:
            if self._states[layer] == 'READY':
                f=Future(); f.set_result({'status':'ok','code':'LAYER_READY','layer_id':layer,'already_ready':True}); return f
            if self._states[layer] == 'LOADING':
                f=Future(); f.set_result({'status':'error','code':'LAYER_NOT_READY','layer_id':layer,'layer_state':'LOADING','retryable':True}); return f
            self._states[layer] = 'LOADING'
        f=self._pool.submit(self._load_pair,layer,str(routed_dir.resolve()),str(shared_dir.resolve()))
        self._futures.append(f); return f

    def _load_pair(self, layer: int, routed_dir: str, shared_dir: str) -> dict[str, Any]:
        fs=[self._pool.submit(self._send,'routed',f'LOAD {layer} {routed_dir}'),
            self._pool.submit(self._send,'shared',f'LOAD {layer} {shared_dir}')]
        results=[x.result() for x in fs]
        ready=all(x.get('status')=='ok' and x.get('code')=='LAYER_READY' for x in results)
        with self._state_lock: self._states[layer]='READY' if ready else 'ERROR'
        return {'layer_id':layer,'state':'READY' if ready else 'ERROR','routed':results[0],'shared':results[1]}

    def infer(self, request_id: str, layer: int, request_path: Path,
              routed_output: Path, routed_partials: Path,
              shared_output: Path, shared_partials: Path) -> dict[str, Any]:
        if not 0 <= layer < len(self._states):
            return {'request_id':request_id,'status':'error','code':'INVALID_LAYER',
                    'layer_id':layer,'valid_layer_range':[0,len(self._states)-1]}
        with self._state_lock: state=self._states[layer]
        if state == 'ERROR':
            return {'request_id':request_id,'status':'error','code':'LAYER_LOAD_FAILED','layer_id':layer,'retryable':False}
        if state != 'READY':
            return {'request_id':request_id,'status':'error','code':'LAYER_NOT_READY','layer_id':layer,'layer_state':state,'retryable':True}
        cr=f'INFER {request_id} {layer} {request_path.resolve()} {routed_output.resolve()} {routed_partials.resolve()}'
        cs=f'INFER {request_id} {layer} {request_path.resolve()} {shared_output.resolve()} {shared_partials.resolve()}'
        pair_started = time.perf_counter()
        fs=[self._pool.submit(self._timed_send,'routed',cr),self._pool.submit(self._timed_send,'shared',cs)]
        (routed, routed_s), (shared, shared_s) = [x.result() for x in fs]
        return {'request_id':request_id,'layer_id':layer,'routed':routed,'shared':shared,
                'routed_wall_seconds':routed_s,'shared_wall_seconds':shared_s,
                'pair_wall_seconds':time.perf_counter()-pair_started,
                'status':'ok' if routed.get('status')=='ok' and shared.get('status')=='ok' else 'error'}

    def ple_lookup(self, entries: list[tuple[int, str, int, int]], count: int) -> bytes:
        """Gather indexed 160-byte rows through each service's resident kernel.

        entries are (output_index, branch, branch_local_device, byte_offset).
        Both service processes retain their PLE IPC imports for the entire run.
        """
        if count < 0 or count > 65536:
            raise ValueError(f'PLE request count out of range: {count}')
        tag = uuid.uuid4().hex
        paths = {}
        commands = {}
        for branch in ('routed', 'shared'):
            subset = [e for e in entries if e[1] == branch]
            if not subset:
                continue
            rp = self._scratch / f'{tag}.{branch}.req'
            op = self._scratch / f'{tag}.{branch}.out'
            with rp.open('wb') as f:
                f.write(struct.pack('<I', count))
                for output_index, _branch, device, offset in subset:
                    f.write(struct.pack('<IIQ', output_index, device, offset))
            if any(ch.isspace() for ch in str(rp)) or any(ch.isspace() for ch in str(op)):
                raise ValueError('PLE scratch paths cannot contain whitespace')
            paths[branch] = (rp, op, subset)
            commands[branch] = f'PLE {rp.resolve()} {op.resolve()}'
        futures = {b: self._pool.submit(self._send, b, c) for b, c in commands.items()}
        responses = {b: f.result() for b, f in futures.items()}
        result = bytearray(count * 160)
        for branch, response in responses.items():
            if response.get('status') != 'ok' or response.get('code') != 'PLE_READY':
                raise RuntimeError(f'{branch} resident PLE failed: {response}')
            _rp, op, subset = paths[branch]
            data = op.read_bytes()
            if len(data) != count * 160:
                raise RuntimeError(f'{branch} PLE output length {len(data)} != {count * 160}')
            for output_index, _branch, _device, _offset in subset:
                begin = output_index * 160
                result[begin:begin+160] = data[begin:begin+160]
            paths[branch][0].unlink(missing_ok=True)
            op.unlink(missing_ok=True)
        return bytes(result)

    def close(self):
        self._pool.shutdown(wait=True)
        for branch,p in [('routed',self._r),('shared',self._s)]:
            try:
                self._send(branch,'STOP'); p.stdin.close(); p.wait(timeout=30)
            except Exception:
                p.terminate(); p.wait(timeout=10)
        self._rlog.close(); self._slog.close()
